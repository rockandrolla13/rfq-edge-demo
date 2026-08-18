"""Exogenous market simulation for the event-driven control environment.

The hidden-information mechanism mirrors the static synthetic market:

* each RFQ carries a latent future driver g ~ N(0, 1) with future clean
  residual epsilon = residual_scale * g;
* the client observes a correlated signal h = rho * g + sqrt(1 - rho^2) * eta;
* the true win logit is alpha + beta * z - kappa * side_sign * h, so
  dealer-buy RFQs fill more easily when future value is poor and dealer-sell
  RFQs fill more easily when future value is strong.

Everything exogenous to the policies (regimes, CP+ path, RFQ arrivals and
attributes, hidden draws, and fill uniforms) is generated once per episode
seed so that different policies face identical paths (common random numbers).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rfq_edge.control_config import (
    CLIENT_TIERS,
    ControlMarketConfig,
    EpisodeConfig,
    MarketRegime,
    REGIME_ORDER,
)
from rfq_edge.control_state import RFQEvent


@dataclass(frozen=True)
class ExogenousPath:
    """All exogenous randomness of one episode, drawn once per seed.

    :param regimes: Regime index per step.
    :param cp_plus: CP+ clean price per step, in points.
    :param events: RFQ event or None per step.
    :param fill_uniforms: Uniform draw per step used to realize fills.
    """

    regimes: np.ndarray
    cp_plus: np.ndarray
    events: tuple[RFQEvent | None, ...]
    fill_uniforms: np.ndarray


def true_win_logit(
    config: ControlMarketConfig,
    regime: MarketRegime,
    aggressiveness: float,
    side_sign: int,
    size: int,
    hidden_client_signal: float,
) -> float:
    """True win logit of the simulated fill mechanism.

    :param config: Market configuration.
    :param regime: Regime at arrival.
    :param aggressiveness: Normalized aggressiveness z of the quote.
    :param side_sign: +1 dealer buy, -1 dealer sell.
    :param size: RFQ size in units.
    :param hidden_client_signal: Latent client signal h.
    :return: Logit of the true fill probability.
    """

    params = config.parameters_for(regime)
    return (
        params.fill_intercept
        + config.fill_aggressiveness_coef * aggressiveness
        - params.information_coefficient * float(side_sign) * hidden_client_signal
        - config.size_win_penalty * float(size - 1)
    )


def true_fill_probability(
    config: ControlMarketConfig,
    event: RFQEvent,
    aggressiveness: float,
) -> float:
    """True fill probability of one event at a candidate aggressiveness.

    :param config: Market configuration.
    :param event: RFQ event carrying the hidden client signal.
    :param aggressiveness: Normalized aggressiveness z of the quote.
    :return: Probability in (0, 1).
    """

    logit = true_win_logit(
        config=config,
        regime=event.regime,
        aggressiveness=aggressiveness,
        side_sign=event.side_sign,
        size=event.size,
        hidden_client_signal=event.hidden_client_signal,
    )
    return 1.0 / (1.0 + math.exp(-logit))


def simulate_exogenous_path(
    market_config: ControlMarketConfig,
    episode_config: EpisodeConfig,
    random_state: int,
) -> ExogenousPath:
    """Simulate the exogenous market and RFQ path of one episode.

    :param market_config: Market configuration.
    :param episode_config: Episode configuration (provides the horizon).
    :param random_state: Seed; identical seeds give identical paths.
    :return: Exogenous path shared by every policy on this episode.
    """

    rng = np.random.default_rng(random_state)
    n_steps = episode_config.n_steps
    transition = np.asarray(market_config.transition_matrix, dtype=float)

    regimes = np.empty(n_steps, dtype=int)
    cp_plus = np.empty(n_steps, dtype=float)
    events: list[RFQEvent | None] = []

    regime_index = market_config.initial_regime.value
    price = market_config.initial_cp_plus
    for step in range(n_steps):
        regimes[step] = regime_index
        cp_plus[step] = price
        regime = REGIME_ORDER[regime_index]
        params = market_config.parameters_for(regime)

        if rng.random() < params.arrival_probability:
            events.append(
                _draw_rfq_event(
                    market_config=market_config,
                    regime=regime,
                    event_id=step,
                    time_index=step,
                    cp_plus=price,
                    rng=rng,
                )
            )
        else:
            events.append(None)

        price = price + params.volatility * rng.standard_normal()
        regime_index = int(rng.choice(len(REGIME_ORDER), p=transition[regime_index]))

    fill_uniforms = rng.random(n_steps)
    return ExogenousPath(
        regimes=regimes,
        cp_plus=cp_plus,
        events=tuple(events),
        fill_uniforms=fill_uniforms,
    )


def generate_training_history(
    market_config: ControlMarketConfig,
    n_events: int,
    random_state: int,
    historical_aggressiveness_std: float = 0.60,
) -> pd.DataFrame:
    """Generate an observable RFQ history for fitting control models.

    Historical quotes follow a clipped random rule so that fill and selection
    models see support across the aggressiveness grid. The returned frame
    contains only observable columns plus realized outcomes; hidden draws are
    integrated out through the realized fills and selection target.

    :param market_config: Market configuration.
    :param n_events: Number of historical RFQs.
    :param random_state: Seed for reproducibility.
    :param historical_aggressiveness_std: Std of the historical quote rule.
    :return: History with regime, side, size, aggressiveness, won, and the
        realized selection target on fills.
    """

    rng = np.random.default_rng(random_state)
    stationary = _stationary_distribution(np.asarray(market_config.transition_matrix))
    grid = market_config.aggressiveness_grid
    records: list[dict[str, object]] = []
    for event_index in range(n_events):
        regime_index = int(rng.choice(len(REGIME_ORDER), p=stationary))
        regime = REGIME_ORDER[regime_index]
        params = market_config.parameters_for(regime)
        side_sign = 1 if rng.random() < 0.5 else -1
        size = int(rng.choice(params.size_values, p=params.size_probabilities))
        aggressiveness = float(
            np.clip(
                rng.normal(0.0, historical_aggressiveness_std),
                min(grid),
                max(grid),
            )
        )
        future_driver = rng.standard_normal()
        rho = params.information_strength
        client_signal = rho * future_driver + math.sqrt(1.0 - rho * rho) * rng.standard_normal()
        future_residual = params.residual_scale * future_driver

        logit = (
            params.fill_intercept
            + market_config.fill_aggressiveness_coef * aggressiveness
            - params.information_coefficient * float(side_sign) * client_signal
            - market_config.size_win_penalty * float(size - 1)
        )
        p_win = 1.0 / (1.0 + math.exp(-logit))
        won = bool(rng.random() < p_win)
        # D = side_sign * (V0 - y5) with V0 = cp and y5 = cp + epsilon.
        realized_selection = -float(side_sign) * future_residual if won else float("nan")
        records.append(
            {
                "event_index": event_index,
                "regime_index": regime_index,
                "regime": regime.name,
                "side_sign": side_sign,
                "size": size,
                "aggressiveness": aggressiveness,
                "won": won,
                "realized_selection_points": realized_selection,
            }
        )
    return pd.DataFrame(records)


def _draw_rfq_event(
    market_config: ControlMarketConfig,
    regime: MarketRegime,
    event_id: int,
    time_index: int,
    cp_plus: float,
    rng: np.random.Generator,
) -> RFQEvent:
    params = market_config.parameters_for(regime)
    side_sign = 1 if rng.random() < 0.5 else -1
    size = int(rng.choice(params.size_values, p=params.size_probabilities))
    tier = str(rng.choice(CLIENT_TIERS, p=market_config.tier_probabilities))
    future_driver = rng.standard_normal()
    rho = params.information_strength
    client_signal = rho * future_driver + math.sqrt(1.0 - rho * rho) * rng.standard_normal()
    future_residual = params.residual_scale * future_driver
    return RFQEvent(
        event_id=event_id,
        time_index=time_index,
        side="dealer_buy" if side_sign == 1 else "dealer_sell",
        side_sign=side_sign,
        size=size,
        client_tier=tier,
        liquidity_score=params.liquidity_score,
        market_width=params.market_width,
        regime=regime,
        cp_plus=cp_plus,
        hidden_client_signal=client_signal,
        hidden_future_residual=future_residual,
    )


def _stationary_distribution(transition: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eig(transition.T)
    index = int(np.argmin(np.abs(eigenvalues - 1.0)))
    stationary = np.real(eigenvectors[:, index])
    stationary = np.abs(stationary)
    return stationary / stationary.sum()
