"""Synthetic RFQ history that demonstrates RFQ-responder economics before live data."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

STUDENT_T_DF = 5.0
TRADING_DAYS = 252
SECTORS: tuple[str, ...] = (
    "Technology",
    "Financials",
    "Health Care",
    "Industrials",
    "Consumer",
    "Energy",
    "Utilities",
    "Communication",
)
RATING_BUCKETS: tuple[str, ...] = ("AAA", "AA", "A", "BBB", "BB", "HY")
CLIENT_TIERS: tuple[str, ...] = ("retail", "professional", "informational")
REGIMES: tuple[str, ...] = ("calm", "normal", "volatile")
VENUES: tuple[str, ...] = ("tradeweb", "marketaxess", "bloomberg_rfq")

SECONDS_PER_DAY = 86_400
FIRST_TRADE_STALENESS_SECONDS = 30.0 * SECONDS_PER_DAY

# Data-generating-process constants shared with the synthetic oracle so that
# diagnostics stay consistent with the simulation without duplicating numbers.
FUTURE_NOISE_SCALE = 0.20
CLIENT_INFO_NOISE_STD = 0.05
MARKOUT_SCALE_BY_REGIME: dict[str, float] = {
    "calm": 0.85,
    "normal": 1.00,
    "volatile": 1.35,
}

OBSERVABLE_COLUMNS: tuple[str, ...] = (
    "rfq_id",
    "timestamp",
    "bond_id",
    "issuer_id",
    "client_id",
    "venue",
    "sector",
    "rating_bucket",
    "client_tier",
    "regime",
    "side",
    "side_sign",
    "cp_plus",
    "internal_mid",
    "quote",
    "won",
    "y5",
    "size",
    "liquidity_score",
    "market_width",
    "volatility",
    "inventory",
    "market_signal",
    "issuer_signal",
    "number_of_dealers",
    "quote_deadline_ms",
    "bond_age_days",
    "time_since_last_trade_seconds",
    "recent_trade_count",
    "recent_market_move",
    "recent_issuer_move",
    "is_inventory_axe",
)

LATENT_COLUMNS: tuple[str, ...] = (
    "latent_mu_value",
    "latent_future_residual",
    "latent_client_information",
    "latent_information_strength",
    "latent_aggressiveness",
    "latent_p_win",
)


class Side(Enum):
    """Client side on the RFQ ticket used by legacy modeling adapters."""

    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class SyntheticConfig:
    """Controls the scale and calibration of the synthetic RFQ generator.

    :param n_rfqs: Target number of RFQ rows to generate.
    :param n_bonds: Number of bonds in the simulated market.
    :param n_issuers: Number of issuers in the simulated market.
    :param n_clients: Number of clients sending RFQs; each sends many RFQs.
    :param trading_days: Number of business days spanned by the simulation.
    :param student_t_df: Degrees of freedom for heavy-tailed innovations.
    :param win_intercept: Logistic intercept for the win model.
    :param win_aggressiveness_coef: Logistic coefficient on normalized aggressiveness.
    :param win_information_coef: Logistic coefficient on hidden client information.
    :param win_dealer_coef: Win-logit penalty per dealer above the typical panel.
    :param activity_gamma_shape: Shape parameter for bond activity weights.
    :param activity_gamma_scale: Scale parameter for bond activity weights.
    :param recent_window_days: Lookback window for recent move and flow features.
    :param deadline_noise_scale: Aggressiveness noise scale for short deadlines.
    :param axe_aggressiveness_boost: Extra dealer aggressiveness on inventory axes.
    """

    n_rfqs: int = 15_000
    n_bonds: int = 300
    n_issuers: int = 60
    n_clients: int = 150
    trading_days: int = TRADING_DAYS
    student_t_df: float = STUDENT_T_DF
    win_intercept: float = -1.05
    win_aggressiveness_coef: float = 1.8
    win_information_coef: float = 0.8
    win_dealer_coef: float = 0.28
    activity_gamma_shape: float = 0.55
    activity_gamma_scale: float = 28.0
    recent_window_days: int = 5
    deadline_noise_scale: float = 0.22
    axe_aggressiveness_boost: float = 0.35


@dataclass(frozen=True)
class IssuerInfo:
    """Issuer metadata attached to a legacy synthetic RFQ row."""

    issuer_id: str
    issuer_name: str
    sector: str
    rating: str


@dataclass(frozen=True)
class BondInfo:
    """Bond metadata attached to a legacy synthetic RFQ row."""

    bond_id: str
    issuer: IssuerInfo
    coupon_pct: float
    maturity_year: int


@dataclass(frozen=True)
class SyntheticRfq:
    """Legacy row wrapper used by the small-book demo helpers."""

    rfq_id: str
    bond: BondInfo
    side: Side
    cp_plus_mid: float
    internal_mid: float
    quote: float
    quote_won: bool
    t5_clean_mark: float


@dataclass(frozen=True)
class RfqRequest:
    """Modeling input derived from a synthetic or live RFQ."""

    rfq_id: str
    side: Side
    quantity: float
    mid_price: float
    volatility: float
    inventory: float
    time_to_hedge: float
    competition_count: int


@dataclass(frozen=True)
class SyntheticBookSpec:
    """Legacy book specification retained for notebook and pipeline helpers."""

    n_rfqs: int = 40
    seed: int = 7
    cp_plus_mid_mean: float = 100.0
    cp_plus_mid_std: float = 2.5
    internal_mid_noise_bps: float = 1.5
    quote_edge_min_bps: float = 1.0
    quote_edge_max_bps: float = 12.0
    t5_daily_vol_bps: float = 6.0
    win_intercept: float = 1.8
    win_edge_coef: float = 0.35
    selection_bps_on_win: float = 2.5
    log_quantity_mean: float = field(default=math.log(2_000.0))
    log_quantity_std: float = 0.6
    vol_min: float = 0.12
    vol_max: float = 0.35
    inventory_std: float = 800.0
    hedge_horizon_min: float = 1.0 / 252.0
    hedge_horizon_max: float = 5.0 / 252.0
    max_competition: int = 8


@dataclass(frozen=True)
class MarketStructure:
    """Static issuer, bond, client, and regime objects used to simulate RFQs."""

    issuer_ids: np.ndarray
    issuer_sectors: np.ndarray
    issuer_ratings: np.ndarray
    issuer_value_effect: np.ndarray
    issuer_liquidity: np.ndarray
    issuer_spread: np.ndarray
    issuer_signal_daily: np.ndarray
    bond_ids: np.ndarray
    bond_issuer_idx: np.ndarray
    bond_base_price: np.ndarray
    bond_value_effect: np.ndarray
    bond_liquidity_adj: np.ndarray
    bond_activity_weight: np.ndarray
    bond_age_at_start_days: np.ndarray
    client_ids: np.ndarray
    client_tier: np.ndarray
    client_information_shift: np.ndarray
    client_aggressiveness_bias: np.ndarray
    client_activity_weight: np.ndarray
    regime_daily: np.ndarray
    cp_plus_daily: np.ndarray
    market_signal_daily: np.ndarray


def make_synthetic_rfqs(
    config: SyntheticConfig | None = None,
    random_state: int = 42,
    include_latent: bool = False,
) -> pd.DataFrame:
    """Generate a reproducible synthetic RFQ history with explicit economics.

    The simulation creates temporally dependent CP+ prices, latent future
    value, imperfect internal marks, hidden client information that drives
    adverse selection, and quotes that affect win probability but not y5.

    :param config: Simulation scale and calibration. Defaults to 15k RFQs.
    :param random_state: Seed for the numpy Generator.
    :param include_latent: Whether to append diagnostic latent_* columns.
    :return: Chronologically ordered RFQ dataframe.
    :raises ValueError: If configuration counts are invalid.
    """

    simulation_config = config or SyntheticConfig()
    _validate_config(simulation_config)
    rng = np.random.default_rng(random_state)
    market = _build_market_structure(rng, simulation_config)
    rfq_state = _build_rfq_state(rng, simulation_config, market)
    economics = _simulate_rfq_economics(rng, simulation_config, market, rfq_state)
    return _assemble_output_frame(
        rfq_state=rfq_state,
        economics=economics,
        include_latent=include_latent,
    )


def validate_synthetic_data(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize whether synthetic data exhibit the intended economics.

    Realized selection uses the internal mid as the pre-trade value mark V0:

        selection = side_sign * (internal_mid - y5)

    :param df: Output from :func:`make_synthetic_rfqs`.
    :return: Validation metrics and pass/fail flags for key relationships.
    :raises ValueError: If required columns are missing.
    """

    _require_columns(df, OBSERVABLE_COLUMNS)
    aggressiveness = df["side_sign"] * (df["quote"] - df["cp_plus"]) / df["market_width"]
    selection = df["side_sign"] * (df["internal_mid"] - df["y5"])
    cp_plus_error = np.abs(df["cp_plus"] - df["y5"])
    internal_error = np.abs(df["internal_mid"] - df["y5"])
    bond_counts = df.groupby("bond_id").size()
    fill_counts = df.loc[df["won"]].groupby("bond_id").size()
    aggressiveness_bucket = pd.qcut(aggressiveness, q=5, duplicates="drop")
    win_by_bucket = df.groupby(aggressiveness_bucket, observed=False)["won"].mean()
    monotonic_buckets = win_by_bucket.is_monotonic_increasing

    overall_win_rate = float(df["won"].mean())
    side_win_rates = df.groupby("side")["won"].mean().to_dict()
    client_counts = df.groupby("client_id").size()
    dealer_win_correlation = float(
        df["number_of_dealers"].astype(float).corr(df["won"].astype(float))
    )
    log_staleness = np.log1p(df["time_since_last_trade_seconds"] / SECONDS_PER_DAY)
    staleness_width_correlation = float(log_staleness.corr(df["market_width"]))
    axe_aggressiveness_gap = float(
        aggressiveness.loc[df["is_inventory_axe"]].mean()
        - aggressiveness.loc[~df["is_inventory_axe"]].mean()
    )
    volatile_width = float(df.loc[df["regime"] == "volatile", "market_width"].mean())
    calm_width = float(df.loc[df["regime"] == "calm", "market_width"].mean())
    volatile_markout = float(
        np.abs(
            df.loc[df["regime"] == "volatile", "y5"]
            - df.loc[df["regime"] == "volatile", "cp_plus"]
        ).mean()
    )
    calm_markout = float(
        np.abs(
            df.loc[df["regime"] == "calm", "y5"] - df.loc[df["regime"] == "calm", "cp_plus"]
        ).mean()
    )
    return {
        "n_rfqs": int(len(df)),
        "n_bonds": int(df["bond_id"].nunique()),
        "n_issuers": int(df["issuer_id"].nunique()),
        "n_clients": int(df["client_id"].nunique()),
        "overall_win_rate": overall_win_rate,
        "win_rate_by_side": {str(key): float(value) for key, value in side_win_rates.items()},
        "median_rfqs_per_bond": float(bond_counts.median()),
        "median_fills_per_bond": float(fill_counts.median()) if not fill_counts.empty else 0.0,
        "median_rfqs_per_client": float(client_counts.median()),
        "aggressiveness_win_correlation": float(aggressiveness.corr(df["won"].astype(float))),
        "cp_plus_mae_vs_y5": float(cp_plus_error.mean()),
        "internal_mid_mae_vs_y5": float(internal_error.mean()),
        "internal_mid_more_informative_than_cp_plus": bool(
            internal_error.mean() < cp_plus_error.mean()
        ),
        "mean_selection_all": float(selection.mean()),
        "mean_selection_wins": float(selection.loc[df["won"]].mean()),
        "aggressiveness_bucket_win_rates_monotone": bool(monotonic_buckets),
        "win_rate_in_target_band": bool(0.20 <= overall_win_rate <= 0.50),
        "bond_activity_sparse": bool(bond_counts.median() <= 80.0 and bond_counts.max() >= 100.0),
        "dealer_win_correlation": dealer_win_correlation,
        "more_dealers_reduce_win_probability": bool(dealer_win_correlation < 0.0),
        "staleness_width_correlation": staleness_width_correlation,
        "stale_bonds_have_wider_markets": bool(staleness_width_correlation > 0.0),
        "axe_aggressiveness_gap": axe_aggressiveness_gap,
        "axes_quote_more_aggressively": bool(axe_aggressiveness_gap > 0.0),
        "volatile_vs_calm_width_ratio": _safe_ratio(volatile_width, calm_width),
        "volatile_vs_calm_markout_ratio": _safe_ratio(volatile_markout, calm_markout),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Divide safely, returning NaN when either regime slice is empty.

    :param numerator: Ratio numerator.
    :param denominator: Ratio denominator.
    :return: Finite ratio, or NaN when inputs are missing or non-positive.
    """

    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return float("nan")
    if denominator <= 0.0:
        return float("nan")
    return numerator / denominator


def demo_book_spec() -> SyntheticBookSpec:
    """Return the legacy small-book specification used in quick demos."""

    return SyntheticBookSpec()


def generate_rfq_book(spec: SyntheticBookSpec | None = None) -> tuple[SyntheticRfq, ...]:
    """Draw a reproducible legacy book from the economic simulator.

    :param spec: Optional legacy specification. Defaults to :func:`demo_book_spec`.
    :return: Tuple of legacy RFQ rows.
    """

    book_spec = spec or demo_book_spec()
    config = SyntheticConfig(n_rfqs=book_spec.n_rfqs)
    frame = make_synthetic_rfqs(config=config, random_state=book_spec.seed)
    return tuple(_row_to_synthetic_rfq(row) for _, row in frame.iterrows())


def to_rfq_request(
    record: SyntheticRfq,
    spec: SyntheticBookSpec | None = None,
    index: int = 0,
) -> RfqRequest:
    """Map a legacy synthetic row to the modeling input used by the pipeline.

    :param record: Legacy synthetic RFQ row.
    :param spec: Legacy book specification used for latent feature draws.
    :param index: Row index inside the legacy book.
    :return: Modeling input for value, fill, selection, and cost modules.
    """

    book_spec = spec or demo_book_spec()
    rng = np.random.default_rng(book_spec.seed * 1_000_003 + index)
    quantity = float(math.exp(rng.normal(book_spec.log_quantity_mean, book_spec.log_quantity_std)))
    volatility = float(rng.uniform(book_spec.vol_min, book_spec.vol_max))
    inventory = float(rng.normal(0.0, book_spec.inventory_std))
    time_to_hedge = float(rng.uniform(book_spec.hedge_horizon_min, book_spec.hedge_horizon_max))
    competition_count = int(rng.integers(1, book_spec.max_competition + 1))
    return RfqRequest(
        rfq_id=record.rfq_id,
        side=record.side,
        quantity=quantity,
        mid_price=record.internal_mid,
        volatility=volatility,
        inventory=inventory,
        time_to_hedge=time_to_hedge,
        competition_count=competition_count,
    )


def generate_modeling_book(spec: SyntheticBookSpec | None = None) -> tuple[RfqRequest, ...]:
    """Draw synthetic RFQs and convert them to modeling inputs.

    :param spec: Optional legacy specification.
    :return: Modeling requests aligned with the synthetic book order.
    """

    records = generate_rfq_book(spec)
    book_spec = spec or demo_book_spec()
    return tuple(
        to_rfq_request(record, book_spec, index)
        for index, record in enumerate(records)
    )


def _validate_config(config: SyntheticConfig) -> None:
    if config.n_rfqs < 1:
        raise ValueError("n_rfqs must be at least 1")
    if config.n_bonds < 1:
        raise ValueError("n_bonds must be at least 1")
    if config.n_issuers < 1:
        raise ValueError("n_issuers must be at least 1")
    if config.n_clients < 1:
        raise ValueError("n_clients must be at least 1")
    if config.trading_days < 5:
        raise ValueError("trading_days must be at least 5")
    if config.recent_window_days < 1:
        raise ValueError("recent_window_days must be at least 1")


def _build_market_structure(rng: np.random.Generator, config: SyntheticConfig) -> MarketStructure:
    issuer_ids = np.array([f"issuer-{index:03d}" for index in range(config.n_issuers)])
    issuer_sectors = rng.choice(np.array(SECTORS), size=config.n_issuers)
    rating_probs = np.array([0.05, 0.10, 0.20, 0.25, 0.20, 0.20])
    issuer_ratings = rng.choice(np.array(RATING_BUCKETS), size=config.n_issuers, p=rating_probs)
    issuer_value_effect = rng.normal(0.0, 0.12, size=config.n_issuers)
    issuer_liquidity = rng.beta(2.0, 2.0, size=config.n_issuers)
    hy_penalty = np.isin(issuer_ratings, np.array(["BB", "HY"])).astype(float) * 0.08
    issuer_spread = 0.10 + 0.18 * (1.0 - issuer_liquidity) + hy_penalty

    bond_ids = np.array([f"bond-{index:04d}" for index in range(config.n_bonds)])
    bond_issuer_idx = rng.integers(0, config.n_issuers, size=config.n_bonds)
    bond_base_price = rng.normal(100.0, 4.0, size=config.n_bonds)
    bond_value_effect = rng.normal(0.0, 0.08, size=config.n_bonds)
    bond_liquidity_adj = rng.normal(0.0, 0.10, size=config.n_bonds)
    raw_activity = rng.gamma(
        config.activity_gamma_shape,
        config.activity_gamma_scale,
        size=config.n_bonds,
    )
    bond_activity_weight = raw_activity / raw_activity.sum()
    # Age from one month to roughly ten years past issuance at simulation start.
    bond_age_at_start_days = rng.integers(30, 3_600, size=config.n_bonds).astype(float)

    client_ids = np.array([f"client-{index:04d}" for index in range(config.n_clients)])
    client_tier = rng.choice(
        np.array(CLIENT_TIERS),
        size=config.n_clients,
        p=np.array([0.45, 0.40, 0.15]),
    )
    # Within-tier heterogeneity: individual clients deviate from tier averages
    # in both how informed they are and how aggressively they push dealers.
    client_information_shift = rng.normal(0.0, 0.15, size=config.n_clients)
    client_aggressiveness_bias = rng.normal(0.0, 0.20, size=config.n_clients)
    raw_client_activity = rng.gamma(0.8, 10.0, size=config.n_clients)
    client_activity_weight = raw_client_activity / raw_client_activity.sum()

    regime_daily = _simulate_regime_path(rng, config.trading_days)

    cp_plus_daily, market_signal_daily, issuer_signal_daily = _simulate_cp_plus_paths(
        rng=rng,
        config=config,
        bond_base_price=bond_base_price,
        bond_issuer_idx=bond_issuer_idx,
        issuer_value_effect=issuer_value_effect,
    )
    return MarketStructure(
        issuer_ids=issuer_ids,
        issuer_sectors=issuer_sectors,
        issuer_ratings=issuer_ratings,
        issuer_value_effect=issuer_value_effect,
        issuer_liquidity=issuer_liquidity,
        issuer_spread=issuer_spread,
        issuer_signal_daily=issuer_signal_daily,
        bond_ids=bond_ids,
        bond_issuer_idx=bond_issuer_idx,
        bond_base_price=bond_base_price,
        bond_value_effect=bond_value_effect,
        bond_liquidity_adj=bond_liquidity_adj,
        bond_activity_weight=bond_activity_weight,
        bond_age_at_start_days=bond_age_at_start_days,
        client_ids=client_ids,
        client_tier=client_tier,
        client_information_shift=client_information_shift,
        client_aggressiveness_bias=client_aggressiveness_bias,
        client_activity_weight=client_activity_weight,
        regime_daily=regime_daily,
        cp_plus_daily=cp_plus_daily,
        market_signal_daily=market_signal_daily,
    )


def _simulate_regime_path(rng: np.random.Generator, days: int) -> np.ndarray:
    """Simulate a persistent daily volatility regime via a Markov chain.

    :param rng: Seeded numpy Generator.
    :param days: Number of trading days to simulate.
    :return: Regime label per day.
    """

    transition_by_regime = {
        "calm": np.array([0.90, 0.10, 0.00]),
        "normal": np.array([0.06, 0.88, 0.06]),
        "volatile": np.array([0.00, 0.15, 0.85]),
    }
    regimes = np.empty(days, dtype=object)
    current = "normal"
    for day in range(days):
        regimes[day] = current
        probabilities = transition_by_regime[current]
        current = REGIMES[int(rng.choice(3, p=probabilities))]
    return regimes.astype(str)


def _simulate_cp_plus_paths(
    rng: np.random.Generator,
    config: SyntheticConfig,
    bond_base_price: np.ndarray,
    bond_issuer_idx: np.ndarray,
    issuer_value_effect: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    days = config.trading_days
    n_bonds = config.n_bonds
    n_issuers = config.n_issuers
    scale = _student_t_scale(config.student_t_df)

    market_shock = rng.standard_t(config.student_t_df, size=days) * scale * 0.08
    market_level = np.cumsum(market_shock)
    market_signal_daily = market_level + rng.normal(0.0, 0.03, size=days)

    issuer_shock = rng.standard_t(config.student_t_df, size=(n_issuers, days)) * scale * 0.05
    issuer_level = np.cumsum(issuer_shock, axis=1)
    issuer_signal_daily = issuer_level + rng.normal(0.0, 0.02, size=(n_issuers, days))

    bond_shock = rng.standard_t(config.student_t_df, size=(n_bonds, days)) * scale * 0.04
    bond_level = np.cumsum(bond_shock, axis=1)
    issuer_component = issuer_level[bond_issuer_idx, :]
    cp_plus_daily = (
        bond_base_price[:, None]
        + market_level[None, :]
        + issuer_component
        + bond_level
        + issuer_value_effect[bond_issuer_idx][:, None] * 0.20
    )
    return cp_plus_daily, market_signal_daily, issuer_signal_daily


def _build_rfq_state(
    rng: np.random.Generator,
    config: SyntheticConfig,
    market: MarketStructure,
) -> dict[str, np.ndarray]:
    bond_idx = _sample_bond_indices(rng, config, market.bond_activity_weight)
    day_idx = rng.integers(0, config.trading_days, size=config.n_rfqs)
    # Intra-day arrival between 08:00 and 17:00 so staleness features carry
    # information at second resolution rather than only at day resolution.
    second_of_day = rng.integers(8 * 3_600, 17 * 3_600, size=config.n_rfqs)
    order = np.lexsort((second_of_day, day_idx))
    bond_idx = bond_idx[order]
    day_idx = day_idx[order]
    second_of_day = second_of_day[order]

    issuer_idx = market.bond_issuer_idx[bond_idx]
    client_idx = rng.choice(
        config.n_clients,
        size=config.n_rfqs,
        replace=True,
        p=market.client_activity_weight,
    )
    side_is_dealer_buy = rng.random(config.n_rfqs) < 0.5
    side_sign = np.where(side_is_dealer_buy, 1, -1)
    side_label = np.where(side_is_dealer_buy, "dealer_buy", "dealer_sell")

    liquidity_score = rng.beta(1.5, 1.5, size=config.n_rfqs)
    liquidity_score = np.clip(
        liquidity_score + (1.0 - market.issuer_liquidity[issuer_idx]) * 0.15,
        0.05,
        0.95,
    )
    size = rng.lognormal(mean=math.log(2_000.0), sigma=0.55, size=config.n_rfqs)
    volatility = rng.lognormal(mean=math.log(0.18), sigma=0.35, size=config.n_rfqs)
    inventory = rng.normal(0.0, 900.0, size=config.n_rfqs)
    client_tier = market.client_tier[client_idx]
    regime = market.regime_daily[day_idx]

    venue = rng.choice(
        np.array(VENUES),
        size=config.n_rfqs,
        p=np.array([0.45, 0.40, 0.15]),
    )
    # Larger venues and liquid bonds attract wider dealer panels.
    venue_panel_base = np.select(
        [venue == "tradeweb", venue == "marketaxess"],
        [4.0, 3.5],
        default=3.0,
    )
    dealer_intensity = venue_panel_base + 2.0 * liquidity_score
    number_of_dealers = np.clip(1 + rng.poisson(dealer_intensity), 2, 12)

    deadline_regime_factor = np.select(
        [regime == "calm", regime == "volatile"],
        [1.20, 0.70],
        default=1.0,
    )
    quote_deadline_ms = np.clip(
        rng.lognormal(mean=math.log(45_000.0), sigma=0.8, size=config.n_rfqs)
        * deadline_regime_factor,
        3_000.0,
        300_000.0,
    )

    bond_age_days = market.bond_age_at_start_days[bond_idx] + day_idx.astype(float)
    timestamp_seconds = day_idx.astype(np.int64) * SECONDS_PER_DAY + second_of_day
    time_since_last_trade_seconds, recent_trade_count = _bond_flow_history(
        bond_idx=bond_idx,
        timestamp_seconds=timestamp_seconds,
        window_seconds=config.recent_window_days * SECONDS_PER_DAY,
    )

    window = config.recent_window_days
    previous_day = np.maximum(day_idx - window, 0)
    recent_market_move = (
        market.market_signal_daily[day_idx] - market.market_signal_daily[previous_day]
    )
    recent_issuer_move = (
        market.issuer_signal_daily[issuer_idx, day_idx]
        - market.issuer_signal_daily[issuer_idx, previous_day]
    )

    # Axes concentrate on RFQs that reduce absolute dealer inventory.
    inventory_delta = np.where(side_sign > 0, size, -size)
    reduces_inventory = np.abs(inventory + inventory_delta) < np.abs(inventory)
    axe_probability = np.where(reduces_inventory, 0.30, 0.05)
    is_inventory_axe = rng.random(config.n_rfqs) < axe_probability

    rating_bucket = market.issuer_ratings[issuer_idx]
    is_hy = np.isin(rating_bucket, np.array(["BB", "HY"]))
    regime_multiplier = np.select(
        [regime == "calm", regime == "normal", regime == "volatile"],
        [0.85, 1.00, 1.35],
        default=1.0,
    )
    staleness_term = 0.03 * np.log1p(time_since_last_trade_seconds / SECONDS_PER_DAY)
    age_term = 0.015 * np.log1p(bond_age_days / 365.0)
    market_width = (
        market.issuer_spread[issuer_idx]
        + market.bond_liquidity_adj[bond_idx]
        + 0.06 * (1.0 - liquidity_score)
        + 0.05 * np.log1p(size / 1_000.0)
        + 0.10 * volatility
        + 0.04 * is_hy.astype(float)
        + staleness_term
        + age_term
    )
    market_width = np.clip(market_width * regime_multiplier, 0.08, 1.50)

    timestamps = pd.to_datetime("2025-01-01") + pd.to_timedelta(timestamp_seconds, unit="s")
    rfq_ids = np.array([f"rfq-{index:06d}" for index in range(config.n_rfqs)])

    return {
        "rfq_id": rfq_ids,
        "timestamp": timestamps.to_numpy(),
        "bond_idx": bond_idx,
        "issuer_idx": issuer_idx,
        "client_idx": client_idx,
        "bond_id": market.bond_ids[bond_idx],
        "issuer_id": market.issuer_ids[issuer_idx],
        "client_id": market.client_ids[client_idx],
        "venue": venue,
        "sector": market.issuer_sectors[issuer_idx],
        "rating_bucket": rating_bucket,
        "client_tier": client_tier,
        "regime": regime,
        "side": side_label,
        "side_sign": side_sign.astype(float),
        "day_idx": day_idx,
        "size": size,
        "liquidity_score": liquidity_score,
        "market_width": market_width,
        "volatility": volatility,
        "inventory": inventory,
        "number_of_dealers": number_of_dealers.astype(int),
        "quote_deadline_ms": quote_deadline_ms,
        "bond_age_days": bond_age_days,
        "time_since_last_trade_seconds": time_since_last_trade_seconds,
        "recent_trade_count": recent_trade_count,
        "recent_market_move": recent_market_move,
        "recent_issuer_move": recent_issuer_move,
        "is_inventory_axe": is_inventory_axe,
    }


def _bond_flow_history(
    bond_idx: np.ndarray,
    timestamp_seconds: np.ndarray,
    window_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute point-in-time staleness and recent flow per bond.

    Both outputs use strictly prior RFQs on the same bond, so they are safe
    to expose to models at quote time.

    :param bond_idx: Bond index per RFQ in chronological order.
    :param timestamp_seconds: RFQ arrival time in seconds, chronological.
    :param window_seconds: Trailing window for the recent trade count.
    :return: Seconds since previous same-bond RFQ, and prior-window RFQ count.
    """

    n_rfqs = len(bond_idx)
    time_since = np.full(n_rfqs, FIRST_TRADE_STALENESS_SECONDS, dtype=float)
    recent_count = np.zeros(n_rfqs, dtype=int)
    frame = pd.DataFrame({"bond": bond_idx, "time": timestamp_seconds})
    for _, group in frame.groupby("bond", sort=False):
        positions = group.index.to_numpy()
        times = group["time"].to_numpy()
        if len(times) > 1:
            time_since[positions[1:]] = (times[1:] - times[:-1]).astype(float)
        window_start = np.searchsorted(times, times - window_seconds, side="left")
        recent_count[positions] = np.arange(len(times)) - window_start
    return time_since, recent_count


def _simulate_rfq_economics(
    rng: np.random.Generator,
    config: SyntheticConfig,
    market: MarketStructure,
    rfq_state: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    bond_idx = rfq_state["bond_idx"]
    issuer_idx = rfq_state["issuer_idx"]
    day_idx = rfq_state["day_idx"]
    cp_plus = market.cp_plus_daily[bond_idx, day_idx]
    market_signal = market.market_signal_daily[day_idx]
    issuer_signal = market.issuer_signal_daily[issuer_idx, day_idx]

    regime = rfq_state["regime"]
    regime_effect = np.select(
        [regime == "calm", regime == "normal", regime == "volatile"],
        [-0.02, 0.0, 0.05],
        default=0.0,
    )
    mu_value = (
        market.issuer_value_effect[issuer_idx] * 0.35
        + market.bond_value_effect[bond_idx] * 0.30
        + regime_effect
        + 0.08 * (rfq_state["volatility"] - 0.18)
        + 0.05 * (0.5 - rfq_state["liquidity_score"])
        + 0.10 * market_signal
        + 0.08 * issuer_signal
        + 0.05 * rfq_state["recent_market_move"]
        + 0.04 * rfq_state["recent_issuer_move"]
    )

    # Volatile regimes carry heavier markout tails, not only wider quoted markets.
    markout_scale = np.select(
        [regime == "calm", regime == "volatile"],
        [MARKOUT_SCALE_BY_REGIME["calm"], MARKOUT_SCALE_BY_REGIME["volatile"]],
        default=MARKOUT_SCALE_BY_REGIME["normal"],
    )
    future_noise = (
        _student_t_draw(rng, config.n_rfqs, config.student_t_df, scale=FUTURE_NOISE_SCALE)
        * markout_scale
    )
    future_residual = mu_value + future_noise
    y5 = cp_plus + future_residual

    internal_noise = rng.normal(0.0, 0.03, size=config.n_rfqs)
    regime_bias = np.select(
        [regime == "calm", regime == "volatile"],
        [0.012, -0.020],
        default=0.0,
    )
    partially_informative_signal = 0.08 * market_signal + 0.05 * issuer_signal
    internal_signal = mu_value + partially_informative_signal + regime_bias + internal_noise
    internal_mid = cp_plus + internal_signal

    info_strength = _information_strength(rfq_state, market)
    client_information = info_strength * future_residual + rng.normal(
        0.0, CLIENT_INFO_NOISE_STD, size=config.n_rfqs
    )
    standardized_client_information = _standardize(client_information)

    internal_alpha = internal_signal - mu_value * 0.35
    standardized_internal_alpha = _standardize(internal_alpha)
    standardized_inventory = _standardize(rfq_state["inventory"])
    standardized_log_size = _standardize(np.log1p(rfq_state["size"] / 1_000.0))
    tier_effect = np.select(
        [rfq_state["client_tier"] == "retail", rfq_state["client_tier"] == "professional"],
        [-0.05, 0.0],
        default=0.08,
    )
    regime_effect_win = np.select(
        [rfq_state["regime"] == "calm", rfq_state["regime"] == "volatile"],
        [-0.05, 0.10],
        default=0.0,
    )

    # Short deadlines force the desk to quote under time pressure, adding noise
    # to the historical aggressiveness that models must learn through.
    deadline_pressure = np.sqrt(30_000.0 / rfq_state["quote_deadline_ms"])
    deadline_noise = (
        rng.normal(0.0, 1.0, size=config.n_rfqs)
        * config.deadline_noise_scale
        * deadline_pressure
    )
    client_bias = market.client_aggressiveness_bias[rfq_state["client_idx"]]
    axe_boost = config.axe_aggressiveness_boost * rfq_state["is_inventory_axe"].astype(float)

    base_aggressiveness = rng.normal(-0.1, 0.7, size=config.n_rfqs)
    aggressiveness = (
        base_aggressiveness
        + 0.18 * standardized_internal_alpha
        - 0.10 * standardized_inventory
        + tier_effect
        + client_bias
        - 0.08 * standardized_log_size
        + 0.06 * (rfq_state["liquidity_score"] - 0.5)
        + axe_boost
        + deadline_noise
    )
    aggressiveness = np.clip(aggressiveness, -2.2, 2.2)

    quote = cp_plus + rfq_state["side_sign"] * aggressiveness * rfq_state["market_width"]

    # Hidden-information term uses -beta * side_sign * client_information so informed
    # clients trade against the dealer: sellers arrive when future value is lower on
    # dealer-buy RFQs, and buyers arrive when future value is higher on dealer-sell RFQs.
    typical_panel_size = 4.0
    logit_p_win = (
        config.win_intercept
        + config.win_aggressiveness_coef * aggressiveness
        - config.win_information_coef * rfq_state["side_sign"] * standardized_client_information
        - config.win_dealer_coef * (rfq_state["number_of_dealers"].astype(float) - typical_panel_size)
        + tier_effect
        + 0.30 * rfq_state["liquidity_score"]
        - 0.15 * standardized_log_size
        + regime_effect_win
    )
    p_win = _sigmoid(logit_p_win)
    won = rng.random(config.n_rfqs) < p_win

    return {
        "cp_plus": cp_plus,
        "internal_mid": internal_mid,
        "quote": quote,
        "won": won,
        "y5": y5,
        "market_signal": market_signal,
        "issuer_signal": issuer_signal,
        "latent_mu_value": mu_value,
        "latent_future_residual": future_residual,
        "latent_client_information": client_information,
        "latent_information_strength": info_strength,
        "latent_aggressiveness": aggressiveness,
        "latent_p_win": p_win,
    }


def _assemble_output_frame(
    rfq_state: dict[str, np.ndarray],
    economics: dict[str, np.ndarray],
    include_latent: bool,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "rfq_id": rfq_state["rfq_id"],
            "timestamp": rfq_state["timestamp"],
            "bond_id": rfq_state["bond_id"],
            "issuer_id": rfq_state["issuer_id"],
            "client_id": rfq_state["client_id"],
            "venue": rfq_state["venue"],
            "sector": rfq_state["sector"],
            "rating_bucket": rfq_state["rating_bucket"],
            "client_tier": rfq_state["client_tier"],
            "regime": rfq_state["regime"],
            "side": rfq_state["side"],
            "side_sign": rfq_state["side_sign"],
            "cp_plus": economics["cp_plus"],
            "internal_mid": economics["internal_mid"],
            "quote": economics["quote"],
            "won": economics["won"],
            "y5": economics["y5"],
            "size": rfq_state["size"],
            "liquidity_score": rfq_state["liquidity_score"],
            "market_width": rfq_state["market_width"],
            "volatility": rfq_state["volatility"],
            "inventory": rfq_state["inventory"],
            "market_signal": economics["market_signal"],
            "issuer_signal": economics["issuer_signal"],
            "number_of_dealers": rfq_state["number_of_dealers"],
            "quote_deadline_ms": rfq_state["quote_deadline_ms"],
            "bond_age_days": rfq_state["bond_age_days"],
            "time_since_last_trade_seconds": rfq_state["time_since_last_trade_seconds"],
            "recent_trade_count": rfq_state["recent_trade_count"],
            "recent_market_move": rfq_state["recent_market_move"],
            "recent_issuer_move": rfq_state["recent_issuer_move"],
            "is_inventory_axe": rfq_state["is_inventory_axe"],
        }
    )
    if include_latent:
        for column in LATENT_COLUMNS:
            frame[column] = economics[column]
    return frame


def _sample_bond_indices(
    rng: np.random.Generator,
    config: SyntheticConfig,
    activity_weight: np.ndarray,
) -> np.ndarray:
    """Sample RFQ bond assignments with guaranteed coverage and heavy tails.

    Every bond receives at least one RFQ when the book is large enough. Remaining
    draws follow the heavy-tailed activity weights so a few names dominate flow.

    :param rng: Seeded numpy Generator.
    :param config: Simulation configuration.
    :param activity_weight: Normalized bond activity probabilities.
    :return: Bond index for each RFQ before chronological sorting.
    """

    n_bonds = config.n_bonds
    if config.n_rfqs >= n_bonds:
        guaranteed = np.arange(n_bonds, dtype=int)
        remaining = config.n_rfqs - n_bonds
        extra = rng.choice(n_bonds, size=remaining, replace=True, p=activity_weight)
        bond_idx = np.concatenate([guaranteed, extra])
        rng.shuffle(bond_idx)
        return bond_idx
    return rng.choice(n_bonds, size=config.n_rfqs, replace=False, p=activity_weight)


def _information_strength(
    rfq_state: dict[str, np.ndarray],
    market: MarketStructure,
) -> np.ndarray:
    tier_boost = np.select(
        [rfq_state["client_tier"] == "retail", rfq_state["client_tier"] == "professional"],
        [0.35, 0.55],
        default=0.85,
    )
    hy_boost = np.isin(rfq_state["rating_bucket"], np.array(["BB", "HY"])).astype(float) * 0.15
    illiquid_boost = (0.5 - rfq_state["liquidity_score"]) * 0.25
    volatile_boost = (rfq_state["regime"] == "volatile").astype(float) * 0.10
    client_shift = market.client_information_shift[rfq_state["client_idx"]]
    strength = tier_boost + hy_boost + illiquid_boost + volatile_boost + client_shift
    return np.clip(strength, 0.05, None)


def _row_to_synthetic_rfq(row: pd.Series) -> SyntheticRfq:
    issuer = IssuerInfo(
        issuer_id=str(row["issuer_id"]),
        issuer_name=str(row["issuer_id"]).replace("issuer-", "Issuer "),
        sector=str(row["sector"]),
        rating=str(row["rating_bucket"]),
    )
    bond = BondInfo(
        bond_id=str(row["bond_id"]),
        issuer=issuer,
        coupon_pct=4.0,
        maturity_year=2029,
    )
    side = Side.SELL if row["side"] == "dealer_buy" else Side.BUY
    return SyntheticRfq(
        rfq_id=str(row["rfq_id"]),
        bond=bond,
        side=side,
        cp_plus_mid=float(row["cp_plus"]),
        internal_mid=float(row["internal_mid"]),
        quote=float(row["quote"]),
        quote_won=bool(row["won"]),
        t5_clean_mark=float(row["y5"]),
    )


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _student_t_scale(df: float) -> float:
    if df <= 2.0:
        raise ValueError("student_t_df must exceed 2 for a finite scale")
    return math.sqrt((df - 2.0) / df)


def _student_t_draw(
    rng: np.random.Generator,
    size: int,
    df: float,
    scale: float,
) -> np.ndarray:
    return rng.standard_t(df, size=size) * _student_t_scale(df) * scale


def _standardize(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - mean) / std


def _sigmoid(logit: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-logit))
