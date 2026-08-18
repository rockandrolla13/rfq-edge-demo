"""Configuration for the event-driven market-making and execution environment.

Units and conventions used throughout the control layer:

* Prices and quotes are clean prices in points.
* Rewards, costs, and penalties are in cents per normalized inventory unit
  (1 point = 100 cents on a par-100 bond).
* Inventory is measured in normalized integer units; one unit represents
  ``unit_notional_usd`` of notional (default $100,000). The conversion is
  explicit and appears nowhere else.
* ``side_sign = +1`` when the dealer buys from the client, ``-1`` when the
  dealer sells to the client. A filled RFQ of size ``n`` changes dealer
  inventory by ``side_sign * n``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class MarketRegime(Enum):
    """Observable market regime driving liquidity, volatility, and selection."""

    CALM_LIQUID = 0
    NORMAL = 1
    STRESSED_ILLIQUID = 2


REGIME_ORDER: tuple[MarketRegime, ...] = (
    MarketRegime.CALM_LIQUID,
    MarketRegime.NORMAL,
    MarketRegime.STRESSED_ILLIQUID,
)

CLIENT_TIERS: tuple[str, ...] = ("retail", "professional", "informed")
POINTS_TO_CENTS = 100.0


@dataclass(frozen=True)
class RegimeParameters:
    """Per-regime environment parameters.

    :param arrival_probability: Bernoulli RFQ arrival probability per step.
    :param size_values: Possible RFQ sizes in inventory units.
    :param size_probabilities: Probabilities matching ``size_values``.
    :param market_width: Quoted market width in points.
    :param volatility: CP+ innovation standard deviation per step, in points.
    :param liquidity_score: Observable liquidity score in [0, 1].
    :param active_half_spread_cents: Active execution half-spread per unit.
    :param active_impact_cents: Temporary impact coefficient (cents per unit^2).
    :param active_fixed_fee_cents: Fixed fee when active execution is nonzero.
    :param fill_intercept: Intercept of the true win logit.
    :param information_coefficient: Hidden-signal coefficient in the win logit.
    :param residual_scale: Future clean-value residual scale in points.
    :param information_strength: Correlation between the hidden client signal
        and the future residual, in [0, 1).
    """

    arrival_probability: float
    size_values: tuple[int, ...]
    size_probabilities: tuple[float, ...]
    market_width: float
    volatility: float
    liquidity_score: float
    active_half_spread_cents: float
    active_impact_cents: float
    active_fixed_fee_cents: float
    fill_intercept: float
    information_coefficient: float
    residual_scale: float
    information_strength: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.arrival_probability <= 1.0:
            raise ValueError("arrival_probability must lie in [0, 1]")
        if len(self.size_values) != len(self.size_probabilities):
            raise ValueError("size_values and size_probabilities must align")
        if abs(sum(self.size_probabilities) - 1.0) > 1e-9:
            raise ValueError("size_probabilities must sum to one")
        if any(size < 1 for size in self.size_values):
            raise ValueError("RFQ sizes must be at least one unit")
        if self.market_width <= 0.0:
            raise ValueError("market_width must be positive")
        if self.volatility <= 0.0:
            raise ValueError("volatility must be positive")
        if min(
            self.active_half_spread_cents,
            self.active_impact_cents,
            self.active_fixed_fee_cents,
        ) < 0.0:
            raise ValueError("active execution cost components must be non-negative")
        if not 0.0 <= self.information_strength < 1.0:
            raise ValueError("information_strength must lie in [0, 1)")
        if self.residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive")


@dataclass(frozen=True)
class ControlMarketConfig:
    """Market-level configuration shared by every episode.

    :param regime_parameters: Parameters per regime, ordered as REGIME_ORDER.
    :param transition_matrix: Row-stochastic regime transition matrix.
    :param fill_aggressiveness_coef: Slope of the win logit in normalized
        aggressiveness z (shared across regimes).
    :param size_win_penalty: Win-logit penalty per extra unit of size.
    :param aggressiveness_grid: Candidate normalized aggressiveness values.
    :param rfq_transaction_cost_cents: RFQ transaction cost per unit, cents.
    :param unit_notional_usd: Notional represented by one inventory unit.
    :param initial_regime: Regime at episode start.
    :param initial_cp_plus: CP+ clean price at episode start, in points.
    :param tier_probabilities: Client-tier sampling probabilities.
    """

    regime_parameters: tuple[RegimeParameters, RegimeParameters, RegimeParameters]
    transition_matrix: tuple[tuple[float, ...], ...]
    fill_aggressiveness_coef: float = 1.6
    size_win_penalty: float = 0.15
    aggressiveness_grid: tuple[float, ...] = (
        -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5,
    )
    rfq_transaction_cost_cents: float = 1.25
    unit_notional_usd: float = 100_000.0
    initial_regime: MarketRegime = MarketRegime.NORMAL
    initial_cp_plus: float = 100.0
    tier_probabilities: tuple[float, ...] = (0.35, 0.45, 0.20)

    def __post_init__(self) -> None:
        if len(self.regime_parameters) != len(REGIME_ORDER):
            raise ValueError("regime_parameters must cover every regime")
        if len(self.transition_matrix) != len(REGIME_ORDER):
            raise ValueError("transition_matrix must have one row per regime")
        for row in self.transition_matrix:
            if len(row) != len(REGIME_ORDER):
                raise ValueError("transition_matrix rows must cover every regime")
            if any(entry < 0.0 for entry in row):
                raise ValueError("transition probabilities must be non-negative")
            if abs(sum(row) - 1.0) > 1e-9:
                raise ValueError("transition_matrix rows must sum to one")
        if self.fill_aggressiveness_coef <= 0.0:
            raise ValueError("fill_aggressiveness_coef must be positive")
        if self.rfq_transaction_cost_cents < 0.0:
            raise ValueError("rfq_transaction_cost_cents must be non-negative")
        if self.unit_notional_usd <= 0.0:
            raise ValueError("unit_notional_usd must be positive")
        if len(self.tier_probabilities) != len(CLIENT_TIERS):
            raise ValueError("tier_probabilities must cover every client tier")
        if abs(sum(self.tier_probabilities) - 1.0) > 1e-9:
            raise ValueError("tier_probabilities must sum to one")

    def parameters_for(self, regime: MarketRegime) -> RegimeParameters:
        """Return the parameters of one regime.

        :param regime: Market regime.
        :return: Regime parameters.
        """

        return self.regime_parameters[regime.value]


@dataclass(frozen=True)
class EpisodeConfig:
    """Configuration of one control episode.

    :param name: Stable episode identifier used for bookkeeping.
    :param n_steps: Number of discrete decision steps (the horizon T).
    :param initial_inventory: Inventory at step 0, in units.
    :param target_inventory: Target inventory, in units.
    :param inventory_limit: Hard bound: |inventory| must never exceed it.
    :param running_penalty_cents: phi, cents per unit^2 per step.
    :param terminal_penalty_cents: eta, cents per unit^2 at the horizon.
    :param myopic_inventory_penalty_cents: Static per-trade inventory penalty
        used only by the myopic edge-consistent responder.
    :param active_execution_allowed: Whether active execution is available.
    :param active_action_grid: Discrete active execution amounts, in units.
    """

    name: str
    n_steps: int
    initial_inventory: int
    target_inventory: int
    inventory_limit: int
    running_penalty_cents: float
    terminal_penalty_cents: float
    myopic_inventory_penalty_cents: float
    active_execution_allowed: bool
    active_action_grid: tuple[int, ...] = (-2, -1, 0, 1, 2)

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ValueError("n_steps must be at least one")
        if self.inventory_limit < 1:
            raise ValueError("inventory_limit must be at least one")
        if abs(self.initial_inventory) > self.inventory_limit:
            raise ValueError("initial_inventory must respect the inventory limit")
        if abs(self.target_inventory) > self.inventory_limit:
            raise ValueError("target_inventory must respect the inventory limit")
        if self.running_penalty_cents < 0.0:
            raise ValueError("running_penalty_cents must be non-negative")
        if self.terminal_penalty_cents < 0.0:
            raise ValueError("terminal_penalty_cents must be non-negative")
        if self.myopic_inventory_penalty_cents < 0.0:
            raise ValueError("myopic_inventory_penalty_cents must be non-negative")
        if 0 not in self.active_action_grid:
            raise ValueError("active_action_grid must include zero (wait)")


def default_control_market() -> ControlMarketConfig:
    """Return the calibrated three-regime control market.

    Stressed regimes have fewer but larger RFQs, wider markets, higher
    volatility, costlier active execution, and stronger client information.

    :return: Default market configuration.
    """

    calm = RegimeParameters(
        arrival_probability=0.70,
        size_values=(1, 2),
        size_probabilities=(0.75, 0.25),
        market_width=0.08,
        volatility=0.05,
        liquidity_score=0.85,
        active_half_spread_cents=6.0,
        active_impact_cents=1.5,
        active_fixed_fee_cents=2.0,
        fill_intercept=-0.40,
        information_coefficient=1.10,
        residual_scale=0.10,
        information_strength=0.30,
    )
    normal = RegimeParameters(
        arrival_probability=0.50,
        size_values=(1, 2),
        size_probabilities=(0.60, 0.40),
        market_width=0.12,
        volatility=0.08,
        liquidity_score=0.55,
        active_half_spread_cents=10.0,
        active_impact_cents=3.0,
        active_fixed_fee_cents=2.0,
        fill_intercept=-0.60,
        information_coefficient=1.50,
        residual_scale=0.15,
        information_strength=0.45,
    )
    stressed = RegimeParameters(
        arrival_probability=0.30,
        size_values=(1, 2, 3),
        size_probabilities=(0.40, 0.40, 0.20),
        market_width=0.25,
        volatility=0.18,
        liquidity_score=0.20,
        active_half_spread_cents=25.0,
        active_impact_cents=8.0,
        active_fixed_fee_cents=3.0,
        fill_intercept=-0.90,
        information_coefficient=2.00,
        residual_scale=0.30,
        information_strength=0.65,
    )
    transition = (
        (0.90, 0.09, 0.01),
        (0.05, 0.90, 0.05),
        (0.02, 0.13, 0.85),
    )
    return ControlMarketConfig(
        regime_parameters=(calm, normal, stressed),
        transition_matrix=transition,
    )


def market_making_episode() -> EpisodeConfig:
    """Two-sided market making around zero inventory with no deadline urgency.

    :return: Market-making episode configuration.
    """

    return EpisodeConfig(
        name="market_making",
        n_steps=60,
        initial_inventory=0,
        target_inventory=0,
        inventory_limit=10,
        running_penalty_cents=0.60,
        terminal_penalty_cents=6.0,
        myopic_inventory_penalty_cents=0.60,
        active_execution_allowed=True,
    )


def liquidation_episode() -> EpisodeConfig:
    """Liquidate a long position of +8 units before a finite deadline.

    Client-buy RFQs (dealer sells, side_sign = -1) help liquidation;
    client-sell RFQs (dealer buys, side_sign = +1) enlarge the position.

    :return: Liquidation episode configuration.
    """

    return EpisodeConfig(
        name="liquidation",
        n_steps=40,
        initial_inventory=8,
        target_inventory=0,
        inventory_limit=12,
        running_penalty_cents=0.35,
        terminal_penalty_cents=60.0,
        myopic_inventory_penalty_cents=0.35,
        active_execution_allowed=True,
    )


def acquisition_episode() -> EpisodeConfig:
    """Build a long position of +8 units from flat before a finite deadline.

    Client-sell RFQs (dealer buys, side_sign = +1) help acquisition;
    client-buy RFQs (dealer sells, side_sign = -1) move away from target.

    :return: Acquisition episode configuration.
    """

    return EpisodeConfig(
        name="acquisition",
        n_steps=40,
        initial_inventory=0,
        target_inventory=8,
        inventory_limit=12,
        running_penalty_cents=0.35,
        terminal_penalty_cents=60.0,
        myopic_inventory_penalty_cents=0.35,
        active_execution_allowed=True,
    )


def with_overrides(config: EpisodeConfig, **overrides: object) -> EpisodeConfig:
    """Return a copy of an episode configuration with fields replaced.

    :param config: Base episode configuration.
    :param overrides: Field values to replace.
    :return: New validated configuration.
    """

    return replace(config, **overrides)
