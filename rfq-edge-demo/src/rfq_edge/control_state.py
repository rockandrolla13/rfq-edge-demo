"""State, event, and action types for the event-driven control layer.

The controller never receives a manually assigned market-making/execution
mode as an input. The economic mode is derived *after* an action is selected
(see :func:`classify_economic_mode`) purely as an explanatory label.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rfq_edge.control_config import MarketRegime


class ActionType(Enum):
    """Kinds of actions available to a control policy."""

    WAIT = "wait"
    DECLINE_RFQ = "decline_rfq"
    QUOTE_RFQ = "quote_rfq"
    ACTIVE_EXECUTION = "active_execution"


class EconomicMode(Enum):
    """Descriptive labels derived from the selected action and state."""

    MARKET_MAKING = "market_making"
    PASSIVE_EXECUTION = "passive_execution"
    DEFENSIVE_MARKET_MAKING = "defensive_market_making"
    ACTIVE_EXECUTION = "active_execution"
    WAIT = "wait"
    DECLINE = "decline"


@dataclass(frozen=True)
class RFQEvent:
    """One incoming RFQ.

    ``hidden_client_signal`` and ``hidden_future_residual`` exist only for
    oracle simulation and realized-outcome accounting. Fitted policies must
    never read them; the audit tests verify this by perturbing them.

    :param event_id: Stable identifier within the episode.
    :param time_index: Step at which the RFQ arrives.
    :param side: "dealer_buy" or "dealer_sell" from the dealer's perspective.
    :param side_sign: +1 when the dealer buys, -1 when the dealer sells.
    :param size: RFQ size in inventory units.
    :param client_tier: Observable client tier.
    :param liquidity_score: Observable liquidity score in [0, 1].
    :param market_width: Quoted market width in points.
    :param regime: Observable market regime at arrival.
    :param cp_plus: CP+ clean price at arrival, in points.
    :param hidden_client_signal: Latent client signal (oracle only).
    :param hidden_future_residual: Latent future residual in points (oracle only).
    """

    event_id: int
    time_index: int
    side: str
    side_sign: int
    size: int
    client_tier: str
    liquidity_score: float
    market_width: float
    regime: MarketRegime
    cp_plus: float
    hidden_client_signal: float
    hidden_future_residual: float

    def __post_init__(self) -> None:
        if self.side_sign not in (-1, 1):
            raise ValueError("side_sign must be +1 or -1")
        expected_side = "dealer_buy" if self.side_sign == 1 else "dealer_sell"
        if self.side != expected_side:
            raise ValueError(f"side {self.side} inconsistent with side_sign {self.side_sign}")
        if self.size < 1:
            raise ValueError("size must be at least one unit")
        if self.market_width <= 0.0:
            raise ValueError("market_width must be positive")


@dataclass(frozen=True)
class ControlState:
    """Observable control state at one decision step.

    :param time_index: Current discrete step k.
    :param time_remaining: Steps remaining, T - k.
    :param market_regime: Observable regime.
    :param inventory: Current inventory in units.
    :param target_inventory: Target inventory in units.
    :param initial_inventory: Inventory at episode start.
    :param inventory_limit: Hard inventory bound.
    :param current_cp_plus: Current CP+ clean price in points.
    :param volatility: Current per-step CP+ volatility in points.
    :param liquidity_score: Current liquidity score in [0, 1].
    :param market_width: Current market width in points.
    :param active_execution_available: Whether active execution is allowed.
    :param current_rfq: RFQ arriving this step, if any.
    """

    time_index: int
    time_remaining: int
    market_regime: MarketRegime
    inventory: int
    target_inventory: int
    initial_inventory: int
    inventory_limit: int
    current_cp_plus: float
    volatility: float
    liquidity_score: float
    market_width: float
    active_execution_available: bool
    current_rfq: RFQEvent | None = None

    def __post_init__(self) -> None:
        if abs(self.inventory) > self.inventory_limit:
            raise ValueError("inventory exceeds the inventory limit")
        if self.time_remaining < 0:
            raise ValueError("time_remaining must be non-negative")


@dataclass(frozen=True)
class ControlAction:
    """One selected action.

    :param action_type: Kind of action.
    :param quote: Clean quote in points, when quoting an RFQ.
    :param normalized_aggressiveness: z of the quote, when quoting.
    :param active_execution_amount: Signed active execution in units
        (u > 0 buys / increases inventory, u < 0 sells / decreases it).
    :param respond_or_decline: True when an RFQ is quoted, False otherwise.
    :param policy_name: Name of the policy that chose the action.
    :param predicted_p_win: Policy's own win-probability estimate at the quote.
    :param predicted_selection: Policy's adverse-selection estimate in points.
    :param predicted_post_win_value: Policy's post-win value in points.
    """

    action_type: ActionType
    quote: float | None
    normalized_aggressiveness: float | None
    active_execution_amount: int
    respond_or_decline: bool
    policy_name: str
    predicted_p_win: float = float("nan")
    predicted_selection: float = float("nan")
    predicted_post_win_value: float = float("nan")

    def __post_init__(self) -> None:
        if self.action_type is ActionType.QUOTE_RFQ:
            if self.quote is None or self.normalized_aggressiveness is None:
                raise ValueError("QUOTE_RFQ requires quote and aggressiveness")
            if not self.respond_or_decline:
                raise ValueError("QUOTE_RFQ implies respond_or_decline is True")
        if self.action_type is ActionType.DECLINE_RFQ and self.respond_or_decline:
            raise ValueError("DECLINE_RFQ implies respond_or_decline is False")
        if self.action_type is ActionType.ACTIVE_EXECUTION and self.active_execution_amount == 0:
            raise ValueError("ACTIVE_EXECUTION requires a nonzero amount")
        if self.action_type in (ActionType.WAIT, ActionType.DECLINE_RFQ):
            if self.active_execution_amount != 0:
                raise ValueError(f"{self.action_type} must not carry active execution")


def apply_rfq_fill(inventory: int, side_sign: int, size: int) -> int:
    """Apply a filled RFQ to inventory.

    :param inventory: Inventory before the fill, in units.
    :param side_sign: +1 dealer buy, -1 dealer sell.
    :param size: RFQ size in units.
    :return: Inventory after the fill: I + side_sign * size.
    """

    if side_sign not in (-1, 1):
        raise ValueError("side_sign must be +1 or -1")
    if size < 1:
        raise ValueError("size must be at least one unit")
    return inventory + side_sign * size


def target_shortfall(inventory: int, target_inventory: int) -> int:
    """Absolute distance between inventory and target, in units.

    :param inventory: Current inventory.
    :param target_inventory: Target inventory.
    :return: |inventory - target_inventory|.
    """

    return abs(inventory - target_inventory)


DEFENSIVE_AGGRESSIVENESS_THRESHOLD = -0.75


def classify_economic_mode(
    inventory_before: int,
    target_inventory: int,
    rfq_action: ControlAction | None,
    rfq_side_sign: int | None,
    rfq_size: int | None,
    active_amount: int,
) -> EconomicMode:
    """Derive the descriptive economic mode of a completed step.

    The label explains the selected action; it never determines it.

    Priority: a quoted RFQ is classified first (passive execution if it moves
    inventory toward target, defensive if it moves away and is quoted
    passively, market making otherwise); active execution is labelled next;
    declines and waits close the list.

    :param inventory_before: Inventory before the RFQ stage.
    :param target_inventory: Target inventory.
    :param rfq_action: Action taken on the RFQ, or None if no RFQ arrived.
    :param rfq_side_sign: Side sign of the RFQ, if one arrived.
    :param rfq_size: Size of the RFQ, if one arrived.
    :param active_amount: Signed active execution amount this step.
    :return: Economic mode label.
    """

    if rfq_action is not None and rfq_action.action_type is ActionType.QUOTE_RFQ:
        if rfq_side_sign is None or rfq_size is None:
            raise ValueError("quoted RFQ requires side_sign and size for labelling")
        shortfall_before = target_shortfall(inventory_before, target_inventory)
        inventory_after_fill = apply_rfq_fill(inventory_before, rfq_side_sign, rfq_size)
        shortfall_after = target_shortfall(inventory_after_fill, target_inventory)
        if shortfall_after < shortfall_before:
            return EconomicMode.PASSIVE_EXECUTION
        if shortfall_after > shortfall_before:
            aggressiveness = rfq_action.normalized_aggressiveness
            if aggressiveness is not None and aggressiveness <= DEFENSIVE_AGGRESSIVENESS_THRESHOLD:
                return EconomicMode.DEFENSIVE_MARKET_MAKING
        return EconomicMode.MARKET_MAKING
    if active_amount != 0:
        return EconomicMode.ACTIVE_EXECUTION
    if rfq_action is not None and rfq_action.action_type is ActionType.DECLINE_RFQ:
        return EconomicMode.DECLINE
    return EconomicMode.WAIT
