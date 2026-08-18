"""Public behavior of control state, events, actions, and mode labels."""

from __future__ import annotations

import pytest

from rfq_edge.control_config import MarketRegime
from rfq_edge.control_state import (
    ActionType,
    ControlAction,
    ControlState,
    EconomicMode,
    RFQEvent,
    apply_rfq_fill,
    classify_economic_mode,
    target_shortfall,
)


def _event(side_sign: int, size: int = 1) -> RFQEvent:
    return RFQEvent(
        event_id=0,
        time_index=0,
        side="dealer_buy" if side_sign == 1 else "dealer_sell",
        side_sign=side_sign,
        size=size,
        client_tier="professional",
        liquidity_score=0.5,
        market_width=0.12,
        regime=MarketRegime.NORMAL,
        cp_plus=100.0,
        hidden_client_signal=0.0,
        hidden_future_residual=0.0,
    )


def _quote_action(aggressiveness: float) -> ControlAction:
    return ControlAction(
        action_type=ActionType.QUOTE_RFQ,
        quote=100.0,
        normalized_aggressiveness=aggressiveness,
        active_execution_amount=0,
        respond_or_decline=True,
        policy_name="test",
    )


def test_dealer_buy_fill_increases_inventory() -> None:
    assert apply_rfq_fill(inventory=2, side_sign=1, size=3) == 5


def test_dealer_sell_fill_decreases_inventory() -> None:
    assert apply_rfq_fill(inventory=2, side_sign=-1, size=3) == -1


def test_rfq_event_rejects_inconsistent_side() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        RFQEvent(
            event_id=0,
            time_index=0,
            side="dealer_sell",
            side_sign=1,
            size=1,
            client_tier="retail",
            liquidity_score=0.5,
            market_width=0.1,
            regime=MarketRegime.NORMAL,
            cp_plus=100.0,
            hidden_client_signal=0.0,
            hidden_future_residual=0.0,
        )


def test_control_state_rejects_inventory_beyond_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        ControlState(
            time_index=0,
            time_remaining=10,
            market_regime=MarketRegime.NORMAL,
            inventory=11,
            target_inventory=0,
            initial_inventory=0,
            inventory_limit=10,
            current_cp_plus=100.0,
            volatility=0.08,
            liquidity_score=0.5,
            market_width=0.12,
            active_execution_available=True,
        )


def test_quote_action_requires_quote_and_aggressiveness() -> None:
    with pytest.raises(ValueError, match="requires quote"):
        ControlAction(
            action_type=ActionType.QUOTE_RFQ,
            quote=None,
            normalized_aggressiveness=None,
            active_execution_amount=0,
            respond_or_decline=True,
            policy_name="test",
        )


def test_active_execution_action_requires_nonzero_amount() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        ControlAction(
            action_type=ActionType.ACTIVE_EXECUTION,
            quote=None,
            normalized_aggressiveness=None,
            active_execution_amount=0,
            respond_or_decline=False,
            policy_name="test",
        )


def test_helpful_quoted_rfq_is_passive_execution() -> None:
    # Long 8 versus target 0: a dealer sell reduces the shortfall.
    mode = classify_economic_mode(
        inventory_before=8,
        target_inventory=0,
        rfq_action=_quote_action(aggressiveness=0.5),
        rfq_side_sign=-1,
        rfq_size=2,
        active_amount=0,
    )
    assert mode is EconomicMode.PASSIVE_EXECUTION


def test_harmful_rfq_quoted_defensively_is_defensive_market_making() -> None:
    mode = classify_economic_mode(
        inventory_before=8,
        target_inventory=0,
        rfq_action=_quote_action(aggressiveness=-1.25),
        rfq_side_sign=1,
        rfq_size=1,
        active_amount=0,
    )
    assert mode is EconomicMode.DEFENSIVE_MARKET_MAKING


def test_neutral_inventory_quote_is_market_making() -> None:
    mode = classify_economic_mode(
        inventory_before=0,
        target_inventory=0,
        rfq_action=_quote_action(aggressiveness=0.0),
        rfq_side_sign=1,
        rfq_size=1,
        active_amount=0,
    )
    assert mode is EconomicMode.MARKET_MAKING


def test_active_amount_labels_active_execution() -> None:
    mode = classify_economic_mode(
        inventory_before=5,
        target_inventory=0,
        rfq_action=None,
        rfq_side_sign=None,
        rfq_size=None,
        active_amount=-2,
    )
    assert mode is EconomicMode.ACTIVE_EXECUTION


def test_decline_and_wait_labels() -> None:
    decline = ControlAction(
        action_type=ActionType.DECLINE_RFQ,
        quote=None,
        normalized_aggressiveness=None,
        active_execution_amount=0,
        respond_or_decline=False,
        policy_name="test",
    )
    assert (
        classify_economic_mode(0, 0, decline, 1, 1, active_amount=0)
        is EconomicMode.DECLINE
    )
    assert (
        classify_economic_mode(0, 0, None, None, None, active_amount=0)
        is EconomicMode.WAIT
    )


def test_target_shortfall_is_absolute_distance() -> None:
    assert target_shortfall(inventory=3, target_inventory=8) == 5
    assert target_shortfall(inventory=8, target_inventory=8) == 0
