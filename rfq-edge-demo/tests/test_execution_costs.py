"""Public behavior of active execution costs and inventory penalties."""

from __future__ import annotations

import pytest

from rfq_edge.control_config import (
    MarketRegime,
    default_control_market,
    liquidation_episode,
    market_making_episode,
    with_overrides,
)
from rfq_edge.execution_costs import (
    active_execution_cost_cents,
    running_inventory_penalty_cents,
    terminal_penalty_cents,
)

MARKET = default_control_market()


def test_active_cost_is_zero_when_waiting() -> None:
    params = MARKET.parameters_for(MarketRegime.NORMAL)
    assert active_execution_cost_cents(0, params) == 0.0


@pytest.mark.parametrize("amount", [-2, -1, 1, 2])
def test_active_cost_is_positive_for_any_trade(amount: int) -> None:
    params = MARKET.parameters_for(MarketRegime.NORMAL)
    assert active_execution_cost_cents(amount, params) > 0.0


def test_active_cost_is_symmetric_in_direction() -> None:
    params = MARKET.parameters_for(MarketRegime.NORMAL)
    assert active_execution_cost_cents(2, params) == active_execution_cost_cents(-2, params)


def test_active_cost_is_convex_in_size() -> None:
    params = MARKET.parameters_for(MarketRegime.NORMAL)
    one = active_execution_cost_cents(1, params)
    two = active_execution_cost_cents(2, params)
    assert two > 2.0 * one - params.active_fixed_fee_cents


def test_stressed_regime_costs_more_than_calm() -> None:
    calm = MARKET.parameters_for(MarketRegime.CALM_LIQUID)
    stressed = MARKET.parameters_for(MarketRegime.STRESSED_ILLIQUID)
    assert active_execution_cost_cents(2, stressed) > active_execution_cost_cents(2, calm)


def test_terminal_penalty_is_zero_at_target() -> None:
    config = liquidation_episode()
    assert terminal_penalty_cents(config.target_inventory, config) == 0.0


def test_terminal_penalty_grows_quadratically_with_shortfall() -> None:
    config = liquidation_episode()
    one = terminal_penalty_cents(config.target_inventory + 1, config)
    two = terminal_penalty_cents(config.target_inventory + 2, config)
    three = terminal_penalty_cents(config.target_inventory - 3, config)
    assert one == pytest.approx(config.terminal_penalty_cents)
    assert two == pytest.approx(4.0 * one)
    assert three == pytest.approx(9.0 * one)


def test_running_penalty_is_quadratic_around_target() -> None:
    config = with_overrides(market_making_episode(), running_penalty_cents=0.5)
    assert running_inventory_penalty_cents(0, config) == 0.0
    assert running_inventory_penalty_cents(2, config) == pytest.approx(2.0)
    assert running_inventory_penalty_cents(-2, config) == pytest.approx(2.0)


def test_all_penalties_are_non_negative_over_grid() -> None:
    config = liquidation_episode()
    params = MARKET.parameters_for(MarketRegime.STRESSED_ILLIQUID)
    for inventory in range(-config.inventory_limit, config.inventory_limit + 1):
        assert running_inventory_penalty_cents(inventory, config) >= 0.0
        assert terminal_penalty_cents(inventory, config) >= 0.0
    for amount in (-2, -1, 0, 1, 2):
        assert active_execution_cost_cents(amount, params) >= 0.0
