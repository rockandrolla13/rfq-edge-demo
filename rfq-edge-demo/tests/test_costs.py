"""Public trading-cost behavior."""

import pandas as pd
import pytest

from rfq_edge import CostParams, OptimizerConfig, trading_cost_bps
from rfq_edge.costs import rfq_inventory_value_points, rfq_trading_cost_points
from rfq_edge.synthetic import RfqRequest, Side


def _request(side: Side, inventory: float) -> RfqRequest:
    return RfqRequest(
        rfq_id="rfq-test",
        side=side,
        quantity=1_000.0,
        mid_price=100.0,
        volatility=0.2,
        inventory=inventory,
        time_to_hedge=0.01,
        competition_count=3,
    )


def test_trading_cost_is_positive() -> None:
    params = CostParams(
        transaction_bps=0.8,
        inventory_bps_per_unit=0.0015,
        risk_aversion=8.0,
    )
    cost = trading_cost_bps(_request(Side.BUY, 0.0), params)
    assert cost > params.transaction_bps


def test_inventory_reducing_fill_does_not_charge_inventory_penalty() -> None:
    params = CostParams(
        transaction_bps=0.8,
        inventory_bps_per_unit=0.0015,
        risk_aversion=8.0,
    )
    reducing = trading_cost_bps(_request(Side.BUY, 5_000.0), params)
    increasing = trading_cost_bps(_request(Side.BUY, -5_000.0), params)
    assert reducing < increasing


def _cost_row(quote_deadline_ms: float) -> pd.Series:
    return pd.Series(
        {
            "cp_plus": 100.0,
            "market_width": 0.3,
            "volatility": 0.2,
            "size": 1_000.0,
            "liquidity_score": 0.5,
            "quote_deadline_ms": quote_deadline_ms,
        }
    )


def test_shorter_deadlines_cost_more() -> None:
    config = OptimizerConfig()
    rushed = rfq_trading_cost_points(_cost_row(quote_deadline_ms=5_000.0), config)
    relaxed = rfq_trading_cost_points(_cost_row(quote_deadline_ms=120_000.0), config)
    assert rushed > relaxed


def test_trading_cost_requires_contract_columns() -> None:
    row = pd.Series({"cp_plus": 100.0})
    with pytest.raises(ValueError, match="cost contract"):
        rfq_trading_cost_points(row, OptimizerConfig())


def _inventory_row(inventory: float, is_axe: bool) -> pd.Series:
    return pd.Series(
        {
            "side": "dealer_sell",
            "size": 1_000.0,
            "inventory": inventory,
            "is_inventory_axe": is_axe,
        }
    )


def test_axed_inventory_reduction_earns_bonus() -> None:
    config = OptimizerConfig()
    plain = rfq_inventory_value_points(_inventory_row(5_000.0, is_axe=False), config)
    axed = rfq_inventory_value_points(_inventory_row(5_000.0, is_axe=True), config)
    assert plain > 0.0
    assert axed > plain
