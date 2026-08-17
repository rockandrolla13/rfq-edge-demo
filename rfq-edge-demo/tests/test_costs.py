"""Public trading-cost behavior."""

from rfq_edge import CostParams, trading_cost_bps
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
