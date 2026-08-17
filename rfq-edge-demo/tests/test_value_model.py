"""Public reservation-value and quoted-price behavior."""

from rfq_edge import (
    Side,
    ValueModelParams,
    estimate_fair_value,
    quoted_price,
)
from rfq_edge.synthetic import RfqRequest


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


def test_long_inventory_lowers_reservation_price() -> None:
    params = ValueModelParams(inventory_skew=0.00002)
    long_value = estimate_fair_value(_request(Side.BUY, 2_000.0), params)
    short_value = estimate_fair_value(_request(Side.BUY, -2_000.0), params)
    assert long_value.reservation_price < 100.0
    assert short_value.reservation_price > 100.0


def test_quoted_price_is_offer_when_client_buys() -> None:
    request = _request(Side.BUY, 0.0)
    fair_value = estimate_fair_value(request, ValueModelParams(inventory_skew=0.00002))
    price = quoted_price(request, fair_value, 10.0)
    assert price > fair_value.reservation_price


def test_quoted_price_is_bid_when_client_sells() -> None:
    request = _request(Side.SELL, 0.0)
    fair_value = estimate_fair_value(request, ValueModelParams(inventory_skew=0.00002))
    price = quoted_price(request, fair_value, 10.0)
    assert price < fair_value.reservation_price
