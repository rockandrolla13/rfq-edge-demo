"""Reservation-value model for an RFQ responder."""

from __future__ import annotations

from dataclasses import dataclass

from rfq_edge.synthetic import RfqRequest, Side


@dataclass(frozen=True)
class ValueModelParams:
    """Inventory-skew parameters for reservation value.

    :param inventory_skew: Price shift per inventory unit per unit volatility.
    """

    inventory_skew: float


@dataclass(frozen=True)
class FairValue:
    """Dealer value marks used to quote around.

    :param mid: Observed mid copied from the request.
    :param inventory_skew: Signed price skew applied to the mid.
    :param reservation_price: Inventory-adjusted fair price.
    """

    mid: float
    inventory_skew: float
    reservation_price: float


def estimate_fair_value(request: RfqRequest, params: ValueModelParams) -> FairValue:
    """Map an RFQ to an inventory-adjusted reservation price.

    Positive inventory lowers reservation so the dealer is more willing to sell.
    Negative inventory raises reservation so the dealer is more willing to buy.

    :param request: Incoming RFQ.
    :param params: Skew calibration.
    :return: Mid, skew, and reservation price.
    :raises ValueError: If the request is economically invalid.
    """

    _validate_request(request)
    inventory_skew = params.inventory_skew * request.inventory * request.volatility
    reservation_price = request.mid_price - inventory_skew
    if reservation_price <= 0.0:
        raise ValueError("reservation_price must remain positive")
    return FairValue(
        mid=request.mid_price,
        inventory_skew=inventory_skew,
        reservation_price=reservation_price,
    )


def quoted_price(request: RfqRequest, fair_value: FairValue, edge_bps: float) -> float:
    """Convert a dealer edge into the price shown to the client.

    :param request: Incoming RFQ.
    :param fair_value: Reservation mark.
    :param edge_bps: Dealer markup in basis points of mid.
    :return: Bid if the client sells, offer if the client buys.
    :raises ValueError: If edge_bps is negative or the side is unknown.
    """

    if edge_bps < 0.0:
        raise ValueError("edge_bps must be non-negative")
    edge_dollars = fair_value.mid * edge_bps / 10_000.0
    if request.side is Side.BUY:
        return fair_value.reservation_price + edge_dollars
    if request.side is Side.SELL:
        return fair_value.reservation_price - edge_dollars
    raise ValueError(f"unsupported side: {request.side}")


def _validate_request(request: RfqRequest) -> None:
    if request.quantity <= 0.0:
        raise ValueError("quantity must be positive")
    if request.mid_price <= 0.0:
        raise ValueError("mid_price must be positive")
    if request.volatility <= 0.0:
        raise ValueError("volatility must be positive")
    if request.time_to_hedge <= 0.0:
        raise ValueError("time_to_hedge must be positive")
    if request.competition_count < 1:
        raise ValueError("competition_count must be at least 1")
