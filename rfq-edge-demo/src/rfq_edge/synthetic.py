"""Synthetic RFQ request generation for the responder demo."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum


class Side(Enum):
    """Client-requested side of an RFQ."""

    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class RfqRequest:
    """A single client RFQ the dealer may answer.

    :param rfq_id: Stable identifier for the request.
    :param side: Client side. BUY means the dealer sells.
    :param quantity: Number of units requested.
    :param mid_price: Observed mid at request time.
    :param volatility: Positive volatility feature used by cost and selection.
    :param inventory: Dealer inventory in the name, in units.
    :param time_to_hedge: Expected hedge horizon in years.
    :param competition_count: Number of competing responders.
    """

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
    """Fully specified sampling ranges for a synthetic RFQ book.

    :param n_rfqs: Number of requests to draw. Must be at least 1.
    :param seed: RNG seed for reproducible books.
    :param mid_mean: Mean mid price.
    :param mid_std: Mid price standard deviation.
    :param log_quantity_mean: Mean of log quantity.
    :param log_quantity_std: Std of log quantity.
    :param vol_min: Inclusive lower volatility bound.
    :param vol_max: Inclusive upper volatility bound.
    :param inventory_std: Inventory standard deviation around zero.
    :param hedge_horizon_min: Inclusive lower hedge horizon in years.
    :param hedge_horizon_max: Inclusive upper hedge horizon in years.
    :param max_competition: Inclusive maximum competitor count.
    """

    n_rfqs: int
    seed: int
    mid_mean: float
    mid_std: float
    log_quantity_mean: float
    log_quantity_std: float
    vol_min: float
    vol_max: float
    inventory_std: float
    hedge_horizon_min: float
    hedge_horizon_max: float
    max_competition: int


def demo_book_spec() -> SyntheticBookSpec:
    """Return the default book used by the notebook and tests.

    :return: A fully populated sampling specification.
    """

    return SyntheticBookSpec(
        n_rfqs=40,
        seed=7,
        mid_mean=100.0,
        mid_std=4.0,
        log_quantity_mean=math.log(2_000.0),
        log_quantity_std=0.6,
        vol_min=0.12,
        vol_max=0.35,
        inventory_std=800.0,
        hedge_horizon_min=1.0 / 252.0,
        hedge_horizon_max=5.0 / 252.0,
        max_competition=8,
    )


def generate_rfq_book(spec: SyntheticBookSpec) -> tuple[RfqRequest, ...]:
    """Draw a reproducible book of RFQ requests.

    :param spec: Sampling ranges and seed.
    :return: Requests in draw order.
    :raises ValueError: If any sampling bound is invalid.
    """

    _validate_spec(spec)
    rng = random.Random(spec.seed)
    return tuple(_draw_rfq(spec, rng, index) for index in range(spec.n_rfqs))


def _validate_spec(spec: SyntheticBookSpec) -> None:
    if spec.n_rfqs < 1:
        raise ValueError("n_rfqs must be at least 1")
    if spec.mid_mean <= 0.0:
        raise ValueError("mid_mean must be positive")
    if spec.mid_std < 0.0:
        raise ValueError("mid_std must be non-negative")
    if spec.log_quantity_std < 0.0:
        raise ValueError("log_quantity_std must be non-negative")
    if spec.vol_min <= 0.0 or spec.vol_max < spec.vol_min:
        raise ValueError("volatility bounds must be positive and ordered")
    if spec.inventory_std < 0.0:
        raise ValueError("inventory_std must be non-negative")
    if spec.hedge_horizon_min <= 0.0 or spec.hedge_horizon_max < spec.hedge_horizon_min:
        raise ValueError("hedge horizon bounds must be positive and ordered")
    if spec.max_competition < 1:
        raise ValueError("max_competition must be at least 1")


def _draw_rfq(
    spec: SyntheticBookSpec,
    rng: random.Random,
    index: int,
) -> RfqRequest:
    mid_price = _draw_positive_mid(spec, rng)
    quantity = math.exp(rng.gauss(spec.log_quantity_mean, spec.log_quantity_std))
    volatility = rng.uniform(spec.vol_min, spec.vol_max)
    inventory = rng.gauss(0.0, spec.inventory_std)
    time_to_hedge = rng.uniform(spec.hedge_horizon_min, spec.hedge_horizon_max)
    competition_count = rng.randint(1, spec.max_competition)
    side = rng.choice((Side.BUY, Side.SELL))
    return RfqRequest(
        rfq_id=f"rfq-{index:04d}",
        side=side,
        quantity=quantity,
        mid_price=mid_price,
        volatility=volatility,
        inventory=inventory,
        time_to_hedge=time_to_hedge,
        competition_count=competition_count,
    )


def _draw_positive_mid(spec: SyntheticBookSpec, rng: random.Random) -> float:
    # Reject non-positive mids instead of clipping, so invalid draws stay visible.
    for _ in range(32):
        mid_price = rng.gauss(spec.mid_mean, spec.mid_std)
        if mid_price > 0.0:
            return mid_price
    raise ValueError("failed to draw a positive mid_price")
