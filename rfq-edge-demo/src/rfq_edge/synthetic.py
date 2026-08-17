"""Synthetic RFQ history for demonstrating the framework before live data."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum


class Side(Enum):
    """Client side on the RFQ ticket."""

    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class IssuerInfo:
    """Issuer metadata attached to a synthetic bond.

    :param issuer_id: Stable issuer identifier.
    :param issuer_name: Human-readable issuer name.
    :param sector: GICS-style sector label used by the demo catalog.
    :param rating: Simple rating bucket for the demo book.
    """

    issuer_id: str
    issuer_name: str
    sector: str
    rating: str


@dataclass(frozen=True)
class BondInfo:
    """Bond metadata for one RFQ line.

    :param bond_id: CUSIP-style identifier.
    :param issuer: Issuer metadata for the name.
    :param coupon_pct: Annual coupon in percent of par.
    :param maturity_year: Calendar maturity year.
    """

    bond_id: str
    issuer: IssuerInfo
    coupon_pct: float
    maturity_year: int


@dataclass(frozen=True)
class SyntheticRfq:
    """One historical RFQ row used to demo calibration and edge analysis.

    :param rfq_id: Stable identifier for the request.
    :param bond: Bond and issuer metadata.
    :param side: Client buy or sell side.
    :param cp_plus_mid: CP+ clean mid at request time.
    :param internal_mid: Dealer internal clean mid at request time.
    :param quote: Dealer clean price submitted on the ticket.
    :param quote_won: Whether the dealer quote won the RFQ.
    :param t5_clean_mark: Realized clean mark five business days later.
    """

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
    """Modeling input derived from a synthetic or live RFQ.

    :param rfq_id: Stable identifier for the request.
    :param side: Client side. BUY means the dealer sells.
    :param quantity: Notional in thousands of par.
    :param mid_price: Internal mid used by the value model.
    :param volatility: Positive volatility feature used by cost and selection.
    :param inventory: Dealer inventory in the name, in thousands of par.
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
    """Sampling ranges for a reproducible synthetic RFQ book.

    :param n_rfqs: Number of requests to draw. Must be at least 1.
    :param seed: RNG seed for reproducible books.
    :param cp_plus_mid_mean: Mean CP+ clean mid around par.
    :param cp_plus_mid_std: CP+ mid standard deviation.
    :param internal_mid_noise_bps: Typical gap between internal and CP+ mids.
    :param quote_edge_min_bps: Inclusive lower dealer edge in basis points.
    :param quote_edge_max_bps: Inclusive upper dealer edge in basis points.
    :param t5_daily_vol_bps: Daily clean-price volatility for the t+5 mark.
    :param win_intercept: Logit intercept for synthetic win probability.
    :param win_edge_coef: Penalty on win odds as quote moves away from CP+.
    :param selection_bps_on_win: Adverse-selection drift applied when quote_won.
    :param log_quantity_mean: Mean of log notional in thousands of par.
    :param log_quantity_std: Std of log notional in thousands of par.
    :param vol_min: Inclusive lower latent volatility bound for modeling.
    :param vol_max: Inclusive upper latent volatility bound for modeling.
    :param inventory_std: Inventory standard deviation around zero.
    :param hedge_horizon_min: Inclusive lower hedge horizon in years.
    :param hedge_horizon_max: Inclusive upper hedge horizon in years.
    :param max_competition: Inclusive maximum competitor count.
    """

    n_rfqs: int
    seed: int
    cp_plus_mid_mean: float
    cp_plus_mid_std: float
    internal_mid_noise_bps: float
    quote_edge_min_bps: float
    quote_edge_max_bps: float
    t5_daily_vol_bps: float
    win_intercept: float
    win_edge_coef: float
    selection_bps_on_win: float
    log_quantity_mean: float
    log_quantity_std: float
    vol_min: float
    vol_max: float
    inventory_std: float
    hedge_horizon_min: float
    hedge_horizon_max: float
    max_competition: int


_BOND_CATALOG: tuple[BondInfo, ...] = (
    BondInfo(
        bond_id="037833AK4",
        issuer=IssuerInfo(
            issuer_id="iss-aapl",
            issuer_name="Apple Inc.",
            sector="Technology",
            rating="AA+",
        ),
        coupon_pct=3.45,
        maturity_year=2029,
    ),
    BondInfo(
        bond_id="88160RAG8",
        issuer=IssuerInfo(
            issuer_id="iss-tsla",
            issuer_name="Tesla Inc.",
            sector="Consumer Discretionary",
            rating="BB+",
        ),
        coupon_pct=5.30,
        maturity_year=2030,
    ),
    BondInfo(
        bond_id="00206RGE7",
        issuer=IssuerInfo(
            issuer_id="iss-atnt",
            issuer_name="AT&T Inc.",
            sector="Communication Services",
            rating="BBB",
        ),
        coupon_pct=4.75,
        maturity_year=2028,
    ),
    BondInfo(
        bond_id="46647PAA1",
        issuer=IssuerInfo(
            issuer_id="iss-jpm",
            issuer_name="JPMorgan Chase & Co.",
            sector="Financials",
            rating="A-",
        ),
        coupon_pct=4.20,
        maturity_year=2031,
    ),
    BondInfo(
        bond_id="58933YBD5",
        issuer=IssuerInfo(
            issuer_id="iss-mrk",
            issuer_name="Merck & Co. Inc.",
            sector="Health Care",
            rating="A+",
        ),
        coupon_pct=3.90,
        maturity_year=2027,
    ),
)


def demo_book_spec() -> SyntheticBookSpec:
    """Return the default synthetic book used by the notebook and tests.

    :return: A fully populated sampling specification.
    """

    return SyntheticBookSpec(
        n_rfqs=40,
        seed=7,
        cp_plus_mid_mean=100.0,
        cp_plus_mid_std=2.5,
        internal_mid_noise_bps=1.5,
        quote_edge_min_bps=1.0,
        quote_edge_max_bps=12.0,
        t5_daily_vol_bps=6.0,
        win_intercept=1.8,
        win_edge_coef=0.35,
        selection_bps_on_win=2.5,
        log_quantity_mean=math.log(2_000.0),
        log_quantity_std=0.6,
        vol_min=0.12,
        vol_max=0.35,
        inventory_std=800.0,
        hedge_horizon_min=1.0 / 252.0,
        hedge_horizon_max=5.0 / 252.0,
        max_competition=8,
    )


def generate_rfq_book(spec: SyntheticBookSpec) -> tuple[SyntheticRfq, ...]:
    """Draw a reproducible book of synthetic RFQ history rows.

    :param spec: Sampling ranges and seed.
    :return: RFQ rows in draw order.
    :raises ValueError: If any sampling bound is invalid.
    """

    _validate_spec(spec)
    rng = random.Random(spec.seed)
    return tuple(_draw_rfq(spec, rng, index) for index in range(spec.n_rfqs))


def to_rfq_request(record: SyntheticRfq, spec: SyntheticBookSpec, index: int) -> RfqRequest:
    """Map a synthetic history row to the modeling input used by the pipeline.

    Latent features are regenerated deterministically from the book seed and row
    index so the demo stays reproducible without storing them on the RFQ row.

    :param record: Synthetic RFQ history row.
    :param spec: Book specification used to draw the row.
    :param index: Zero-based position of the row inside the book.
    :return: Modeling input for value, fill, selection, and cost modules.
    """

    rng = random.Random(spec.seed * 1_000_003 + index)
    quantity = math.exp(rng.gauss(spec.log_quantity_mean, spec.log_quantity_std))
    volatility = rng.uniform(spec.vol_min, spec.vol_max)
    inventory = rng.gauss(0.0, spec.inventory_std)
    time_to_hedge = rng.uniform(spec.hedge_horizon_min, spec.hedge_horizon_max)
    competition_count = rng.randint(1, spec.max_competition)
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


def generate_modeling_book(spec: SyntheticBookSpec) -> tuple[RfqRequest, ...]:
    """Draw synthetic RFQs and convert them to modeling inputs.

    :param spec: Sampling ranges and seed.
    :return: Modeling requests aligned with the synthetic book order.
    """

    records = generate_rfq_book(spec)
    return tuple(
        to_rfq_request(record, spec, index)
        for index, record in enumerate(records)
    )


def _validate_spec(spec: SyntheticBookSpec) -> None:
    if spec.n_rfqs < 1:
        raise ValueError("n_rfqs must be at least 1")
    if spec.cp_plus_mid_mean <= 0.0:
        raise ValueError("cp_plus_mid_mean must be positive")
    if spec.cp_plus_mid_std < 0.0:
        raise ValueError("cp_plus_mid_std must be non-negative")
    if spec.internal_mid_noise_bps < 0.0:
        raise ValueError("internal_mid_noise_bps must be non-negative")
    if spec.quote_edge_min_bps < 0.0:
        raise ValueError("quote_edge_min_bps must be non-negative")
    if spec.quote_edge_max_bps < spec.quote_edge_min_bps:
        raise ValueError("quote_edge_max_bps must be at least quote_edge_min_bps")
    if spec.t5_daily_vol_bps <= 0.0:
        raise ValueError("t5_daily_vol_bps must be positive")
    if spec.win_edge_coef < 0.0:
        raise ValueError("win_edge_coef must be non-negative")
    if spec.selection_bps_on_win < 0.0:
        raise ValueError("selection_bps_on_win must be non-negative")
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
) -> SyntheticRfq:
    bond = _BOND_CATALOG[index % len(_BOND_CATALOG)]
    cp_plus_mid = _draw_positive_mid(
        rng,
        spec.cp_plus_mid_mean,
        spec.cp_plus_mid_std,
    )
    internal_mid = _apply_bps_noise(cp_plus_mid, rng.gauss(0.0, spec.internal_mid_noise_bps))
    side = rng.choice((Side.BUY, Side.SELL))
    edge_bps = rng.uniform(spec.quote_edge_min_bps, spec.quote_edge_max_bps)
    quote = _quote_from_edge(internal_mid, side, edge_bps)
    quote_won = _draw_quote_won(cp_plus_mid, quote, side, spec, rng)
    t5_clean_mark = _draw_t5_mark(
        internal_mid=internal_mid,
        side=side,
        quote_won=quote_won,
        spec=spec,
        rng=rng,
    )
    return SyntheticRfq(
        rfq_id=f"rfq-{index:04d}",
        bond=bond,
        side=side,
        cp_plus_mid=cp_plus_mid,
        internal_mid=internal_mid,
        quote=quote,
        quote_won=quote_won,
        t5_clean_mark=t5_clean_mark,
    )


def _draw_positive_mid(rng: random.Random, mean: float, std: float) -> float:
    for _ in range(32):
        mid_price = rng.gauss(mean, std)
        if mid_price > 0.0:
            return mid_price
    raise ValueError("failed to draw a positive mid price")


def _apply_bps_noise(price: float, noise_bps: float) -> float:
    adjusted = price * (1.0 + noise_bps / 10_000.0)
    if adjusted <= 0.0:
        raise ValueError("mid price must remain positive after noise")
    return adjusted


def _quote_from_edge(internal_mid: float, side: Side, edge_bps: float) -> float:
    edge_dollars = internal_mid * edge_bps / 10_000.0
    if side is Side.BUY:
        return internal_mid + edge_dollars
    if side is Side.SELL:
        return internal_mid - edge_dollars
    raise ValueError(f"unsupported side: {side}")


def _draw_quote_won(
    cp_plus_mid: float,
    quote: float,
    side: Side,
    spec: SyntheticBookSpec,
    rng: random.Random,
) -> bool:
    disadvantage_bps = _quote_disadvantage_bps(cp_plus_mid, quote, side)
    logit = spec.win_intercept - spec.win_edge_coef * disadvantage_bps
    win_probability = _sigmoid(logit)
    return rng.random() < win_probability


def _quote_disadvantage_bps(cp_plus_mid: float, quote: float, side: Side) -> float:
    if side is Side.BUY:
        return max(0.0, (quote - cp_plus_mid) / cp_plus_mid * 10_000.0)
    if side is Side.SELL:
        return max(0.0, (cp_plus_mid - quote) / cp_plus_mid * 10_000.0)
    raise ValueError(f"unsupported side: {side}")


def _draw_t5_mark(
    internal_mid: float,
    side: Side,
    quote_won: bool,
    spec: SyntheticBookSpec,
    rng: random.Random,
) -> float:
    horizon_days = 5.0
    random_move_bps = rng.gauss(0.0, spec.t5_daily_vol_bps * math.sqrt(horizon_days))
    mark = _apply_bps_noise(internal_mid, random_move_bps)
    if not quote_won:
        return mark
    if side is Side.BUY:
        return _apply_bps_noise(mark, spec.selection_bps_on_win)
    if side is Side.SELL:
        return _apply_bps_noise(mark, -spec.selection_bps_on_win)
    raise ValueError(f"unsupported side: {side}")


def _sigmoid(logit: float) -> float:
    if logit >= 0.0:
        exp_neg = math.exp(-logit)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(logit)
    return exp_pos / (1.0 + exp_pos)
