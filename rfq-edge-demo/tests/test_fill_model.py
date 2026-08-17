"""Public fill-probability behavior."""

from rfq_edge import FillModelParams, fill_probability
from rfq_edge.synthetic import RfqRequest, Side


def _request(competition_count: int) -> RfqRequest:
    return RfqRequest(
        rfq_id="rfq-test",
        side=Side.BUY,
        quantity=1_000.0,
        mid_price=100.0,
        volatility=0.2,
        inventory=0.0,
        time_to_hedge=0.01,
        competition_count=competition_count,
    )


def test_fill_probability_falls_as_quoted_edge_rises() -> None:
    params = FillModelParams(
        intercept=3.0,
        edge_coef=0.12,
        competition_coef=0.18,
        size_coef=0.12,
    )
    request = _request(3)
    tight = fill_probability(2.0, request, params)
    wide = fill_probability(20.0, request, params)
    assert 0.0 < wide < tight < 1.0


def test_fill_probability_falls_as_competition_rises() -> None:
    params = FillModelParams(
        intercept=3.0,
        edge_coef=0.12,
        competition_coef=0.18,
        size_coef=0.12,
    )
    sparse = fill_probability(8.0, _request(1), params)
    crowded = fill_probability(8.0, _request(8), params)
    assert crowded < sparse
