"""Public adverse-selection behavior."""

from rfq_edge import SelectionModelParams, adverse_selection_bps
from rfq_edge.synthetic import RfqRequest, Side


def _request() -> RfqRequest:
    return RfqRequest(
        rfq_id="rfq-test",
        side=Side.BUY,
        quantity=1_000.0,
        mid_price=100.0,
        volatility=0.2,
        inventory=0.0,
        time_to_hedge=0.01,
        competition_count=3,
    )


def test_adverse_selection_falls_as_quoted_edge_rises() -> None:
    params = SelectionModelParams(base_bps=1.5, scale=0.08, decay=0.09)
    request = _request()
    tight = adverse_selection_bps(1.0, request, params)
    wide = adverse_selection_bps(25.0, request, params)
    assert wide < tight
    assert wide >= params.base_bps
