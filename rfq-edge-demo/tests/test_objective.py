"""Public quote-objective behavior."""

from rfq_edge import default_config, evaluate_quote
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


def test_evaluate_quote_net_edge_subtracts_selection_and_costs() -> None:
    models = default_config().models
    components = evaluate_quote(_request(), models, 12.0)
    expected_net = (
        components.quoted_edge_bps - components.selection_bps - components.cost_bps
    )
    assert components.net_edge_bps == expected_net
    assert components.required_edge_bps == (
        components.selection_bps + components.cost_bps + models.target_edge_bps
    )


def test_evaluate_quote_expected_pnl_uses_fill_probability() -> None:
    models = default_config().models
    request = _request()
    components = evaluate_quote(request, models, 12.0)
    dollar_net = components.net_edge_bps / 10_000.0 * request.mid_price * request.quantity
    assert abs(components.expected_pnl - components.fill_probability * dollar_net) < 1e-9
