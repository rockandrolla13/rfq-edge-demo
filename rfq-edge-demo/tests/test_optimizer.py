"""Public optimizer behavior for consistent and expected-PnL quotes."""

from rfq_edge import (
    default_config,
    maximize_expected_pnl,
    solve_consistent_edge,
)
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


def test_consistent_quote_matches_required_edge() -> None:
    config = default_config()
    solution = solve_consistent_edge(_request(), config.models, config.search)
    gap = (
        solution.components.quoted_edge_bps - solution.components.required_edge_bps
    )
    assert abs(gap) < 1e-6
    assert solution.rule == "edge_consistent"
    net_versus_target = (
        solution.components.net_edge_bps - config.models.target_edge_bps
    )
    assert abs(net_versus_target) < 1e-6


def test_expected_pnl_quote_stays_inside_search_bounds() -> None:
    config = default_config()
    solution = maximize_expected_pnl(_request(), config.models, config.search)
    edge = solution.components.quoted_edge_bps
    assert config.search.min_edge_bps <= edge <= config.search.max_edge_bps
    assert solution.rule == "expected_pnl"
