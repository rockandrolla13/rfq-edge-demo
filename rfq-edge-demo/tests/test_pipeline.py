"""Public pipeline behavior for a single RFQ and a book."""

from rfq_edge import (
    default_config,
    demo_book_spec,
    generate_modeling_book,
    generate_rfq_book,
    run_book,
    run_responder,
    to_rfq_request,
)


def test_run_responder_returns_both_quote_rules() -> None:
    config = default_config()
    spec = demo_book_spec()
    record = generate_rfq_book(spec)[0]
    request = to_rfq_request(record, spec, 0)
    decision = run_responder(request, config)
    assert decision.consistent.rule == "edge_consistent"
    assert decision.optimal.rule == "expected_pnl"
    assert decision.consistent_price > 0.0
    assert decision.optimal_price > 0.0


def test_run_book_preserves_request_order() -> None:
    config = default_config()
    requests = generate_modeling_book(demo_book_spec())
    decisions = run_book(requests, config)
    assert len(decisions) == len(requests)
    for request, decision in zip(requests, decisions, strict=True):
        assert decision.request == request
