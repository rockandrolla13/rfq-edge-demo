"""Public pipeline behavior for a single RFQ and a book."""

from rfq_edge import (
    cold_start_comparison,
    default_config,
    demo_book_spec,
    generate_modeling_book,
    generate_rfq_book,
    run_book,
    run_responder,
    score_rfq,
    to_rfq_request,
)


def test_cold_start_comparison_covers_history_regimes(demo_framework) -> None:
    comparison = cold_start_comparison(demo_framework)
    assert list(comparison["case"]) == [
        "active bond",
        "sparse bond",
        "unseen bond, known issuer",
        "unseen issuer",
    ]
    counts = comparison.set_index("case")["train_rfqs_on_bond"]
    assert counts["active bond"] > counts["sparse bond"]
    assert counts["unseen bond, known issuer"] == 0
    assert counts["unseen issuer"] == 0
    assert comparison["predicted_v0"].notna().all()
    assert comparison["v0_minus_cp_plus_cents"].notna().all()


def test_score_rfq_returns_grid_and_comparison(demo_framework) -> None:
    rfq = demo_framework.test_df.loc[demo_framework.test_df["v0_oof"].notna()].iloc[[0]]
    scored = score_rfq(demo_framework, rfq)
    assert set(scored) == {"grid", "comparison"}
    assert len(scored["comparison"]) == 3
    assert not scored["grid"].empty


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
