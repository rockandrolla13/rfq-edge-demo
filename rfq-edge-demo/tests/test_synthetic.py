"""Public synthetic-book behavior."""

from rfq_edge import demo_book_spec, generate_rfq_book, to_rfq_request
from rfq_edge.synthetic import Side, SyntheticBookSpec


def test_generate_rfq_book_is_reproducible() -> None:
    spec = demo_book_spec()
    first = generate_rfq_book(spec)
    second = generate_rfq_book(spec)
    assert first == second
    assert len(first) == spec.n_rfqs


def test_generate_rfq_book_contains_required_fields() -> None:
    book = generate_rfq_book(demo_book_spec())
    for record in book:
        assert record.rfq_id
        assert record.bond.bond_id
        assert record.bond.issuer.issuer_name
        assert record.side in (Side.BUY, Side.SELL)
        assert record.cp_plus_mid > 0.0
        assert record.internal_mid > 0.0
        assert record.quote > 0.0
        assert isinstance(record.quote_won, bool)
        assert record.t5_clean_mark > 0.0


def test_to_rfq_request_is_reproducible() -> None:
    spec = demo_book_spec()
    record = generate_rfq_book(spec)[0]
    first = to_rfq_request(record, spec, 0)
    second = to_rfq_request(record, spec, 0)
    assert first == second
    assert first.mid_price == record.internal_mid
    assert first.side == record.side


def test_generate_rfq_book_rejects_empty_books() -> None:
    spec = SyntheticBookSpec(
        n_rfqs=0,
        seed=1,
        cp_plus_mid_mean=100.0,
        cp_plus_mid_std=1.0,
        internal_mid_noise_bps=1.0,
        quote_edge_min_bps=1.0,
        quote_edge_max_bps=5.0,
        t5_daily_vol_bps=6.0,
        win_intercept=1.0,
        win_edge_coef=0.2,
        selection_bps_on_win=2.0,
        log_quantity_mean=7.0,
        log_quantity_std=0.2,
        vol_min=0.1,
        vol_max=0.2,
        inventory_std=1.0,
        hedge_horizon_min=0.01,
        hedge_horizon_max=0.02,
        max_competition=2,
    )
    try:
        generate_rfq_book(spec)
    except ValueError as exc:
        assert "n_rfqs" in str(exc)
        return
    raise AssertionError("expected ValueError for empty books")
