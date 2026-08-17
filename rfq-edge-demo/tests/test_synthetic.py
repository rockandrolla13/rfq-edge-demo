"""Public synthetic-book behavior."""

from rfq_edge import demo_book_spec, generate_rfq_book
from rfq_edge.synthetic import SyntheticBookSpec


def test_generate_rfq_book_is_reproducible() -> None:
    spec = demo_book_spec()
    first = generate_rfq_book(spec)
    second = generate_rfq_book(spec)
    assert first == second
    assert len(first) == spec.n_rfqs


def test_generate_rfq_book_draws_positive_economics() -> None:
    book = generate_rfq_book(demo_book_spec())
    for request in book:
        assert request.quantity > 0.0
        assert request.mid_price > 0.0
        assert request.volatility > 0.0
        assert request.time_to_hedge > 0.0
        assert request.competition_count >= 1


def test_generate_rfq_book_rejects_empty_books() -> None:
    spec = SyntheticBookSpec(
        n_rfqs=0,
        seed=1,
        mid_mean=100.0,
        mid_std=1.0,
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
