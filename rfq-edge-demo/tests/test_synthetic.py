"""Public synthetic-book behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rfq_edge import (
    demo_book_spec,
    generate_modeling_book,
    generate_rfq_book,
    make_synthetic_rfqs,
    to_rfq_request,
    validate_synthetic_data,
)
from rfq_edge.synthetic import (
    OBSERVABLE_COLUMNS,
    LATENT_COLUMNS,
    Side,
    SyntheticBookSpec,
    SyntheticConfig,
)


def _small_config() -> SyntheticConfig:
    return SyntheticConfig(n_rfqs=2_000, n_bonds=120, n_issuers=30)


def test_make_synthetic_rfqs_is_reproducible() -> None:
    first = make_synthetic_rfqs(config=_small_config(), random_state=42)
    second = make_synthetic_rfqs(config=_small_config(), random_state=42)
    pd.testing.assert_frame_equal(first, second)


def test_make_synthetic_rfqs_differs_by_seed() -> None:
    first = make_synthetic_rfqs(config=_small_config(), random_state=42)
    second = make_synthetic_rfqs(config=_small_config(), random_state=43)
    assert not first.equals(second)


def test_make_synthetic_rfqs_required_columns_exist() -> None:
    frame = make_synthetic_rfqs(config=_small_config(), random_state=42)
    for column in OBSERVABLE_COLUMNS:
        assert column in frame.columns


def test_make_synthetic_rfqs_has_no_missing_values() -> None:
    frame = make_synthetic_rfqs(config=_small_config(), random_state=42)
    assert not frame.isna().any().any()


def test_side_sign_matches_side() -> None:
    frame = make_synthetic_rfqs(config=_small_config(), random_state=42)
    dealer_buy_sign = frame.loc[frame["side"] == "dealer_buy", "side_sign"]
    dealer_sell_sign = frame.loc[frame["side"] == "dealer_sell", "side_sign"]
    assert (dealer_buy_sign == 1.0).all()
    assert (dealer_sell_sign == -1.0).all()


def test_market_width_and_size_are_positive() -> None:
    frame = make_synthetic_rfqs(config=_small_config(), random_state=42)
    assert (frame["market_width"] > 0.0).all()
    assert (frame["size"] > 0.0).all()


def test_quote_identity_holds() -> None:
    frame = make_synthetic_rfqs(config=_small_config(), random_state=42)
    aggressiveness = frame["side_sign"] * (frame["quote"] - frame["cp_plus"]) / frame["market_width"]
    reconstructed = frame["cp_plus"] + frame["side_sign"] * aggressiveness * frame["market_width"]
    np.testing.assert_allclose(reconstructed, frame["quote"], rtol=1e-10, atol=1e-10)


def test_higher_aggressiveness_is_associated_with_higher_win_rate() -> None:
    frame = make_synthetic_rfqs(config=_small_config(), random_state=42)
    aggressiveness = frame["side_sign"] * (frame["quote"] - frame["cp_plus"]) / frame["market_width"]
    bucket = pd.qcut(aggressiveness, q=5, duplicates="drop")
    win_by_bucket = frame.groupby(bucket, observed=False)["won"].mean()
    assert win_by_bucket.is_monotonic_increasing


def test_overall_win_rate_in_target_band() -> None:
    frame = make_synthetic_rfqs(config=_small_config(), random_state=42)
    win_rate = float(frame["won"].mean())
    assert 0.20 <= win_rate <= 0.50


def test_bond_level_data_are_sparse() -> None:
    frame = make_synthetic_rfqs(config=_small_config(), random_state=42)
    counts = frame.groupby("bond_id").size()
    assert counts.median() >= 5.0
    assert counts.max() >= counts.median() * 2.0


def test_internal_mid_is_informative_but_not_perfect() -> None:
    frame = make_synthetic_rfqs(config=_small_config(), random_state=42)
    cp_plus_mae = float(np.abs(frame["cp_plus"] - frame["y5"]).mean())
    internal_mae = float(np.abs(frame["internal_mid"] - frame["y5"]).mean())
    assert internal_mae < cp_plus_mae
    assert internal_mae > 0.0


def test_wins_show_measurable_adverse_selection() -> None:
    frame = make_synthetic_rfqs(config=_small_config(), random_state=42)
    selection = frame["side_sign"] * (frame["internal_mid"] - frame["y5"])
    assert float(selection.loc[frame["won"]].mean()) > float(selection.mean())


def test_latent_columns_absent_by_default() -> None:
    frame = make_synthetic_rfqs(config=_small_config(), random_state=42)
    for column in LATENT_COLUMNS:
        assert column not in frame.columns


def test_latent_columns_present_when_requested() -> None:
    frame = make_synthetic_rfqs(
        config=_small_config(),
        random_state=42,
        include_latent=True,
    )
    for column in LATENT_COLUMNS:
        assert column in frame.columns


def test_default_scale_matches_requested_counts() -> None:
    frame = make_synthetic_rfqs(random_state=42)
    assert len(frame) == 15_000
    assert frame["bond_id"].nunique() == 300
    assert frame["issuer_id"].nunique() == 60


def test_validate_synthetic_data_reports_expected_metrics() -> None:
    frame = make_synthetic_rfqs(config=_small_config(), random_state=42)
    summary = validate_synthetic_data(frame)
    assert summary["n_rfqs"] == len(frame)
    assert summary["win_rate_in_target_band"]
    assert summary["internal_mid_more_informative_than_cp_plus"]
    assert summary["mean_selection_wins"] > summary["mean_selection_all"]


def test_generate_rfq_book_legacy_api_still_works() -> None:
    spec = demo_book_spec()
    book = generate_rfq_book(spec)
    assert len(book) == spec.n_rfqs
    assert book[0].cp_plus_mid > 0.0


def test_to_rfq_request_legacy_api_still_works() -> None:
    spec = demo_book_spec()
    record = generate_rfq_book(spec)[0]
    request = to_rfq_request(record, spec, 0)
    assert request.mid_price == record.internal_mid
    assert request.side in (Side.BUY, Side.SELL)


def test_generate_modeling_book_legacy_api_still_works() -> None:
    requests = generate_modeling_book(demo_book_spec())
    assert len(requests) == demo_book_spec().n_rfqs


def test_generate_rfq_book_rejects_empty_books() -> None:
    spec = SyntheticBookSpec(n_rfqs=0)
    with pytest.raises(ValueError, match="n_rfqs"):
        generate_rfq_book(spec)


def test_validation_summary_for_default_seed() -> None:
    frame = make_synthetic_rfqs(random_state=42)
    summary = validate_synthetic_data(frame)
    print("\nvalidation summary:", summary)
    assert summary["aggressiveness_win_correlation"] > 0.0
    assert summary["aggressiveness_bucket_win_rates_monotone"]
    assert summary["win_rate_in_target_band"]
    assert summary["internal_mid_more_informative_than_cp_plus"]
    assert summary["mean_selection_wins"] > 0.0
