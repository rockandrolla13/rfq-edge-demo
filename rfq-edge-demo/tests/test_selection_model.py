"""Public adverse-selection model behavior."""

import pandas as pd
import pytest

from rfq_edge import (
    SelectionModelConfig,
    ValueModelConfig,
    evaluate_selection_model,
    fit_selection_model,
    make_chronological_oof_v0,
    make_selection_target,
    make_synthetic_rfqs,
    predict_conditional_mark,
    predict_selection,
    selection_train_test_split,
    validate_rfq_schema,
)
from rfq_edge.responder_selection import SelectionModelParams, adverse_selection_bps
from rfq_edge.synthetic import RfqRequest, Side, SyntheticConfig


def _prepared_frame() -> pd.DataFrame:
    frame = make_synthetic_rfqs(
        config=SyntheticConfig(n_rfqs=2_000, n_bonds=80, n_issuers=20),
        random_state=42,
    )
    oof = make_chronological_oof_v0(frame, ValueModelConfig(number_of_oof_splits=3))
    return frame.merge(oof[["rfq_id", "v0_oof"]], on="rfq_id", how="left")


def _fills(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["won"] & frame["v0_oof"].notna()].copy()


def _config() -> SelectionModelConfig:
    return SelectionModelConfig(number_of_cv_splits=3)


def test_selection_target_formula() -> None:
    frame = _prepared_frame()
    target = make_selection_target(frame)
    expected = frame["side_sign"] * (frame["v0_oof"] - frame["y5"])
    pd.testing.assert_series_equal(target, expected, check_names=False)


def test_fit_selection_model_requires_fills_only() -> None:
    frame = _prepared_frame()
    with pytest.raises(ValueError, match="won RFQs only"):
        fit_selection_model(frame, _config())


def test_fit_selection_model_requires_oof_v0() -> None:
    frame = make_synthetic_rfqs(
        config=SyntheticConfig(n_rfqs=200, n_bonds=20, n_issuers=5),
        random_state=42,
    )
    fills = frame.loc[frame["won"]].copy()
    with pytest.raises(ValueError, match="v0_oof"):
        fit_selection_model(fills, _config())


def test_selection_model_uses_only_fills_with_oof_v0() -> None:
    frame = _prepared_frame()
    fills = _fills(frame)
    split = selection_train_test_split(fills, _config())
    fitted = fit_selection_model(split.train_df, _config())
    predicted = predict_selection(fitted, split.test_df)
    assert predicted.notna().all()


def test_conditional_mark_formula() -> None:
    frame = _prepared_frame()
    fills = _fills(frame).iloc[:10]
    split = selection_train_test_split(fills, _config())
    fitted = fit_selection_model(split.train_df, _config())
    selection = predict_selection(fitted, split.test_df)
    conditional = predict_conditional_mark(
        fitted,
        split.test_df,
        v0=split.test_df["v0_oof"],
    )
    expected = split.test_df["v0_oof"] - split.test_df["side_sign"] * selection
    pd.testing.assert_series_equal(conditional, expected, check_names=False)


def test_quote_changes_selection_prediction() -> None:
    frame = _prepared_frame()
    fills = _fills(frame)
    split = selection_train_test_split(fills, _config())
    fitted = fit_selection_model(split.train_df, _config())
    row = split.test_df.iloc[[0]]
    low_quote = row["cp_plus"] + row["side_sign"] * (-1.0) * row["market_width"]
    high_quote = row["cp_plus"] + row["side_sign"] * (1.0) * row["market_width"]
    low_selection = float(predict_selection(fitted, row, quote=low_quote).iloc[0])
    high_selection = float(predict_selection(fitted, row, quote=high_quote).iloc[0])
    assert low_selection != high_selection


def test_unseen_bonds_can_be_predicted() -> None:
    frame = _prepared_frame()
    fills = _fills(frame)
    split = selection_train_test_split(fills, _config())
    known_bonds = set(split.train_df["bond_id"].unique()[:5])
    train = split.train_df.loc[split.train_df["bond_id"].isin(known_bonds)]
    fitted = fit_selection_model(train, _config())
    unseen = split.test_df.loc[~split.test_df["bond_id"].isin(known_bonds)]
    assert not unseen.empty
    assert predict_selection(fitted, unseen).notna().all()


def test_selection_model_chronological_evaluation_passes_gates() -> None:
    frame = _prepared_frame()
    validate_rfq_schema(frame)
    fills = _fills(frame)
    split = selection_train_test_split(fills, _config())
    metrics = evaluate_selection_model(split.train_df, split.test_df, _config())
    assert metrics["quote_sensitivity"]
    assert metrics["mae_selection"] >= 0.0


def test_legacy_adverse_selection_still_works() -> None:
    request = RfqRequest(
        rfq_id="rfq-test",
        side=Side.BUY,
        quantity=1_000.0,
        mid_price=100.0,
        volatility=0.2,
        inventory=0.0,
        time_to_hedge=0.01,
        competition_count=3,
    )
    params = SelectionModelParams(base_bps=1.5, scale=0.08, decay=0.09)
    tight = adverse_selection_bps(1.0, request, params)
    wide = adverse_selection_bps(25.0, request, params)
    assert wide < tight


def test_selection_model_demo_quality_for_default_seed() -> None:
    frame = make_synthetic_rfqs(random_state=42)
    oof = make_chronological_oof_v0(frame, ValueModelConfig())
    prepared = frame.merge(oof[["rfq_id", "v0_oof"]], on="rfq_id", how="left")
    fills = prepared.loc[prepared["won"] & prepared["v0_oof"].notna()]
    split = selection_train_test_split(fills, SelectionModelConfig())
    metrics = evaluate_selection_model(
        split.train_df,
        split.test_df,
        SelectionModelConfig(),
    )
    assert metrics["quote_sensitivity"]
    print("\nselection metrics:\n", __import__("rfq_edge").format_selection_metrics(metrics))
