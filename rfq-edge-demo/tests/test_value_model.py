"""Public V0 value-model and legacy responder-value behavior."""

import numpy as np
import pandas as pd
import pytest

from rfq_edge import (
    Side,
    ValueForecastKind,
    ValueModelConfig,
    ValueModelParams,
    chronological_train_test_split,
    estimate_fair_value,
    evaluate_value_models,
    fit_value_model,
    make_chronological_oof_v0,
    make_synthetic_rfqs,
    make_value_features,
    predict_v0,
    predict_value_residual,
    quoted_price,
    validate_rfq_schema,
)
from rfq_edge.responder_value import FairValue
from rfq_edge.synthetic import RfqRequest, SyntheticConfig


def _sample_frame() -> pd.DataFrame:
    return make_synthetic_rfqs(
        config=SyntheticConfig(n_rfqs=2_000, n_bonds=80, n_issuers=20),
        random_state=42,
    )


def _config() -> ValueModelConfig:
    return ValueModelConfig(
        chronological_test_fraction=0.20,
        number_of_oof_splits=3,
        random_state=42,
    )


def test_cp_plus_prediction_equals_cp_plus() -> None:
    frame = _sample_frame()
    predicted = predict_v0(ValueForecastKind.CP_PLUS, frame)
    pd.testing.assert_series_equal(
        predicted,
        frame["cp_plus"].astype(float).rename("v0"),
        check_names=False,
    )


def test_raw_internal_prediction_equals_internal_mid() -> None:
    frame = _sample_frame()
    predicted = predict_v0(ValueForecastKind.INTERNAL, frame)
    pd.testing.assert_series_equal(
        predicted,
        frame["internal_mid"].astype(float).rename("v0"),
        check_names=False,
    )


def test_cp_plus_residual_prediction_is_zero() -> None:
    frame = _sample_frame()
    residuals = predict_value_residual(ValueForecastKind.CP_PLUS, frame)
    assert (residuals == 0.0).all()


def test_chronological_test_set_occurs_after_training_set() -> None:
    frame = _sample_frame()
    split = chronological_train_test_split(frame, _config())
    train_max = pd.to_datetime(split.train_df["timestamp"]).max()
    test_min = pd.to_datetime(split.test_df["timestamp"]).min()
    assert test_min >= train_max
    assert split.split_index == len(split.train_df)


def test_out_of_fold_predictions_use_only_earlier_observations() -> None:
    frame = _sample_frame().sort_values("timestamp").reset_index(drop=True)
    config = _config()
    oof = make_chronological_oof_v0(frame, config)
    merged = frame.merge(oof, on="rfq_id", how="left", suffixes=("", "_oof"))
    predicted_rows = merged.dropna(subset=["v0_oof"])
    for fold in sorted(predicted_rows["oof_fold"].dropna().unique()):
        fold_rows = predicted_rows.loc[predicted_rows["oof_fold"] == fold]
        fold_start = pd.to_datetime(fold_rows["timestamp"].min())
        earlier = frame.loc[pd.to_datetime(frame["timestamp"]) < fold_start]
        assert len(earlier) > 0


def test_original_row_order_recoverable_by_rfq_id() -> None:
    frame = _sample_frame()
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    validate_rfq_schema(shuffled)
    split = chronological_train_test_split(shuffled, _config())
    fitted = fit_value_model(split.train_df, _config())
    predictions = predict_v0(fitted, shuffled)
    recovered = shuffled[["rfq_id"]].copy()
    recovered["v0"] = predictions.to_numpy()
    reordered = recovered.set_index("rfq_id").loc[frame["rfq_id"]]["v0"]
    assert len(reordered) == len(frame)


def test_unseen_bonds_and_issuers_can_be_predicted() -> None:
    frame = _sample_frame()
    split = chronological_train_test_split(frame, _config())
    train_bonds = set(split.train_df["bond_id"].unique()[:10])
    train_df = split.train_df.loc[split.train_df["bond_id"].isin(train_bonds)]
    fitted = fit_value_model(train_df, _config())
    unseen = split.test_df.loc[~split.test_df["bond_id"].isin(train_bonds)]
    assert not unseen.empty
    predictions = predict_v0(fitted, unseen)
    assert predictions.notna().all()


def test_fitted_test_predictions_are_not_nan() -> None:
    frame = _sample_frame()
    split = chronological_train_test_split(frame, _config())
    fitted = fit_value_model(split.train_df, _config())
    predictions = predict_v0(fitted, split.test_df)
    assert predictions.notna().all()


def test_output_prices_and_residuals_have_consistent_units() -> None:
    frame = _sample_frame()
    split = chronological_train_test_split(frame, _config())
    fitted = fit_value_model(split.train_df, _config())
    residuals = predict_value_residual(fitted, split.test_df)
    prices = predict_v0(fitted, split.test_df)
    expected_prices = split.test_df["cp_plus"] + residuals
    np.testing.assert_allclose(prices, expected_prices, rtol=1e-10, atol=1e-10)


def test_evaluate_value_models_returns_three_forecasts() -> None:
    frame = _sample_frame()
    split = chronological_train_test_split(frame, _config())
    results = evaluate_value_models(split.train_df, split.test_df, _config())
    assert set(results["metrics_by_model"]) == {
        "cp_plus",
        "internal",
        "regularized",
    }


def test_oof_coverage_leaves_initial_window_missing() -> None:
    frame = _sample_frame()
    oof = make_chronological_oof_v0(frame, _config())
    assert oof["v0_oof"].isna().any()
    assert oof["v0_oof"].notna().any()


def test_value_model_demo_quality_for_default_seed() -> None:
    frame = make_synthetic_rfqs(random_state=42)
    validate_rfq_schema(frame)
    config = ValueModelConfig()
    split = chronological_train_test_split(frame, config)
    results = evaluate_value_models(split.train_df, split.test_df, config)
    metrics = results["metrics_by_model"]
    assert metrics["internal"]["mae_price_points"] < metrics["cp_plus"]["mae_price_points"]
    regularized_mae = metrics["regularized"]["mae_price_points"]
    internal_mae = metrics["internal"]["mae_price_points"]
    assert regularized_mae <= internal_mae * 1.05


def test_long_inventory_lowers_reservation_price() -> None:
    params = ValueModelParams(inventory_skew=0.00002)
    long_value = estimate_fair_value(_legacy_request(Side.BUY, 2_000.0), params)
    short_value = estimate_fair_value(_legacy_request(Side.BUY, -2_000.0), params)
    assert long_value.reservation_price < 100.0
    assert short_value.reservation_price > 100.0


def test_quoted_price_is_offer_when_client_buys() -> None:
    request = _legacy_request(Side.BUY, 0.0)
    fair_value = estimate_fair_value(request, ValueModelParams(inventory_skew=0.00002))
    price = quoted_price(request, fair_value, 10.0)
    assert price > fair_value.reservation_price


def test_quoted_price_is_bid_when_client_sells() -> None:
    request = _legacy_request(Side.SELL, 0.0)
    fair_value = estimate_fair_value(request, ValueModelParams(inventory_skew=0.00002))
    price = quoted_price(request, fair_value, 10.0)
    assert price < fair_value.reservation_price


def _legacy_request(side: Side, inventory: float) -> RfqRequest:
    return RfqRequest(
        rfq_id="rfq-test",
        side=side,
        quantity=1_000.0,
        mid_price=100.0,
        volatility=0.2,
        inventory=inventory,
        time_to_hedge=0.01,
        competition_count=3,
    )
