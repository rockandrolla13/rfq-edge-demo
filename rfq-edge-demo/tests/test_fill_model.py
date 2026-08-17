"""Public fill-model behavior."""

import numpy as np
import pandas as pd
import pytest

from rfq_edge import (
    FillModelConfig,
    evaluate_fill_model,
    fill_train_test_split,
    fit_fill_model,
    make_synthetic_rfqs,
    predict_win_probability,
    quote_aggressiveness,
    validate_rfq_schema,
)
from rfq_edge.features import make_fill_features
from rfq_edge.responder_fill import FillModelParams, fill_probability
from rfq_edge.synthetic import RfqRequest, Side, SyntheticConfig


def _frame() -> pd.DataFrame:
    return make_synthetic_rfqs(
        config=SyntheticConfig(n_rfqs=2_000, n_bonds=80, n_issuers=20),
        random_state=42,
    )


def _config() -> FillModelConfig:
    return FillModelConfig(number_of_cv_splits=3)


def test_quote_aggressiveness_sign_for_buy_and_sell() -> None:
    frame = _frame()
    buy = frame.loc[frame["side"] == "dealer_buy"].iloc[0]
    sell = frame.loc[frame["side"] == "dealer_sell"].iloc[0]
    higher_buy_quote = buy["cp_plus"] + 0.5 * buy["market_width"]
    lower_sell_quote = sell["cp_plus"] - 0.5 * sell["market_width"]
    buy_z = float(
        quote_aggressiveness(
            pd.Series([buy["side_sign"]]),
            pd.Series([higher_buy_quote]),
            pd.Series([buy["cp_plus"]]),
            pd.Series([buy["market_width"]]),
        ).iloc[0]
    )
    sell_z = float(
        quote_aggressiveness(
            pd.Series([sell["side_sign"]]),
            pd.Series([lower_sell_quote]),
            pd.Series([sell["cp_plus"]]),
            pd.Series([sell["market_width"]]),
        ).iloc[0]
    )
    assert buy_z > 0.0
    assert sell_z > 0.0


def test_make_fill_features_excludes_outcome_columns() -> None:
    frame = _frame()
    features = make_fill_features(frame)
    assert "won" not in features.columns
    assert "y5" not in features.columns
    assert "aggressiveness" in features.columns


def test_fill_model_uses_all_rfqs() -> None:
    frame = _frame()
    split = fill_train_test_split(frame, _config())
    assert split.train_df["won"].nunique() == 2
    fitted = fit_fill_model(split.train_df, _config())
    probabilities = predict_win_probability(fitted, split.test_df)
    assert probabilities.between(0.0, 1.0).all()


def test_higher_aggressiveness_raises_win_probability() -> None:
    frame = _frame()
    split = fill_train_test_split(frame, _config())
    fitted = fit_fill_model(split.train_df, _config())
    row = split.test_df.iloc[[0]]
    low_quote = row["cp_plus"] + row["side_sign"] * (-1.0) * row["market_width"]
    high_quote = row["cp_plus"] + row["side_sign"] * (1.0) * row["market_width"]
    low_probability = float(predict_win_probability(fitted, row, quote=low_quote).iloc[0])
    high_probability = float(predict_win_probability(fitted, row, quote=high_quote).iloc[0])
    assert high_probability > low_probability


def test_fill_model_chronological_evaluation_passes_gates() -> None:
    frame = _frame()
    validate_rfq_schema(frame)
    split = fill_train_test_split(frame, _config())
    metrics = evaluate_fill_model(split.train_df, split.test_df, _config())
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["aggressiveness_monotone"]
    assert metrics["buy_side_monotone"]
    assert metrics["sell_side_monotone"]
    assert metrics["counterfactual_quote_sensitivity"]


def test_legacy_fill_probability_still_works() -> None:
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
    params = FillModelParams(
        intercept=3.0,
        edge_coef=0.12,
        competition_coef=0.18,
        size_coef=0.12,
    )
    tight = fill_probability(2.0, request, params)
    wide = fill_probability(20.0, request, params)
    assert 0.0 < wide < tight < 1.0


def test_fill_model_demo_quality_for_default_seed() -> None:
    frame = make_synthetic_rfqs(random_state=42)
    split = fill_train_test_split(frame, FillModelConfig())
    metrics = evaluate_fill_model(split.train_df, split.test_df, FillModelConfig())
    assert metrics["aggressiveness_monotone"]
    print("\nfill metrics:\n", __import__("rfq_edge").format_fill_metrics(metrics))
