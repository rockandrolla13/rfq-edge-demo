"""Win-probability model p(win | q, X) using quote aggressiveness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from rfq_edge.config import FillModelConfig
from rfq_edge.features import (
    FILL_CATEGORICAL_FEATURES,
    FILL_NUMERIC_FEATURES,
    build_feature_preprocessor,
    make_fill_features,
    quote_aggressiveness,
)
from rfq_edge.splits import ChronologicalSplit, chronological_train_test_split


@dataclass(frozen=True)
class FittedFillModel:
    """Chronologically fitted logistic win-probability model.

    :param pipeline: Preprocessing and logistic-regression pipeline.
    :param config: Configuration used during fitting.
    :param selected_c: Inverse regularization chosen by chronological CV.
    """

    pipeline: Pipeline
    config: FillModelConfig
    selected_c: float


def fit_fill_model(
    train_df: pd.DataFrame,
    config: FillModelConfig | None = None,
) -> FittedFillModel:
    """Fit p(win | q, X) on all RFQs, wins and losses.

    :param train_df: Chronological training dataframe with observed outcomes.
    :param config: Fill-model configuration.
    :return: Fitted preprocessing and logistic pipeline.
    """

    model_config = config or FillModelConfig()
    ordered = train_df.sort_values(["timestamp", "rfq_id"])
    features = make_fill_features(ordered)
    target = ordered["won"].astype(int)
    pipeline = _build_logistic_pipeline(model_config)
    pipeline.fit(features, target)
    selected_c = float(pipeline.named_steps["model"].C_[0])
    return FittedFillModel(
        pipeline=pipeline,
        config=model_config,
        selected_c=selected_c,
    )


def predict_win_probability(
    model: FittedFillModel,
    df: pd.DataFrame,
    quote: pd.Series | float | None = None,
) -> pd.Series:
    """Predict calibrated win probability for a candidate quote.

    :param model: Fitted fill model.
    :param df: RFQ dataframe.
    :param quote: Optional counterfactual quote override.
    :return: Win probabilities aligned to ``df.index``.
    """

    features = make_fill_features(df, quote=quote)
    probabilities = model.pipeline.predict_proba(features)[:, 1]
    return pd.Series(probabilities, index=df.index, name="p_win")


def evaluate_fill_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: FillModelConfig | None = None,
) -> dict[str, Any]:
    """Evaluate chronological win-probability performance and calibration.

    :param train_df: Chronological training dataframe.
    :param test_df: Chronological test dataframe.
    :param config: Fill-model configuration.
    :return: Metrics including calibration and monotonicity checks.
    """

    model_config = config or FillModelConfig()
    _assert_chronological_split(train_df, test_df)
    fitted = fit_fill_model(train_df, model_config)
    predicted = predict_win_probability(fitted, test_df)
    actual = test_df["won"].astype(int)
    metrics = _classification_metrics(
        actual=actual,
        predicted=predicted,
        calibration_bins=model_config.calibration_bins,
    )
    metrics["aggressiveness_monotone"] = _aggressiveness_monotonicity(test_df, fitted)
    metrics["buy_side_monotone"] = _side_monotonicity(test_df, fitted, side="dealer_buy")
    metrics["sell_side_monotone"] = _side_monotonicity(test_df, fitted, side="dealer_sell")
    metrics["counterfactual_quote_sensitivity"] = _quote_sensitivity(test_df, fitted)
    metrics["fitted_model"] = fitted
    metrics["predicted_probabilities"] = predicted
    return metrics


def fill_train_test_split(
    df: pd.DataFrame,
    config: FillModelConfig | None = None,
) -> ChronologicalSplit:
    """Split RFQs chronologically for fill-model evaluation.

    :param df: RFQ dataframe.
    :param config: Fill-model configuration.
    :return: Chronological train and test partitions.
    """

    model_config = config or FillModelConfig()
    return chronological_train_test_split(df, model_config.chronological_test_fraction)


def format_fill_metrics(metrics: dict[str, Any]) -> str:
    """Render a compact fill-model metrics table.

    :param metrics: Output from :func:`evaluate_fill_model`.
    :return: Human-readable summary string.
    """

    rows = [
        ("Brier score", f"{metrics['brier_score']:.4f}"),
        ("Log loss", f"{metrics['log_loss']:.4f}"),
        ("Calibration slope", f"{metrics['calibration_slope']:.3f}"),
        ("Calibration intercept", f"{metrics['calibration_intercept']:.3f}"),
        ("Aggressiveness monotone", str(metrics["aggressiveness_monotone"])),
        ("Buy-side monotone", str(metrics["buy_side_monotone"])),
        ("Sell-side monotone", str(metrics["sell_side_monotone"])),
    ]
    name_width = max(len(row[0]) for row in rows)
    return "\n".join(f"{name:<{name_width}}  {value}" for name, value in rows)


def _build_logistic_pipeline(config: FillModelConfig) -> Pipeline:
    preprocessor = build_feature_preprocessor(
        numeric_features=FILL_NUMERIC_FEATURES,
        categorical_features=FILL_CATEGORICAL_FEATURES,
        minimum_category_frequency=config.minimum_category_frequency,
    )
    chronological_cv = TimeSeriesSplit(n_splits=config.number_of_cv_splits)
    model = LogisticRegressionCV(
        Cs=list(config.logistic_c_grid),
        cv=chronological_cv,
        scoring="neg_log_loss",
        max_iter=2_000,
        random_state=config.random_state,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def _classification_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    calibration_bins: int,
) -> dict[str, Any]:
    fraction_positives, mean_predicted = calibration_curve(
        actual,
        predicted,
        n_bins=calibration_bins,
        strategy="quantile",
    )
    slope, intercept = np.polyfit(mean_predicted, fraction_positives, deg=1)
    return {
        "brier_score": float(brier_score_loss(actual, predicted)),
        "log_loss": float(log_loss(actual, predicted, labels=[0, 1])),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "calibration_curve": {
            "mean_predicted": mean_predicted.tolist(),
            "fraction_positives": fraction_positives.tolist(),
        },
    }


def _aggressiveness_monotonicity(test_df: pd.DataFrame, model: FittedFillModel) -> bool:
    aggressiveness = quote_aggressiveness(
        test_df["side_sign"],
        test_df["quote"],
        test_df["cp_plus"],
        test_df["market_width"],
    )
    predicted = predict_win_probability(model, test_df)
    frame = pd.DataFrame({"z": aggressiveness, "p_win": predicted})
    frame["bucket"] = pd.qcut(frame["z"], q=5, duplicates="drop")
    win_by_bucket = frame.groupby("bucket", observed=False)["p_win"].mean()
    return bool(win_by_bucket.is_monotonic_increasing)


def _side_monotonicity(
    test_df: pd.DataFrame,
    model: FittedFillModel,
    side: str,
) -> bool:
    side_frame = test_df.loc[test_df["side"] == side].copy()
    if side_frame.empty:
        return True
    grid = np.linspace(-1.5, 1.5, 7)
    means: list[float] = []
    for aggressiveness in grid:
        quote = side_frame["cp_plus"] + side_frame["side_sign"] * aggressiveness * side_frame["market_width"]
        means.append(float(predict_win_probability(model, side_frame, quote=quote).mean()))
    return bool(pd.Series(means).is_monotonic_increasing)


def _quote_sensitivity(test_df: pd.DataFrame, model: FittedFillModel) -> bool:
    if test_df.empty:
        return True
    row = test_df.iloc[[0]]
    low_quote = row["cp_plus"] + row["side_sign"] * (-1.0) * row["market_width"]
    high_quote = row["cp_plus"] + row["side_sign"] * (1.0) * row["market_width"]
    low_probability = float(predict_win_probability(model, row, quote=low_quote).iloc[0])
    high_probability = float(predict_win_probability(model, row, quote=high_quote).iloc[0])
    return high_probability > low_probability


def _assert_chronological_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_max = pd.to_datetime(train_df["timestamp"]).max()
    test_min = pd.to_datetime(test_df["timestamp"]).min()
    if test_min < train_max:
        raise ValueError("test_df must not begin before the final training timestamp")
