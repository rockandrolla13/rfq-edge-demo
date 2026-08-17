"""Unconditional future-value model V0 and chronological evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rfq_edge.config import ValueModelConfig
from rfq_edge.evaluation import compute_forecast_metrics, format_metrics_table
from rfq_edge.features import (
    VALUE_CATEGORICAL_FEATURES,
    VALUE_NUMERIC_FEATURES,
    make_value_features,
    make_value_target,
)


class ValueForecastKind(str, Enum):
    """Supported unconditional V0 forecast definitions."""

    CP_PLUS = "cp_plus"
    INTERNAL = "internal"
    REGULARIZED = "regularized"


@dataclass(frozen=True)
class FittedValueModel:
    """A chronologically fitted regularized residual model for V0.

    :param pipeline: sklearn preprocessing and Ridge pipeline.
    :param config: Configuration used during fitting.
    :param selected_alpha: Ridge penalty chosen by chronological CV.
    """

    pipeline: Pipeline
    config: ValueModelConfig
    selected_alpha: float


@dataclass(frozen=True)
class ChronologicalSplit:
    """Chronological train and test partitions."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    split_index: int


def chronological_train_test_split(
    df: pd.DataFrame,
    config: ValueModelConfig,
) -> ChronologicalSplit:
    """Split RFQs by timestamp without shuffling.

    :param df: RFQ dataframe containing ``timestamp``.
    :param config: Value-model configuration.
    :return: Chronological train and test partitions.
    """

    ordered = df.sort_values(["timestamp", "rfq_id"]).reset_index(drop=True)
    split_index = int(len(ordered) * (1.0 - config.chronological_test_fraction))
    if split_index <= 0 or split_index >= len(ordered):
        raise ValueError("chronological split produced an empty train or test set")
    return ChronologicalSplit(
        train_df=ordered.iloc[:split_index].copy(),
        test_df=ordered.iloc[split_index:].copy(),
        split_index=split_index,
    )


def fit_value_model(
    train_df: pd.DataFrame,
    config: ValueModelConfig | None = None,
) -> FittedValueModel:
    """Fit the pooled regularized residual model on earlier observations.

    :param train_df: Chronological training dataframe.
    :param config: Value-model configuration.
    :return: Fitted preprocessing and Ridge pipeline.
    """

    model_config = config or ValueModelConfig()
    ordered_train = train_df.sort_values(["timestamp", "rfq_id"])
    features = make_value_features(ordered_train)
    target = make_value_target(ordered_train)
    pipeline = _build_ridge_pipeline(model_config)
    pipeline.fit(features, target)
    selected_alpha = float(pipeline.named_steps["model"].alpha_)
    return FittedValueModel(
        pipeline=pipeline,
        config=model_config,
        selected_alpha=selected_alpha,
    )


def predict_value_residual(
    model: FittedValueModel | ValueForecastKind,
    df: pd.DataFrame,
) -> pd.Series:
    """Predict the future-value residual ``V0 - cp_plus``.

    :param model: Fitted regularized model or a baseline forecast kind.
    :param df: RFQ dataframe.
    :return: Predicted residual aligned to ``df.index``.
    """

    if isinstance(model, ValueForecastKind):
        if model is ValueForecastKind.CP_PLUS:
            return pd.Series(0.0, index=df.index, name="value_residual_prediction")
        if model is ValueForecastKind.INTERNAL:
            return df["internal_mid"] - df["cp_plus"]
        raise ValueError(f"baseline kind {model} does not support direct residual prediction")

    features = make_value_features(df)
    predictions = model.pipeline.predict(features)
    return pd.Series(predictions, index=df.index, name="value_residual_prediction")


def predict_v0(
    model: FittedValueModel | ValueForecastKind,
    df: pd.DataFrame,
) -> pd.Series:
    """Predict unconditional future clean value ``V0``.

    :param model: Fitted regularized model or a baseline forecast kind.
    :param df: RFQ dataframe containing ``cp_plus``.
    :return: Predicted V0 aligned to ``df.index``.
    """

    if model is ValueForecastKind.CP_PLUS:
        return df["cp_plus"].astype(float).rename("v0")
    if model is ValueForecastKind.INTERNAL:
        return df["internal_mid"].astype(float).rename("v0")
    residuals = predict_value_residual(model, df)
    return (df["cp_plus"].astype(float) + residuals).rename("v0")


def evaluate_value_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ValueModelConfig | None = None,
) -> dict[str, Any]:
    """Compare CP+, raw internal, and regularized V0 on the same test rows.

    :param train_df: Chronological training dataframe.
    :param test_df: Chronological test dataframe occurring strictly later.
    :param config: Value-model configuration.
    :return: Metrics for each forecast and the fitted regularized model.
    """

    model_config = config or ValueModelConfig()
    _assert_chronological_split(train_df, test_df)
    fitted = fit_value_model(train_df, model_config)
    group_frame = _evaluation_group_frame(train_df, test_df)
    forecasts = {
        ValueForecastKind.CP_PLUS.value: predict_v0(ValueForecastKind.CP_PLUS, test_df),
        ValueForecastKind.INTERNAL.value: predict_v0(ValueForecastKind.INTERNAL, test_df),
        ValueForecastKind.REGULARIZED.value: predict_v0(fitted, test_df),
    }
    metrics_by_model = {
        name: compute_forecast_metrics(
            actual_prices=test_df["y5"],
            predicted_prices=predictions,
            cp_plus=test_df["cp_plus"],
            sizes=test_df["size"],
            config=model_config,
            group_frame=group_frame,
        )
        for name, predictions in forecasts.items()
    }
    return {
        "metrics_by_model": metrics_by_model,
        "metrics_table": format_metrics_table(metrics_by_model),
        "fitted_model": fitted,
        "forecasts": forecasts,
    }


def make_chronological_oof_v0(
    df: pd.DataFrame,
    config: ValueModelConfig | None = None,
) -> pd.DataFrame:
    """Generate expanding-window out-of-fold V0 predictions.

    The earliest observations remain missing because they only appear in
    training folds and never in a validation fold.

    :param df: Full RFQ dataframe sorted by ``timestamp``.
    :param config: Value-model configuration.
    :return: OOF predictions aligned to ``rfq_id``.
    """

    model_config = config or ValueModelConfig()
    ordered = df.sort_values(["timestamp", "rfq_id"]).reset_index(drop=True)
    oof_predictions = pd.Series(np.nan, index=ordered.index, dtype=float)
    oof_residuals = pd.Series(np.nan, index=ordered.index, dtype=float)
    oof_fold = pd.Series(pd.NA, index=ordered.index, dtype="Int64")

    splitter = TimeSeriesSplit(n_splits=model_config.number_of_oof_splits)
    for fold_number, (train_idx, val_idx) in enumerate(splitter.split(ordered)):
        train_df = ordered.iloc[train_idx]
        val_df = ordered.iloc[val_idx]
        fitted = fit_value_model(train_df, model_config)
        val_v0 = predict_v0(fitted, val_df)
        val_residual = predict_value_residual(fitted, val_df)
        oof_predictions.iloc[val_idx] = val_v0.to_numpy()
        oof_residuals.iloc[val_idx] = val_residual.to_numpy()
        oof_fold.iloc[val_idx] = fold_number

    return pd.DataFrame(
        {
            "rfq_id": ordered["rfq_id"],
            "timestamp": ordered["timestamp"],
            "v0_oof": oof_predictions.to_numpy(),
            "value_residual_prediction_oof": oof_residuals.to_numpy(),
            "oof_fold": oof_fold.to_numpy(),
        }
    )


def _build_ridge_pipeline(config: ValueModelConfig) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=config.minimum_category_frequency,
                    sparse_output=False,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(VALUE_NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(VALUE_CATEGORICAL_FEATURES)),
        ]
    )
    chronological_cv = TimeSeriesSplit(n_splits=config.number_of_oof_splits)
    model = RidgeCV(
        alphas=list(config.ridge_alpha_grid),
        cv=chronological_cv,
        scoring="neg_mean_absolute_error",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def _assert_chronological_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_max = pd.to_datetime(train_df["timestamp"]).max()
    test_min = pd.to_datetime(test_df["timestamp"]).min()
    # Multiple RFQs can share a calendar day; row-order splits may share a boundary date.
    if test_min < train_max:
        raise ValueError("test_df must not begin before the final training timestamp")


def _evaluation_group_frame(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    bond_history = combined.groupby("bond_id", sort=False).cumcount().astype(float) + 1.0
    combined["bond_history_count"] = bond_history.to_numpy()
    test_ids = set(test_df["rfq_id"])
    return combined.loc[combined["rfq_id"].isin(test_ids)].copy()
