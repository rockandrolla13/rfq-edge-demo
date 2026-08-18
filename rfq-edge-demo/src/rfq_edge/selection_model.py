"""Adverse-selection model A(q, X) conditional on winning the RFQ."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from rfq_edge.config import SelectionModelConfig
from rfq_edge.features import (
    SELECTION_CATEGORICAL_FEATURES,
    SELECTION_NUMERIC_FEATURES,
    build_feature_preprocessor,
    make_selection_features,
)
from rfq_edge.splits import ChronologicalSplit, chronological_train_test_split

V0_OOF_COLUMN = "v0_oof"


@dataclass(frozen=True)
class FittedSelectionModel:
    """Chronologically fitted adverse-selection model on fills.

    :param pipeline: Preprocessing and Ridge pipeline for realized selection.
    :param config: Configuration used during fitting.
    :param selected_alpha: Ridge penalty chosen by chronological CV.
    """

    pipeline: Pipeline
    config: SelectionModelConfig
    selected_alpha: float


def make_selection_target(
    df: pd.DataFrame,
    v0_column: str = V0_OOF_COLUMN,
) -> pd.Series:
    """Build realized selection D = side_sign * (V0^OOF - y5).

    :param df: RFQ dataframe with out-of-fold V0 and realized y5.
    :param v0_column: Column containing out-of-fold V0 predictions.
    :return: Realized selection aligned to ``df.index``.
    :raises ValueError: If required columns are missing.
    """

    required = {v0_column, "y5", "side_sign"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing columns for selection target: {sorted(missing)}")
    return df["side_sign"].astype(float) * (df[v0_column].astype(float) - df["y5"].astype(float))


def fit_selection_model(
    train_fills_df: pd.DataFrame,
    config: SelectionModelConfig | None = None,
    v0_column: str = V0_OOF_COLUMN,
) -> FittedSelectionModel:
    """Fit A(q, X) = E[D | W=1, q, X] using won RFQs only.

    :param train_fills_df: Chronological training fills with out-of-fold V0.
    :param config: Selection-model configuration.
    :param v0_column: Column containing out-of-fold V0 predictions.
    :return: Fitted preprocessing and Ridge pipeline.
    :raises ValueError: If non-fill rows or missing V0 are supplied.
    """

    model_config = config or SelectionModelConfig()
    fills = _require_fills_with_v0(train_fills_df, v0_column)
    ordered = fills.sort_values(["timestamp", "rfq_id"])
    features = make_selection_features(ordered)
    target = make_selection_target(ordered, v0_column=v0_column)
    pipeline = _build_ridge_pipeline(model_config)
    pipeline.fit(features, target)
    selected_alpha = float(pipeline.named_steps["model"].alpha_)
    return FittedSelectionModel(
        pipeline=pipeline,
        config=model_config,
        selected_alpha=selected_alpha,
    )


def predict_selection(
    model: FittedSelectionModel,
    df: pd.DataFrame,
    quote: pd.Series | float | None = None,
) -> pd.Series:
    """Predict expected realized selection A(q, X).

    :param model: Fitted selection model.
    :param df: RFQ dataframe, typically fills.
    :param quote: Optional counterfactual quote override.
    :return: Expected selection aligned to ``df.index``.
    """

    features = make_selection_features(df, quote=quote)
    predictions = model.pipeline.predict(features)
    return pd.Series(predictions, index=df.index, name="selection")


def predict_conditional_mark(
    model: FittedSelectionModel,
    df: pd.DataFrame,
    v0: pd.Series,
    quote: pd.Series | float | None = None,
) -> pd.Series:
    """Predict conditional future clean value m(q, X) = V0 - side_sign * A(q, X).

    :param model: Fitted selection model.
    :param df: RFQ dataframe.
    :param v0: Unconditional future-value forecast for each row.
    :param quote: Optional counterfactual quote override.
    :return: Conditional mark aligned to ``df.index``.
    """

    selection = predict_selection(model, df, quote=quote)
    adjusted = v0.astype(float) - df["side_sign"].astype(float) * selection
    return adjusted.rename("conditional_mark")


def evaluate_selection_model(
    train_fills_df: pd.DataFrame,
    test_fills_df: pd.DataFrame,
    config: SelectionModelConfig | None = None,
    v0_column: str = V0_OOF_COLUMN,
) -> dict[str, Any]:
    """Evaluate chronological adverse-selection performance on fills.

    :param train_fills_df: Chronological training fills.
    :param test_fills_df: Chronological test fills.
    :param config: Selection-model configuration.
    :param v0_column: Column containing out-of-fold V0 predictions.
    :return: Metrics including MAE and quote sensitivity.
    """

    model_config = config or SelectionModelConfig()
    _assert_chronological_split(train_fills_df, test_fills_df)
    fitted = fit_selection_model(train_fills_df, model_config, v0_column=v0_column)
    actual = make_selection_target(test_fills_df, v0_column=v0_column)
    predicted = predict_selection(fitted, test_fills_df)
    metrics = {
        "mae_selection": float(mean_absolute_error(actual, predicted)),
        "mean_actual_selection": float(actual.mean()),
        "mean_predicted_selection": float(predicted.mean()),
        "quote_sensitivity": _selection_quote_sensitivity(test_fills_df, fitted),
        "fitted_model": fitted,
    }
    metrics["conditional_mark_mae"] = float(
        mean_absolute_error(
            test_fills_df["y5"].astype(float),
            predict_conditional_mark(
                fitted,
                test_fills_df,
                v0=test_fills_df[v0_column].astype(float),
            ),
        )
    )
    return metrics


def counterfactual_selection_curve(
    model: FittedSelectionModel,
    df: pd.DataFrame,
    aggressiveness_values: tuple[float, ...] = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5),
) -> pd.DataFrame:
    """Mean predicted adverse selection across a counterfactual quote grid.

    :param model: Fitted selection model.
    :param df: RFQ dataframe to average over.
    :param aggressiveness_values: Normalized aggressiveness grid.
    :return: Frame with columns ``aggressiveness`` and ``selection_cents``.
    """

    records: list[dict[str, float]] = []
    for aggressiveness in aggressiveness_values:
        quote = (
            df["cp_plus"].astype(float)
            + df["side_sign"].astype(float) * aggressiveness * df["market_width"].astype(float)
        )
        mean_selection = float(predict_selection(model, df, quote=quote).mean())
        records.append(
            {
                "aggressiveness": aggressiveness,
                "selection_cents": mean_selection * 100.0,
            }
        )
    return pd.DataFrame(records)


def predicted_selection_by_liquidity(
    model: FittedSelectionModel,
    df: pd.DataFrame,
    n_buckets: int = 3,
) -> pd.DataFrame:
    """Mean predicted selection at historical quotes per liquidity bucket.

    :param model: Fitted selection model.
    :param df: RFQ dataframe.
    :param n_buckets: Number of liquidity quantile buckets.
    :return: Frame with columns ``bucket`` and ``selection_cents``.
    """

    labels = ["low", "medium", "high"][:n_buckets]
    buckets = pd.qcut(
        df["liquidity_score"].astype(float),
        q=n_buckets,
        labels=labels,
        duplicates="drop",
    )
    predictions = predict_selection(model, df)
    frame = pd.DataFrame({"bucket": buckets.astype(str), "selection": predictions})
    grouped = frame.groupby("bucket", sort=False, observed=True)["selection"].mean()
    ordered = [label for label in labels if label in grouped.index]
    return pd.DataFrame(
        {
            "bucket": ordered,
            "selection_cents": [float(grouped[label]) * 100.0 for label in ordered],
        }
    )


def selection_train_test_split(
    fills_df: pd.DataFrame,
    config: SelectionModelConfig | None = None,
) -> ChronologicalSplit:
    """Split fills chronologically for selection-model evaluation.

    :param fills_df: Fill-only dataframe.
    :param config: Selection-model configuration.
    :return: Chronological train and test partitions.
    """

    model_config = config or SelectionModelConfig()
    return chronological_train_test_split(fills_df, model_config.chronological_test_fraction)


def format_selection_metrics(metrics: dict[str, Any]) -> str:
    """Render a compact selection-model metrics table.

    :param metrics: Output from :func:`evaluate_selection_model`.
    :return: Human-readable summary string.
    """

    rows = [
        ("Selection MAE", f"{metrics['mae_selection']:.4f}"),
        ("Conditional mark MAE vs y5", f"{metrics['conditional_mark_mae']:.4f}"),
        ("Mean actual selection", f"{metrics['mean_actual_selection']:.4f}"),
        ("Mean predicted selection", f"{metrics['mean_predicted_selection']:.4f}"),
        ("Quote sensitivity", str(metrics["quote_sensitivity"])),
    ]
    name_width = max(len(row[0]) for row in rows)
    return "\n".join(f"{name:<{name_width}}  {value}" for name, value in rows)


def _build_ridge_pipeline(config: SelectionModelConfig) -> Pipeline:
    preprocessor = build_feature_preprocessor(
        numeric_features=SELECTION_NUMERIC_FEATURES,
        categorical_features=SELECTION_CATEGORICAL_FEATURES,
        minimum_category_frequency=config.minimum_category_frequency,
    )
    chronological_cv = TimeSeriesSplit(n_splits=config.number_of_cv_splits)
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


def _require_fills_with_v0(df: pd.DataFrame, v0_column: str) -> pd.DataFrame:
    if "won" not in df.columns:
        raise ValueError("won column is required to identify fills")
    if not df["won"].all():
        raise ValueError("selection model must be fit using won RFQs only")
    if v0_column not in df.columns:
        raise ValueError(f"{v0_column} is required for selection modelling")
    fills = df.loc[df[v0_column].notna()].copy()
    if fills.empty:
        raise ValueError("no fills with out-of-fold V0 available for selection fit")
    return fills


def _selection_quote_sensitivity(
    test_fills_df: pd.DataFrame,
    model: FittedSelectionModel,
) -> bool:
    if test_fills_df.empty:
        return True
    row = test_fills_df.iloc[[0]]
    low_quote = row["cp_plus"] + row["side_sign"] * (-1.0) * row["market_width"]
    high_quote = row["cp_plus"] + row["side_sign"] * (1.0) * row["market_width"]
    low_selection = float(predict_selection(model, row, quote=low_quote).iloc[0])
    high_selection = float(predict_selection(model, row, quote=high_quote).iloc[0])
    return low_selection != high_selection


def _assert_chronological_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_max = pd.to_datetime(train_df["timestamp"]).max()
    test_min = pd.to_datetime(test_df["timestamp"]).min()
    if test_min < train_max:
        raise ValueError("test_df must not begin before the final training timestamp")
