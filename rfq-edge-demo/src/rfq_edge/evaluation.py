"""Evaluation metrics for unconditional future-value forecasts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rfq_edge.config import ValueModelConfig

PRICE_POINTS_TO_CENTS = 100.0
MATERIAL_RESIDUAL_THRESHOLD = 0.05


def capped_size_weights(sizes: pd.Series, size_weight_cap: float) -> pd.Series:
    """Return mean-one RFQ weights with a cap on very large tickets.

    :param sizes: RFQ sizes in notional units.
    :param size_weight_cap: Maximum relative weight before normalization.
    :return: Normalized weights with mean one.
    """

    raw_weights = np.log1p(sizes.astype(float))
    capped = np.minimum(raw_weights, size_weight_cap)
    normalized = capped / capped.mean()
    return pd.Series(normalized, index=sizes.index)


def liquidity_buckets(liquidity_score: pd.Series, n_buckets: int = 3) -> pd.Series:
    """Assign RFQs to labelled liquidity quantile buckets.

    :param liquidity_score: Liquidity scores in [0, 1].
    :param n_buckets: Number of quantile buckets (max 3 labels).
    :return: String bucket labels aligned to the input index.
    """

    labels = ["low", "medium", "high"][:n_buckets]
    return pd.qcut(
        liquidity_score.astype(float),
        q=n_buckets,
        labels=labels,
        duplicates="drop",
    ).astype(str)


def compute_forecast_metrics(
    actual_prices: pd.Series,
    predicted_prices: pd.Series,
    cp_plus: pd.Series,
    sizes: pd.Series,
    config: ValueModelConfig,
    group_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute price and residual forecast metrics for one V0 model.

    :param actual_prices: Realized t+5 clean prices ``y5``.
    :param predicted_prices: Forecast prices ``V0``.
    :param cp_plus: CP+ anchor prices at RFQ time.
    :param sizes: RFQ sizes used for weighted errors.
    :param config: Value-model configuration.
    :param group_frame: Optional columns used for segmented reporting.
    :return: Metric dictionary with headline and segmented statistics.
    """

    actual = actual_prices.astype(float)
    predicted = predicted_prices.astype(float)
    errors = predicted - actual
    residual_actual = actual - cp_plus.astype(float)
    residual_predicted = predicted - cp_plus.astype(float)
    residual_errors = residual_predicted - residual_actual
    weights = capped_size_weights(sizes, config.size_weight_cap)

    metrics: dict[str, Any] = {
        "mae_price_points": float(np.abs(errors).mean()),
        "mae_cents": float(np.abs(errors).mean() * PRICE_POINTS_TO_CENTS),
        "weighted_mae_price_points": float((np.abs(errors) * weights).mean()),
        "weighted_mae_cents": float((np.abs(errors) * weights).mean() * PRICE_POINTS_TO_CENTS),
        "bias_price_points": float(errors.mean()),
        "bias_cents": float(errors.mean() * PRICE_POINTS_TO_CENTS),
        "rmse_price_points": float(np.sqrt((errors**2).mean())),
        "rmse_cents": float(np.sqrt((errors**2).mean()) * PRICE_POINTS_TO_CENTS),
        "directional_accuracy": _directional_accuracy(
            residual_actual=residual_actual,
            residual_predicted=residual_predicted,
        ),
        "calibration_by_predicted_residual_bucket": _calibration_table(
            residual_actual=residual_actual,
            residual_predicted=residual_predicted,
        ),
    }

    if group_frame is not None:
        metrics["by_rating"] = _grouped_mae(
            errors=errors,
            groups=group_frame["rating_bucket"],
        )
        metrics["by_liquidity_bucket"] = _grouped_mae(
            errors=errors,
            groups=_liquidity_bucket(group_frame["liquidity_score"]),
        )
        metrics["by_regime"] = _grouped_mae(
            errors=errors,
            groups=group_frame["regime"],
        )
        metrics["by_bond_history_bucket"] = _grouped_mae(
            errors=errors,
            groups=_bond_history_bucket(group_frame),
        )
    return metrics


def format_metrics_table(metrics_by_model: dict[str, dict[str, Any]]) -> str:
    """Render a compact comparison table for multiple V0 forecasts.

    :param metrics_by_model: Mapping from model name to metric dictionary.
    :return: Human-readable table string.
    """

    headers = ["model", "mae_pts", "mae_cents", "w_mae_pts", "bias_pts", "rmse_pts", "dir_acc"]
    rows = [headers]
    for model_name, metrics in metrics_by_model.items():
        rows.append(
            [
                model_name,
                f"{metrics['mae_price_points']:.4f}",
                f"{metrics['mae_cents']:.2f}",
                f"{metrics['weighted_mae_price_points']:.4f}",
                f"{metrics['bias_price_points']:.4f}",
                f"{metrics['rmse_price_points']:.4f}",
                f"{metrics['directional_accuracy']:.3f}",
            ]
        )
    column_widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]
    lines = [
        "  ".join(value.ljust(column_widths[index]) for index, value in enumerate(row))
        for row in rows
    ]
    return "\n".join(lines)


def _directional_accuracy(
    residual_actual: pd.Series,
    residual_predicted: pd.Series,
) -> float:
    material = residual_actual.abs() >= MATERIAL_RESIDUAL_THRESHOLD
    if int(material.sum()) == 0:
        return float("nan")
    same_sign = np.sign(residual_actual[material]) == np.sign(residual_predicted[material])
    return float(same_sign.mean())


def _calibration_table(
    residual_actual: pd.Series,
    residual_predicted: pd.Series,
) -> list[dict[str, float]]:
    frame = pd.DataFrame(
        {
            "residual_actual": residual_actual,
            "residual_predicted": residual_predicted,
        }
    )
    frame["bucket"] = pd.qcut(
        frame["residual_predicted"],
        q=5,
        duplicates="drop",
    )
    rows: list[dict[str, float]] = []
    for bucket, group in frame.groupby("bucket", observed=False):
        rows.append(
            {
                "bucket": str(bucket),
                "mean_predicted_residual": float(group["residual_predicted"].mean()),
                "mean_actual_residual": float(group["residual_actual"].mean()),
                "count": float(len(group)),
            }
        )
    return rows


def _grouped_mae(errors: pd.Series, groups: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"error": errors.abs(), "group": groups.astype(str)})
    grouped = frame.groupby("group", observed=False)["error"].mean()
    return {str(key): float(value) for key, value in grouped.items()}


def _liquidity_bucket(liquidity_score: pd.Series) -> pd.Series:
    return pd.qcut(
        liquidity_score.astype(float),
        q=3,
        labels=["low", "medium", "high"],
    ).astype(str)


def _bond_history_bucket(group_frame: pd.DataFrame) -> pd.Series:
    if "bond_history_count" in group_frame.columns:
        counts = group_frame["bond_history_count"].astype(float)
    else:
        counts = pd.Series(1.0, index=group_frame.index)
    return pd.qcut(
        counts,
        q=3,
        labels=["sparse", "moderate", "active"],
        duplicates="drop",
    ).astype(str)
