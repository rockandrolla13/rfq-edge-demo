"""Quote optimizer combining fill, selection, value, and cost models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from rfq_edge.config import (
    FillModelConfig,
    OptimizerConfig,
    SelectionModelConfig,
    ValueModelConfig,
)
from rfq_edge.costs import (
    points_to_cents,
    rfq_inventory_value_points,
    rfq_trading_cost_points,
)
from rfq_edge.fill_model import FittedFillModel, fit_fill_model, predict_win_probability
from rfq_edge.selection_model import (
    FittedSelectionModel,
    V0_OOF_COLUMN,
    fit_selection_model,
    predict_selection,
)
from rfq_edge.value_model import FittedValueModel, fit_value_model, predict_v0


@dataclass(frozen=True)
class FittedQuoteModels:
    """Calibrated models used by the quote optimizer.

    :param fill_model: Win-probability model p(win | q, X).
    :param selection_model: Adverse-selection model A(q, X) on fills.
    :param value_model: Unconditional future-value model V0.
    :param v0_column: Optional precomputed V0 column, e.g. out-of-fold values.
    """

    fill_model: FittedFillModel
    selection_model: FittedSelectionModel
    value_model: FittedValueModel
    v0_column: str | None = None


@dataclass(frozen=True)
class QuoteDecision:
    """Selected quote or a decline when every candidate has J <= 0.

    :param accepted: Whether a quote was chosen.
    :param quote: Selected clean quote, if accepted.
    :param expected_value_cents: Expected value of the selected quote in cents.
    :param candidate_table: Grid evaluation for all scanned quotes.
    """

    accepted: bool
    quote: float | None
    expected_value_cents: float
    candidate_table: pd.DataFrame


def fit_quote_models(
    train_df: pd.DataFrame,
    train_fills_df: pd.DataFrame,
    value_config: ValueModelConfig | None = None,
    fill_config: FillModelConfig | None = None,
    selection_config: SelectionModelConfig | None = None,
    v0_column: str = V0_OOF_COLUMN,
) -> FittedQuoteModels:
    """Fit value, fill, and selection models on chronological training data.

    :param train_df: All chronological training RFQs.
    :param train_fills_df: Won training fills with out-of-fold V0.
    :param value_config: Value-model configuration.
    :param fill_config: Fill-model configuration.
    :param selection_config: Selection-model configuration.
    :param v0_column: Column containing out-of-fold V0 for selection fit.
    :return: Bundle of fitted models for quote optimization.
    """

    value_model = fit_value_model(train_df, value_config)
    fill_model = fit_fill_model(train_df, fill_config)
    selection_model = fit_selection_model(
        train_fills_df,
        selection_config,
        v0_column=v0_column,
    )
    return FittedQuoteModels(
        fill_model=fill_model,
        selection_model=selection_model,
        value_model=value_model,
        v0_column=v0_column,
    )


def aggressiveness_grid(config: OptimizerConfig) -> np.ndarray:
    """Return the normalized aggressiveness grid for quote search.

    :param config: Optimizer configuration.
    :return: One-dimensional aggressiveness array.
    """

    values = np.arange(
        config.min_aggressiveness,
        config.max_aggressiveness + config.aggressiveness_step / 2.0,
        config.aggressiveness_step,
    )
    if values.size == 0:
        raise ValueError("aggressiveness grid is empty")
    return values


def quote_from_aggressiveness(row: pd.Series, aggressiveness: float) -> float:
    """Convert normalized aggressiveness into a clean quote.

    :param row: Single RFQ row.
    :param aggressiveness: Normalized aggressiveness z.
    :return: Candidate clean quote in price points.
    """

    return float(
        row["cp_plus"]
        + float(row["side_sign"]) * aggressiveness * float(row["market_width"])
    )


def resolve_v0(row_frame: pd.DataFrame, models: FittedQuoteModels) -> pd.Series:
    """Resolve unconditional V0 for each RFQ row.

    Out-of-fold V0 is preferred when available; remaining rows fall back to the
    fitted value model so quote optimization can proceed at inference time.

    :param row_frame: RFQ dataframe.
    :param models: Fitted quote models.
    :return: V0 series aligned to ``row_frame.index``.
    """

    predicted_v0 = predict_v0(models.value_model, row_frame)
    if models.v0_column is None or models.v0_column not in row_frame.columns:
        return predicted_v0
    v0 = row_frame[models.v0_column].astype(float)
    missing = v0.isna()
    if missing.any():
        v0 = v0.copy()
        v0.loc[missing] = predicted_v0.loc[missing]
    return v0


def evaluate_quote_grid(
    rfq: pd.DataFrame,
    models: FittedQuoteModels,
    config: OptimizerConfig | None = None,
) -> pd.DataFrame:
    """Evaluate J(q, X) over a grid of candidate quotes.

    For each quote the function computes:

    * p(q, X) from the fill model;
    * A(q, X) from the selection model;
    * m(q, X) = V0 - side_sign * A(q, X);
    * e(q, X) = side_sign * (m(q, X) - q);
    * J(q, X) = p(q, X) * [e(q, X) - cost + inventory value].

    :param rfq: Single-row RFQ dataframe.
    :param models: Fitted quote models.
    :param config: Optimizer configuration.
    :return: Candidate quote comparison table.
    :raises ValueError: If ``rfq`` does not contain exactly one row.
    """

    if len(rfq) != 1:
        raise ValueError("evaluate_quote_grid expects exactly one RFQ row")
    optimizer_config = config or OptimizerConfig()
    row = rfq.iloc[0]
    v0 = float(resolve_v0(rfq, models).iloc[0])
    cost_points = rfq_trading_cost_points(row, optimizer_config)
    inventory_value_points = rfq_inventory_value_points(row, optimizer_config)
    cost_cents = points_to_cents(cost_points)
    inventory_value_cents = points_to_cents(inventory_value_points)

    support_low, support_high = models.fill_model.aggressiveness_support
    rows: list[dict[str, float]] = []
    for aggressiveness in aggressiveness_grid(optimizer_config):
        quote = quote_from_aggressiveness(row, float(aggressiveness))
        quote_series = pd.Series([quote], index=rfq.index)
        p_win = float(predict_win_probability(models.fill_model, rfq, quote=quote_series).iloc[0])
        if not np.isfinite(p_win):
            p_win = 0.0
        selection = float(predict_selection(models.selection_model, rfq, quote=quote_series).iloc[0])
        if not np.isfinite(selection):
            selection = 0.0
        post_win_value = v0 - float(row["side_sign"]) * selection
        clean_edge_points = float(row["side_sign"]) * (post_win_value - quote)
        clean_edge_cents = points_to_cents(clean_edge_points)
        expected_value_cents = p_win * (
            clean_edge_cents - cost_cents + inventory_value_cents
        )
        in_support = support_low <= float(aggressiveness) <= support_high
        rows.append(
            {
                "quote": quote,
                "aggressiveness": float(aggressiveness),
                "p_win": p_win,
                "post_win_value": post_win_value,
                "selection": selection,
                "clean_edge_cents": clean_edge_cents,
                "cost_cents": cost_cents,
                "inventory_value_cents": inventory_value_cents,
                "expected_value_cents": expected_value_cents,
                "in_support": in_support,
            }
        )
    return pd.DataFrame(rows)


def optimize_quote(
    rfq: pd.DataFrame,
    models: FittedQuoteModels,
    config: OptimizerConfig | None = None,
) -> QuoteDecision:
    """Select the quote with the highest expected value, or decline.

    Only candidates inside the trained aggressiveness support are eligible,
    because model outputs outside historical quote coverage are extrapolations.

    :param rfq: Single-row RFQ dataframe.
    :param models: Fitted quote models.
    :param config: Optimizer configuration.
    :return: Accepted quote decision or a decline.
    """

    table = evaluate_quote_grid(rfq, models, config)
    candidates = table.loc[table["in_support"]]
    if candidates.empty:
        # No candidate quote lies inside historical coverage, so any model
        # output would be extrapolation; decline rather than guess.
        return QuoteDecision(
            accepted=False,
            quote=None,
            expected_value_cents=float("nan"),
            candidate_table=table,
        )
    if candidates["expected_value_cents"].notna().any():
        best = candidates.loc[candidates["expected_value_cents"].idxmax()]
    else:
        best = candidates.iloc[0]
    best_expected_value = float(best["expected_value_cents"])
    if best_expected_value <= 0.0:
        return QuoteDecision(
            accepted=False,
            quote=None,
            expected_value_cents=best_expected_value,
            candidate_table=table,
        )
    return QuoteDecision(
        accepted=True,
        quote=float(best["quote"]),
        expected_value_cents=best_expected_value,
        candidate_table=table,
    )


def format_quote_table(table: pd.DataFrame) -> str:
    """Render the candidate quote comparison table.

    :param table: Output from :func:`evaluate_quote_grid`.
    :return: Human-readable table string.
    """

    headers = [
        "Quote",
        "Win probability",
        "Post-win value",
        "Clean edge",
        "Cost",
        "Expected value",
    ]
    lines = ["  ".join(f"{header:<16}" for header in headers)]
    for _, row in table.iterrows():
        lines.append(
            "  ".join(
                [
                    f"{row['quote']:<16.2f}",
                    f"{row['p_win'] * 100:.0f}%".ljust(16),
                    f"{row['post_win_value']:<16.2f}",
                    f"{row['clean_edge_cents']:<16.0f}c",
                    f"{row['cost_cents']:<16.1f}c",
                    f"{row['expected_value_cents']:<16.1f}c",
                ]
            )
        )
    return "\n".join(lines)


def demonstrate_quote_optimizer(
    rfq: pd.DataFrame,
    models: FittedQuoteModels,
    config: OptimizerConfig | None = None,
) -> dict[str, Any]:
    """Evaluate and optimize one RFQ, returning table and decision metadata.

    :param rfq: Single-row RFQ dataframe.
    :param models: Fitted quote models.
    :param config: Optimizer configuration.
    :return: Dictionary with table text and optimization decision.
    """

    table = evaluate_quote_grid(rfq, models, config)
    decision = optimize_quote(rfq, models, config)
    return {
        "table": table,
        "table_text": format_quote_table(table),
        "decision": decision,
    }
