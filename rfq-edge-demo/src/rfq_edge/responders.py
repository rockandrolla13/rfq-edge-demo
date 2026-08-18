"""Three RFQ responders sharing one grid, fill model, costs, and support rule.

All responders scan the same candidate quote grid, use the same fitted fill
probability p(q, X), the same cost and inventory-value functions, the same
quote-support restriction, and the same decline option. They differ only in
the post-win value they assume:

* plain CP+ responder:      m(q, X) = CP+
* plain V0 responder:       m(q, X) = V0
* edge-consistent responder: m(q, X) = V0 - side_sign * haircut * A(q, X)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from rfq_edge.config import OptimizerConfig
from rfq_edge.costs import (
    points_to_cents,
    rfq_inventory_value_points,
    rfq_trading_cost_points,
)
from rfq_edge.fill_model import predict_win_probability
from rfq_edge.optimizer import (
    FittedQuoteModels,
    aggressiveness_grid,
    quote_from_aggressiveness,
    resolve_v0,
)
from rfq_edge.selection_model import predict_selection


class ResponderKind(Enum):
    """Identity of a quoting policy compared in the demo."""

    PLAIN_CP_PLUS = "plain_cp_plus"
    PLAIN_V0 = "plain_v0"
    EDGE_CONSISTENT = "edge_consistent"


RESPONDER_LABELS: dict[ResponderKind, str] = {
    ResponderKind.PLAIN_CP_PLUS: "Plain CP+",
    ResponderKind.PLAIN_V0: "Plain V0",
    ResponderKind.EDGE_CONSISTENT: "Edge-consistent",
}

ALL_RESPONDERS: tuple[ResponderKind, ...] = (
    ResponderKind.PLAIN_CP_PLUS,
    ResponderKind.PLAIN_V0,
    ResponderKind.EDGE_CONSISTENT,
)


@dataclass(frozen=True)
class ResponderQuote:
    """One responder's decision for one RFQ.

    :param kind: Responder identity.
    :param accepted: Whether the responder quotes or declines.
    :param quote: Selected clean quote, or None on decline.
    :param aggressiveness: Normalized aggressiveness of the selected quote.
    :param p_win: Predicted win probability at the selected quote.
    :param selection_points: Predicted adverse selection at the selected quote.
    :param post_win_value: Post-win value assumed by this responder.
    :param apparent_edge_cents: side_sign * (V0 - q) in cents, ignoring selection.
    :param conditional_edge_cents: side_sign * (m - q) in cents under this responder.
    :param cost_cents: Quote-independent cost in cents.
    :param inventory_value_cents: Inventory preference value in cents.
    :param expected_value_cents: Responder objective at the selected quote.
    """

    kind: ResponderKind
    accepted: bool
    quote: float | None
    aggressiveness: float | None
    p_win: float
    selection_points: float
    post_win_value: float
    apparent_edge_cents: float
    conditional_edge_cents: float
    cost_cents: float
    inventory_value_cents: float
    expected_value_cents: float


def observable_view(df: pd.DataFrame) -> pd.DataFrame:
    """Return the dataframe without any latent simulator columns.

    Policy evaluation carries latent columns for oracle diagnostics; fitted
    models must never receive them, so predictions go through this view.

    :param df: RFQ dataframe possibly containing latent columns.
    :return: Same rows restricted to observable columns.
    """

    latent_columns = [column for column in df.columns if column.startswith("latent_")]
    if latent_columns:
        return df.drop(columns=latent_columns)
    return df


def scan_responder_grid(
    rfq: pd.DataFrame,
    models: FittedQuoteModels,
    config: OptimizerConfig | None = None,
) -> pd.DataFrame:
    """Evaluate every candidate quote for all three responders at once.

    Shared quantities (p, A, cost, inventory value, support) are computed one
    time per candidate so the responders differ only in their objective.

    :param rfq: Single-row RFQ dataframe.
    :param models: Fitted quote models.
    :param config: Optimizer configuration.
    :return: Grid table with shared columns plus per-responder objectives.
    :raises ValueError: If ``rfq`` does not contain exactly one row.
    """

    if len(rfq) != 1:
        raise ValueError("scan_responder_grid expects exactly one RFQ row")
    optimizer_config = config or OptimizerConfig()
    observable = observable_view(rfq)
    row = observable.iloc[0]
    side_sign = float(row["side_sign"])
    cp_plus = float(row["cp_plus"])
    v0 = float(resolve_v0(observable, models).iloc[0])
    cost_cents = points_to_cents(rfq_trading_cost_points(row, optimizer_config))
    inventory_value_cents = points_to_cents(
        rfq_inventory_value_points(row, optimizer_config)
    )
    support_low, support_high = models.fill_model.aggressiveness_support

    records: list[dict[str, float | bool]] = []
    for aggressiveness in aggressiveness_grid(optimizer_config):
        quote = quote_from_aggressiveness(row, float(aggressiveness))
        quote_series = pd.Series([quote], index=observable.index)
        p_win = float(
            predict_win_probability(models.fill_model, observable, quote=quote_series).iloc[0]
        )
        if not np.isfinite(p_win):
            p_win = 0.0
        selection = float(
            predict_selection(models.selection_model, observable, quote=quote_series).iloc[0]
        )
        if not np.isfinite(selection):
            selection = 0.0

        record: dict[str, float | bool] = {
            "quote": quote,
            "aggressiveness": float(aggressiveness),
            "in_support": bool(support_low <= float(aggressiveness) <= support_high),
            "p_win": p_win,
            "selection_points": selection,
            "cost_cents": cost_cents,
            "inventory_value_cents": inventory_value_cents,
        }
        for kind in ALL_RESPONDERS:
            post_win_value = _post_win_value(
                kind=kind,
                cp_plus=cp_plus,
                v0=v0,
                side_sign=side_sign,
                selection=selection,
                haircut=optimizer_config.selection_haircut,
            )
            edge_cents = points_to_cents(side_sign * (post_win_value - quote))
            expected_value = p_win * (edge_cents - cost_cents + inventory_value_cents)
            record[f"post_win_value_{kind.value}"] = post_win_value
            record[f"edge_cents_{kind.value}"] = edge_cents
            record[f"expected_value_cents_{kind.value}"] = expected_value
        record["apparent_edge_cents"] = points_to_cents(side_sign * (v0 - quote))
        records.append(record)
    return pd.DataFrame(records)


def respond_to_rfq(
    rfq: pd.DataFrame,
    models: FittedQuoteModels,
    kind: ResponderKind,
    config: OptimizerConfig | None = None,
) -> ResponderQuote:
    """Choose a quote (or decline) for one RFQ under one responder.

    The responder quotes the supported candidate with the highest objective,
    and declines when every supported candidate has a nonpositive objective.

    :param rfq: Single-row RFQ dataframe.
    :param models: Fitted quote models.
    :param kind: Responder identity.
    :param config: Optimizer configuration.
    :return: The responder's decision.
    """

    grid = scan_responder_grid(rfq, models, config)
    return choose_from_grid(grid, kind)


def choose_from_grid(grid: pd.DataFrame, kind: ResponderKind) -> ResponderQuote:
    """Apply one responder's decision rule to a precomputed grid.

    :param grid: Output of :func:`scan_responder_grid`.
    :param kind: Responder identity.
    :return: The responder's decision.
    :raises ValueError: If the grid is empty.
    """

    if grid.empty:
        raise ValueError("candidate grid must not be empty")
    objective_column = f"expected_value_cents_{kind.value}"
    supported = grid.loc[grid["in_support"]]
    if supported.empty or not supported[objective_column].notna().any():
        return _decline(grid, kind)
    best = supported.loc[supported[objective_column].idxmax()]
    if float(best[objective_column]) <= 0.0:
        return _decline(grid, kind)
    return ResponderQuote(
        kind=kind,
        accepted=True,
        quote=float(best["quote"]),
        aggressiveness=float(best["aggressiveness"]),
        p_win=float(best["p_win"]),
        selection_points=float(best["selection_points"]),
        post_win_value=float(best[f"post_win_value_{kind.value}"]),
        apparent_edge_cents=float(best["apparent_edge_cents"]),
        conditional_edge_cents=float(best[f"edge_cents_{kind.value}"]),
        cost_cents=float(best["cost_cents"]),
        inventory_value_cents=float(best["inventory_value_cents"]),
        expected_value_cents=float(best[objective_column]),
    )


def compare_responders(
    rfq: pd.DataFrame,
    models: FittedQuoteModels,
    config: OptimizerConfig | None = None,
) -> pd.DataFrame:
    """Compare all three responders on the same RFQ and grid.

    :param rfq: Single-row RFQ dataframe.
    :param models: Fitted quote models.
    :param config: Optimizer configuration.
    :return: One row per responder with its decision and objective terms.
    """

    grid = scan_responder_grid(rfq, models, config)
    records: list[dict[str, object]] = []
    for kind in ALL_RESPONDERS:
        decision = choose_from_grid(grid, kind)
        records.append(
            {
                "responder": RESPONDER_LABELS[kind],
                "kind": kind.value,
                "accepted": decision.accepted,
                "quote": decision.quote,
                "aggressiveness": decision.aggressiveness,
                "p_win": decision.p_win,
                "selection_points": decision.selection_points,
                "post_win_value": decision.post_win_value,
                "apparent_edge_cents": decision.apparent_edge_cents,
                "conditional_edge_cents": decision.conditional_edge_cents,
                "cost_cents": decision.cost_cents,
                "inventory_value_cents": decision.inventory_value_cents,
                "expected_value_cents": decision.expected_value_cents,
            }
        )
    return pd.DataFrame(records)


def _post_win_value(
    kind: ResponderKind,
    cp_plus: float,
    v0: float,
    side_sign: float,
    selection: float,
    haircut: float,
) -> float:
    if kind is ResponderKind.PLAIN_CP_PLUS:
        return cp_plus
    if kind is ResponderKind.PLAIN_V0:
        return v0
    if kind is ResponderKind.EDGE_CONSISTENT:
        return v0 - side_sign * haircut * selection
    raise ValueError(f"unsupported responder kind: {kind}")


def _decline(grid: pd.DataFrame, kind: ResponderKind) -> ResponderQuote:
    objective_column = f"expected_value_cents_{kind.value}"
    best_available = float(grid[objective_column].max())
    return ResponderQuote(
        kind=kind,
        accepted=False,
        quote=None,
        aggressiveness=None,
        p_win=0.0,
        selection_points=0.0,
        post_win_value=float("nan"),
        apparent_edge_cents=0.0,
        conditional_edge_cents=0.0,
        cost_cents=float(grid["cost_cents"].iloc[0]),
        inventory_value_cents=float(grid["inventory_value_cents"].iloc[0]),
        expected_value_cents=best_available,
    )
