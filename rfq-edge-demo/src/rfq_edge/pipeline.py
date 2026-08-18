"""Compose value, fill, selection, and cost models into quote decisions."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rfq_edge.config import (
    FillModelConfig,
    OptimizerConfig,
    SelectionModelConfig,
    ValueModelConfig,
)
from rfq_edge.costs import CostParams
from rfq_edge.optimizer import FittedQuoteModels, fit_quote_models
from rfq_edge.responders import (
    compare_responders,
    observable_view,
    scan_responder_grid,
)
from rfq_edge.splits import chronological_train_test_split
from rfq_edge.value_model import make_chronological_oof_v0
from rfq_edge.responder_fill import FillModelParams
from rfq_edge.objective import ResponderModels
from rfq_edge.responder_optimizer import (
    OptimizerParams,
    QuoteSolution,
    maximize_expected_pnl,
    solve_consistent_edge,
)
from rfq_edge.responder_selection import SelectionModelParams
from rfq_edge.synthetic import RfqRequest
from rfq_edge.responder_value import (
    FairValue,
    ValueModelParams,
    estimate_fair_value,
    quoted_price,
)


@dataclass(frozen=True)
class FittedFramework:
    """All fitted components plus the chronological data partitions.

    :param models: Fitted value, fill, and selection models.
    :param train_df: Training rows with out-of-fold V0 merged in.
    :param test_df: Held-out rows with out-of-fold V0 merged in. Latent
        columns are preserved here for oracle diagnostics only.
    :param value_config: Value-model configuration used.
    :param fill_config: Fill-model configuration used.
    :param selection_config: Selection-model configuration used.
    """

    models: FittedQuoteModels
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    value_config: ValueModelConfig
    fill_config: FillModelConfig
    selection_config: SelectionModelConfig


def fit_framework(
    df: pd.DataFrame,
    value_config: ValueModelConfig | None = None,
    fill_config: FillModelConfig | None = None,
    selection_config: SelectionModelConfig | None = None,
) -> FittedFramework:
    """Fit the complete responder framework on a chronological split.

    Latent simulator columns, if present, are stripped before any model
    fitting or prediction; they survive only in the returned dataframes so
    oracle diagnostics can use them downstream.

    :param df: Full RFQ history, optionally including latent columns.
    :param value_config: Value-model configuration.
    :param fill_config: Fill-model configuration.
    :param selection_config: Selection-model configuration.
    :return: Fitted models plus prepared train and test partitions.
    """

    resolved_value_config = value_config or ValueModelConfig()
    resolved_fill_config = fill_config or FillModelConfig()
    resolved_selection_config = selection_config or SelectionModelConfig()

    observable = observable_view(df)
    oof = make_chronological_oof_v0(observable, resolved_value_config)
    prepared = df.merge(oof[["rfq_id", "v0_oof"]], on="rfq_id", how="left")
    split = chronological_train_test_split(
        prepared, resolved_value_config.chronological_test_fraction
    )
    train_observable = observable_view(split.train_df)
    train_fills = train_observable.loc[
        train_observable["won"] & train_observable["v0_oof"].notna()
    ]
    models = fit_quote_models(
        train_observable,
        train_fills,
        resolved_value_config,
        resolved_fill_config,
        resolved_selection_config,
    )
    return FittedFramework(
        models=models,
        train_df=split.train_df,
        test_df=split.test_df,
        value_config=resolved_value_config,
        fill_config=resolved_fill_config,
        selection_config=resolved_selection_config,
    )


def score_rfq(
    framework: FittedFramework,
    rfq: pd.DataFrame,
    optimizer_config: OptimizerConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Scan the candidate grid and compare responders for one RFQ.

    :param framework: Fitted framework.
    :param rfq: Single-row RFQ dataframe.
    :param optimizer_config: Grid, cost, and inventory calibration.
    :return: Dict with the candidate ``grid`` and the responder ``comparison``.
    """

    grid = scan_responder_grid(rfq, framework.models, optimizer_config)
    comparison = compare_responders(rfq, framework.models, optimizer_config)
    return {
        "grid": grid,
        "comparison": comparison,
    }


def cold_start_comparison(
    framework: FittedFramework,
    optimizer_config: OptimizerConfig | None = None,
) -> pd.DataFrame:
    """Compare predictions across bond-history regimes, including cold starts.

    Four cases are built from the held-out test set: the most active bond,
    a sparse bond, an unseen bond from a known issuer, and a bond from an
    unseen issuer. The unseen cases clone a real RFQ and relabel identifiers,
    so pooled categorical encoders must fall back to issuer or population
    behavior.

    :param framework: Fitted framework.
    :param optimizer_config: Grid, cost, and inventory calibration.
    :return: One row per case with V0, predicted selection, and the
        edge-consistent decision.
    """

    from rfq_edge.responders import ResponderKind, respond_to_rfq
    from rfq_edge.selection_model import predict_selection
    from rfq_edge.value_model import predict_v0

    test_observable = observable_view(framework.test_df)
    train_observable = observable_view(framework.train_df)
    bond_counts = train_observable.groupby("bond_id").size()

    tradeable = test_observable.loc[test_observable["bond_id"].isin(bond_counts.index)]
    counts_for_test = tradeable["bond_id"].map(bond_counts)
    active_row = tradeable.loc[[counts_for_test.idxmax()]]
    sparse_row = tradeable.loc[[counts_for_test.idxmin()]]

    unseen_bond_row = sparse_row.copy()
    unseen_bond_row["bond_id"] = "bond-unseen-0001"
    unseen_issuer_row = sparse_row.copy()
    unseen_issuer_row["bond_id"] = "bond-unseen-0002"
    unseen_issuer_row["issuer_id"] = "issuer-unseen-01"

    cases = {
        "active bond": active_row,
        "sparse bond": sparse_row,
        "unseen bond, known issuer": unseen_bond_row,
        "unseen issuer": unseen_issuer_row,
    }
    records: list[dict[str, object]] = []
    for case_name, row in cases.items():
        v0 = float(predict_v0(framework.models.value_model, row).iloc[0])
        selection = float(
            predict_selection(framework.models.selection_model, row).iloc[0]
        )
        decision = respond_to_rfq(
            row, framework.models, ResponderKind.EDGE_CONSISTENT, optimizer_config
        )
        records.append(
            {
                "case": case_name,
                "bond_id": str(row.iloc[0]["bond_id"]),
                "issuer_id": str(row.iloc[0]["issuer_id"]),
                "train_rfqs_on_bond": int(bond_counts.get(str(row.iloc[0]["bond_id"]), 0)),
                "cp_plus": float(row.iloc[0]["cp_plus"]),
                "predicted_v0": v0,
                "v0_minus_cp_plus_cents": (v0 - float(row.iloc[0]["cp_plus"])) * 100.0,
                "predicted_selection_cents": selection * 100.0,
                "accepted": decision.accepted,
                "selected_quote": decision.quote,
                "expected_value_cents": decision.expected_value_cents,
            }
        )
    return pd.DataFrame(records)


@dataclass(frozen=True)
class PipelineConfig:
    """Responder models plus the edge search used to quote a book.

    :param models: Fill, selection, cost, value, and target-edge calibration.
    :param search: Bounds and grid step for quote rules.
    """

    models: ResponderModels
    search: OptimizerParams


@dataclass(frozen=True)
class ResponderDecision:
    """Quotes produced for one RFQ under both responder rules.

    :param request: Incoming RFQ.
    :param fair_value: Inventory-adjusted reservation mark.
    :param consistent: Quote that earns the target net edge if filled.
    :param optimal: Quote that maximizes expected PnL on the search grid.
    :param consistent_price: Client-facing price for the consistent quote.
    :param optimal_price: Client-facing price for the expected-PnL quote.
    """

    request: RfqRequest
    fair_value: FairValue
    consistent: QuoteSolution
    optimal: QuoteSolution
    consistent_price: float
    optimal_price: float


def default_config() -> PipelineConfig:
    """Return the demo calibration used by the notebook and tests.

    :return: Models and search bounds that admit an interior consistent edge.
    """

    models = ResponderModels(
        value=ValueModelParams(inventory_skew=0.00002),
        fill=FillModelParams(
            intercept=3.2,
            edge_coef=0.12,
            competition_coef=0.18,
            size_coef=0.12,
        ),
        selection=SelectionModelParams(
            base_bps=1.5,
            scale=0.08,
            decay=0.09,
        ),
        costs=CostParams(
            transaction_bps=0.8,
            inventory_bps_per_unit=0.0015,
            risk_aversion=8.0,
        ),
        target_edge_bps=2.0,
    )
    search = OptimizerParams(
        min_edge_bps=0.5,
        max_edge_bps=40.0,
        step_bps=0.25,
    )
    return PipelineConfig(models=models, search=search)


def run_responder(request: RfqRequest, config: PipelineConfig) -> ResponderDecision:
    """Value one RFQ and quote it with both responder rules.

    :param request: Incoming RFQ.
    :param config: Models and search bounds.
    :return: Fair value, both quote rules, and client-facing prices.
    """

    fair_value = estimate_fair_value(request, config.models.value)
    consistent = solve_consistent_edge(request, config.models, config.search)
    optimal = maximize_expected_pnl(request, config.models, config.search)
    consistent_px = quoted_price(
        request,
        fair_value,
        consistent.components.quoted_edge_bps,
    )
    optimal_px = quoted_price(
        request,
        fair_value,
        optimal.components.quoted_edge_bps,
    )
    return ResponderDecision(
        request=request,
        fair_value=fair_value,
        consistent=consistent,
        optimal=optimal,
        consistent_price=consistent_px,
        optimal_price=optimal_px,
    )


def run_book(
    requests: tuple[RfqRequest, ...],
    config: PipelineConfig,
) -> tuple[ResponderDecision, ...]:
    """Quote every RFQ in a book with the same calibration.

    :param requests: RFQ book in evaluation order.
    :param config: Models and search bounds.
    :return: One decision per request, in the same order.
    :raises ValueError: If the book is empty.
    """

    if not requests:
        raise ValueError("requests must not be empty")
    return tuple(run_responder(request, config) for request in requests)
