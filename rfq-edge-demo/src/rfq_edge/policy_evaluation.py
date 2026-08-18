"""Held-out policy evaluation for RFQ responders with a synthetic oracle.

Every responder is evaluated on the same held-out chronological RFQs, the
same candidate grid, and the same seeded uniform draws (common random
numbers), so differences in outcomes come from quoting decisions rather
than sampling noise. Realized fills are simulated from the oracle's exact
counterfactual fill probability; realized values use the dataset's y5,
which the simulator draws independently of the quote.

Simulated clean value is a diagnostic quantity. It is not real trading PnL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rfq_edge.config import OptimizerConfig
from rfq_edge.costs import (
    points_to_cents,
    rfq_inventory_value_points,
    rfq_trading_cost_points,
)
from rfq_edge.fill_model import predict_win_probability
from rfq_edge.optimizer import FittedQuoteModels, aggressiveness_grid, resolve_v0
from rfq_edge.responders import (
    ALL_RESPONDERS,
    RESPONDER_LABELS,
    ResponderKind,
    observable_view,
)
from rfq_edge.selection_model import predict_selection
from rfq_edge.simulation_diagnostics import OracleContext, oracle_fill_probability

DEFAULT_EVALUATION_SEED = 20_260_818
DEFAULT_BOOTSTRAP_SAMPLES = 200


@dataclass(frozen=True)
class PolicyEvaluationResult:
    """Outputs of one held-out policy comparison.

    :param decisions: One row per RFQ and responder with the decision and
        the simulated outcome.
    :param summary: One row per responder with aggregate metrics.
    :param segment_summaries: Net clean value per RFQ split by side,
        liquidity bucket, rating bucket, and regime.
    :param bootstrap: Block-bootstrap confidence intervals for net clean
        value per RFQ, one row per responder.
    """

    decisions: pd.DataFrame
    summary: pd.DataFrame
    segment_summaries: dict[str, pd.DataFrame]
    bootstrap: pd.DataFrame


def evaluate_policies(
    test_df: pd.DataFrame,
    models: FittedQuoteModels,
    oracle_context: OracleContext,
    optimizer_config: OptimizerConfig | None = None,
    responders: tuple[ResponderKind, ...] = ALL_RESPONDERS,
    random_state: int = DEFAULT_EVALUATION_SEED,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> PolicyEvaluationResult:
    """Evaluate responders on the same held-out RFQs with common random numbers.

    :param test_df: Held-out chronological RFQs including latent columns.
    :param models: Fitted quote models (never given latent columns).
    :param oracle_context: Oracle used to simulate counterfactual fills.
    :param optimizer_config: Grid, cost, and inventory calibration.
    :param responders: Responders to compare.
    :param random_state: Seed for fill simulation and bootstrap resampling.
    :param bootstrap_samples: Number of date-block bootstrap resamples.
    :return: Decisions, aggregate summary, segment splits, and bootstrap CIs.
    :raises ValueError: If the test set is empty.
    """

    if test_df.empty:
        raise ValueError("test_df must not be empty")
    config = optimizer_config or OptimizerConfig()
    observable = observable_view(test_df)
    shared = _shared_grid_quantities(observable, models, config)
    rng = np.random.default_rng(random_state)
    # One uniform draw per RFQ, shared across responders and scenarios.
    fill_uniforms = rng.random(len(test_df))

    decision_frames: list[pd.DataFrame] = []
    for kind in responders:
        decision_frames.append(
            _decide_and_simulate(
                test_df=test_df,
                shared=shared,
                kind=kind,
                config=config,
                oracle_context=oracle_context,
                fill_uniforms=fill_uniforms,
            )
        )
    decisions = pd.concat(decision_frames, ignore_index=True)
    summary = _summarize(decisions)
    segment_summaries = _segment_summaries(decisions)
    bootstrap = _date_block_bootstrap(
        decisions=decisions,
        n_samples=bootstrap_samples,
        rng=np.random.default_rng(random_state + 1),
    )
    return PolicyEvaluationResult(
        decisions=decisions,
        summary=summary,
        segment_summaries=segment_summaries,
        bootstrap=bootstrap,
    )


def run_sensitivity_analysis(
    test_df: pd.DataFrame,
    models: FittedQuoteModels,
    oracle_context: OracleContext,
    scenarios: dict[str, OptimizerConfig],
    responders: tuple[ResponderKind, ...] = ALL_RESPONDERS,
    random_state: int = DEFAULT_EVALUATION_SEED,
) -> pd.DataFrame:
    """Re-run the policy comparison under alternative cost calibrations.

    The same RFQs, models, and uniform fill draws are reused in every
    scenario, so metric changes are attributable to the calibration.

    :param test_df: Held-out chronological RFQs including latent columns.
    :param models: Fitted quote models.
    :param oracle_context: Oracle used to simulate counterfactual fills.
    :param scenarios: Mapping from scenario name to optimizer configuration.
    :param responders: Responders to compare.
    :param random_state: Seed shared across scenarios.
    :return: One row per scenario and responder with headline metrics.
    :raises ValueError: If no scenarios are supplied.
    """

    if not scenarios:
        raise ValueError("scenarios must not be empty")
    records: list[pd.DataFrame] = []
    for scenario_name, scenario_config in scenarios.items():
        result = evaluate_policies(
            test_df=test_df,
            models=models,
            oracle_context=oracle_context,
            optimizer_config=scenario_config,
            responders=responders,
            random_state=random_state,
            bootstrap_samples=0,
        )
        scenario_summary = result.summary.copy()
        scenario_summary.insert(0, "scenario", scenario_name)
        records.append(scenario_summary)
    return pd.concat(records, ignore_index=True)


@dataclass(frozen=True)
class _SharedGridQuantities:
    """Model predictions and cost terms shared by every responder."""

    grid: np.ndarray
    quotes: np.ndarray
    p_win: np.ndarray
    selection: np.ndarray
    v0: np.ndarray
    cost_cents: np.ndarray
    inventory_value_cents: np.ndarray
    in_support: np.ndarray


def _shared_grid_quantities(
    observable: pd.DataFrame,
    models: FittedQuoteModels,
    config: OptimizerConfig,
) -> _SharedGridQuantities:
    grid = aggressiveness_grid(config)
    n_rows = len(observable)
    n_grid = len(grid)
    side = observable["side_sign"].astype(float).to_numpy()
    cp_plus = observable["cp_plus"].astype(float).to_numpy()
    width = observable["market_width"].astype(float).to_numpy()

    quotes = np.empty((n_rows, n_grid), dtype=float)
    p_win = np.empty((n_rows, n_grid), dtype=float)
    selection = np.empty((n_rows, n_grid), dtype=float)
    for grid_index, aggressiveness in enumerate(grid):
        quote_vector = cp_plus + side * float(aggressiveness) * width
        quote_series = pd.Series(quote_vector, index=observable.index)
        quotes[:, grid_index] = quote_vector
        p_win[:, grid_index] = (
            predict_win_probability(models.fill_model, observable, quote=quote_series)
            .to_numpy()
        )
        selection[:, grid_index] = (
            predict_selection(models.selection_model, observable, quote=quote_series)
            .to_numpy()
        )
    p_win = np.nan_to_num(p_win, nan=0.0)
    selection = np.nan_to_num(selection, nan=0.0)

    v0 = resolve_v0(observable, models).to_numpy()
    cost_cents = np.empty(n_rows, dtype=float)
    inventory_value_cents = np.empty(n_rows, dtype=float)
    for position, (_, row) in enumerate(observable.iterrows()):
        cost_cents[position] = points_to_cents(rfq_trading_cost_points(row, config))
        inventory_value_cents[position] = points_to_cents(
            rfq_inventory_value_points(row, config)
        )
    support_low, support_high = models.fill_model.aggressiveness_support
    in_support = (grid >= support_low) & (grid <= support_high)
    return _SharedGridQuantities(
        grid=grid,
        quotes=quotes,
        p_win=p_win,
        selection=selection,
        v0=v0,
        cost_cents=cost_cents,
        inventory_value_cents=inventory_value_cents,
        in_support=in_support,
    )


def _decide_and_simulate(
    test_df: pd.DataFrame,
    shared: _SharedGridQuantities,
    kind: ResponderKind,
    config: OptimizerConfig,
    oracle_context: OracleContext,
    fill_uniforms: np.ndarray,
) -> pd.DataFrame:
    side = test_df["side_sign"].astype(float).to_numpy()
    cp_plus = test_df["cp_plus"].astype(float).to_numpy()
    y5 = test_df["y5"].astype(float).to_numpy()

    post_win_value = _post_win_value_matrix(kind, shared, side, cp_plus, config)
    edge_cents = 100.0 * side[:, None] * (post_win_value - shared.quotes)
    objective = shared.p_win * (
        edge_cents
        - shared.cost_cents[:, None]
        + shared.inventory_value_cents[:, None]
    )
    masked_objective = np.where(shared.in_support[None, :], objective, -np.inf)
    best_index = np.argmax(masked_objective, axis=1)
    row_positions = np.arange(len(test_df))
    best_objective = masked_objective[row_positions, best_index]
    accepted = best_objective > 0.0

    chosen_quote = shared.quotes[row_positions, best_index]
    chosen_aggressiveness = shared.grid[best_index]
    chosen_p_win = shared.p_win[row_positions, best_index]
    chosen_selection = shared.selection[row_positions, best_index]
    chosen_edge_cents = edge_cents[row_positions, best_index]
    apparent_edge_cents = 100.0 * side * (shared.v0 - chosen_quote)

    quote_series = pd.Series(chosen_quote, index=test_df.index)
    oracle_p = oracle_fill_probability(test_df, quote_series, oracle_context).to_numpy()
    simulated_win = accepted & (fill_uniforms < oracle_p)
    realized_edge_cents = 100.0 * side * (y5 - chosen_quote)
    realized_value_cents = np.where(
        simulated_win,
        realized_edge_cents - shared.cost_cents + shared.inventory_value_cents,
        0.0,
    )
    realized_selection_cents = 100.0 * side * (shared.v0 - y5)

    frame = pd.DataFrame(
        {
            "rfq_id": test_df["rfq_id"].to_numpy(),
            "date": pd.to_datetime(test_df["timestamp"]).dt.normalize().to_numpy(),
            "responder": RESPONDER_LABELS[kind],
            "kind": kind.value,
            "accepted": accepted,
            "quote": np.where(accepted, chosen_quote, np.nan),
            "aggressiveness": np.where(accepted, chosen_aggressiveness, np.nan),
            "p_win": np.where(accepted, chosen_p_win, 0.0),
            "selection_points": np.where(accepted, chosen_selection, np.nan),
            "expected_value_cents": np.where(accepted, best_objective, 0.0),
            "apparent_edge_cents": np.where(accepted, apparent_edge_cents, np.nan),
            "conditional_edge_cents": np.where(accepted, chosen_edge_cents, np.nan),
            "cost_cents": shared.cost_cents,
            "inventory_value_cents": shared.inventory_value_cents,
            "oracle_p_win": np.where(accepted, oracle_p, 0.0),
            "simulated_win": simulated_win,
            "realized_value_cents": realized_value_cents,
            "realized_selection_cents": np.where(
                simulated_win, realized_selection_cents, np.nan
            ),
            "side": test_df["side"].astype(str).to_numpy(),
            "rating_bucket": test_df["rating_bucket"].astype(str).to_numpy(),
            "regime": test_df["regime"].astype(str).to_numpy(),
            "liquidity_bucket": _liquidity_bucket(test_df["liquidity_score"]).to_numpy(),
        }
    )
    return frame


def _post_win_value_matrix(
    kind: ResponderKind,
    shared: _SharedGridQuantities,
    side: np.ndarray,
    cp_plus: np.ndarray,
    config: OptimizerConfig,
) -> np.ndarray:
    n_grid = shared.quotes.shape[1]
    if kind is ResponderKind.PLAIN_CP_PLUS:
        return np.repeat(cp_plus[:, None], n_grid, axis=1)
    if kind is ResponderKind.PLAIN_V0:
        return np.repeat(shared.v0[:, None], n_grid, axis=1)
    if kind is ResponderKind.EDGE_CONSISTENT:
        return (
            shared.v0[:, None]
            - side[:, None] * config.selection_haircut * shared.selection
        )
    raise ValueError(f"unsupported responder kind: {kind}")


def _summarize(decisions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for responder, group in decisions.groupby("responder", sort=False):
        accepted = group.loc[group["accepted"]]
        fills = group.loc[group["simulated_win"]]
        n_rfqs = len(group)
        n_fills = len(fills)
        records.append(
            {
                "responder": str(responder),
                "n_rfqs": float(n_rfqs),
                "decline_rate": float(1.0 - group["accepted"].mean()),
                "mean_aggressiveness": float(accepted["aggressiveness"].mean())
                if not accepted.empty
                else float("nan"),
                "predicted_fill_rate": float(group["p_win"].mean()),
                "simulated_fill_rate": float(group["simulated_win"].mean()),
                "mean_apparent_edge_cents": float(accepted["apparent_edge_cents"].mean())
                if not accepted.empty
                else float("nan"),
                "mean_conditional_edge_cents": float(
                    accepted["conditional_edge_cents"].mean()
                )
                if not accepted.empty
                else float("nan"),
                "mean_realized_selection_cents": float(
                    fills["realized_selection_cents"].mean()
                )
                if n_fills > 0
                else float("nan"),
                "mean_cost_cents": float(group["cost_cents"].mean()),
                "mean_inventory_value_cents": float(group["inventory_value_cents"].mean()),
                "net_value_per_rfq_cents": float(group["realized_value_cents"].mean()),
                "net_value_per_fill_cents": float(
                    group["realized_value_cents"].sum() / n_fills
                )
                if n_fills > 0
                else float("nan"),
            }
        )
    return pd.DataFrame(records)


def _segment_summaries(decisions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summaries: dict[str, pd.DataFrame] = {}
    for segment_column in ("side", "liquidity_bucket", "rating_bucket", "regime"):
        grouped = (
            decisions.groupby([segment_column, "responder"], sort=False, observed=True)
            .agg(
                net_value_per_rfq_cents=("realized_value_cents", "mean"),
                simulated_fill_rate=("simulated_win", "mean"),
                decline_rate=("accepted", lambda accepted: 1.0 - accepted.mean()),
                n_rfqs=("accepted", "size"),
            )
            .reset_index()
        )
        summaries[segment_column] = grouped
    return summaries


def _date_block_bootstrap(
    decisions: pd.DataFrame,
    n_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Bootstrap net value per RFQ by resampling whole dates, not rows.

    Rows within a date share market conditions, so row-level resampling
    would understate uncertainty.
    """

    responders = list(decisions["responder"].unique())
    if n_samples <= 0:
        return pd.DataFrame(
            {
                "responder": responders,
                "net_value_per_rfq_cents": [
                    float(
                        decisions.loc[
                            decisions["responder"] == responder, "realized_value_cents"
                        ].mean()
                    )
                    for responder in responders
                ],
                "ci_low_cents": np.nan,
                "ci_high_cents": np.nan,
            }
        )

    per_date = (
        decisions.groupby(["date", "responder"], sort=True, observed=True)[
            "realized_value_cents"
        ]
        .agg(["sum", "size"])
        .reset_index()
    )
    sums = per_date.pivot(index="date", columns="responder", values="sum")
    counts = per_date.pivot(index="date", columns="responder", values="size")
    sums = sums[responders].fillna(0.0).to_numpy()
    counts = counts[responders].fillna(0.0).to_numpy()
    n_dates = sums.shape[0]

    sample_indices = rng.integers(0, n_dates, size=(n_samples, n_dates))
    sampled_sums = sums[sample_indices].sum(axis=1)
    sampled_counts = counts[sample_indices].sum(axis=1)
    sampled_counts = np.maximum(sampled_counts, 1.0)
    sampled_means = sampled_sums / sampled_counts

    records: list[dict[str, float | str]] = []
    for responder_index, responder in enumerate(responders):
        point_estimate = float(
            decisions.loc[
                decisions["responder"] == responder, "realized_value_cents"
            ].mean()
        )
        records.append(
            {
                "responder": responder,
                "net_value_per_rfq_cents": point_estimate,
                "ci_low_cents": float(
                    np.quantile(sampled_means[:, responder_index], 0.025)
                ),
                "ci_high_cents": float(
                    np.quantile(sampled_means[:, responder_index], 0.975)
                ),
            }
        )
    return pd.DataFrame(records)


def _liquidity_bucket(liquidity_score: pd.Series) -> pd.Series:
    return pd.qcut(
        liquidity_score.astype(float),
        q=3,
        labels=["low", "medium", "high"],
        duplicates="drop",
    ).astype(str)
