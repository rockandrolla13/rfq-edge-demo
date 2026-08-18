"""Compact reporting and reconciliation for the control layer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rfq_edge.bellman import BellmanSolution
from rfq_edge.control_config import ControlMarketConfig
from rfq_edge.control_evaluation import ControlEvaluationResult
from rfq_edge.control_state import RFQEvent

RECONCILIATION_TOLERANCE_CENTS = 1e-6

_COMPARISON_COLUMNS: dict[str, str] = {
    "response_rate": "response rate",
    "fill_rate": "fill rate",
    "decline_rate": "decline rate",
    "average_aggressiveness": "avg z",
    "realized_clean_edge_cents": "clean edge (c)",
    "adverse_selection_cents": "selection (c)",
    "active_volume_units": "active units",
    "passive_internalized_units": "passive units",
    "target_completion_pct": "completion (%)",
    "terminal_penalty_cents": "terminal pen (c)",
    "total_objective_cents": "total reward (c)",
}


def reconcile_episode_rewards(log: pd.DataFrame) -> dict[str, float]:
    """Verify that reward components sum exactly to the total.

    Checks, in cents:

    * cumulative reward equals RFQ rewards minus active costs, running
      penalties, and the terminal penalty;
    * per filled RFQ, reward equals apparent edge minus realized selection
      minus RFQ cost.

    :param log: Episode event log.
    :return: Component totals and the reconstructed total.
    :raises ValueError: If any reconciliation exceeds tolerance.
    """

    rfq_total = float(log["rfq_reward_cents"].sum())
    active_total = float(log["active_execution_cost_cents"].sum())
    running_total = float(log["running_inventory_penalty_cents"].sum())
    terminal_total = float(log["terminal_penalty_cents"].sum())
    reconstructed = rfq_total - active_total - running_total - terminal_total
    reported = float(log["cumulative_reward_cents"].iloc[-1])
    if abs(reconstructed - reported) > RECONCILIATION_TOLERANCE_CENTS:
        raise ValueError(
            f"total reward mismatch: components {reconstructed} vs log {reported}"
        )

    fills = log.loc[log["filled"]]
    decomposed = (
        fills["apparent_edge_cents"]
        - fills["realized_selection_cents"]
        - fills["rfq_cost_cents"]
    )
    mismatch = (decomposed - fills["rfq_reward_cents"]).abs().max() if len(fills) else 0.0
    if float(mismatch) > RECONCILIATION_TOLERANCE_CENTS:
        raise ValueError(f"per-fill decomposition mismatch of {mismatch} cents")

    return {
        "rfq_reward_cents": rfq_total,
        "active_execution_cost_cents": active_total,
        "running_inventory_penalty_cents": running_total,
        "terminal_penalty_cents": terminal_total,
        "total_reward_cents": reconstructed,
    }


def policy_comparison_table(
    result: ControlEvaluationResult,
    episode_name: str,
) -> pd.DataFrame:
    """Compact policy comparison for one episode type.

    :param result: Evaluation result.
    :param episode_name: Episode configuration to display.
    :return: Rounded table, one row per policy.
    """

    subset = result.policy_metrics.loc[
        result.policy_metrics["episode_name"] == episode_name
    ]
    table = subset.set_index("policy")[list(_COMPARISON_COLUMNS)].rename(
        columns=_COMPARISON_COLUMNS
    )
    return table.round(2)


def reward_decomposition_table(
    result: ControlEvaluationResult,
    episode_name: str,
) -> pd.DataFrame:
    """Mean reward decomposition per policy for one episode type.

    :param result: Evaluation result.
    :param episode_name: Episode configuration to display.
    :return: Component table in cents per episode.
    """

    subset = result.episode_summaries.loc[
        result.episode_summaries["episode_name"] == episode_name
    ]
    means = subset.groupby("policy", sort=False).mean(numeric_only=True)
    table = pd.DataFrame(
        {
            "apparent edge (c)": means["gross_apparent_edge_cents"],
            "adverse selection (c)": -means["adverse_selection_cents"],
            "RFQ costs (c)": -means["rfq_costs_cents"],
            "active costs (c)": -means["active_execution_cost_cents"],
            "impact within active (c)": -means["temporary_impact_cents"],
            "running penalty (c)": -means["running_inventory_penalty_cents"],
            "terminal penalty (c)": -means["terminal_penalty_cents"],
            "total simulated control reward (c)": means["total_objective_cents"],
        }
    )
    return table.round(1)


def training_history_diagnostics(history: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Empirical evidence of the fill and selection mechanism in history.

    :param history: Output of ``generate_training_history``.
    :return: Win rate by aggressiveness bucket, and realized selection on
        fills split by dealer side (positive means winning was costly).
    """

    buckets = pd.cut(history["aggressiveness"], bins=np.linspace(-1.5, 1.5, 7))
    win_rate = (
        history.groupby(buckets, observed=True)["won"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "win_rate", "size": "n_rfqs"})
        .reset_index(names="aggressiveness_bucket")
    )
    fills = history.loc[history["won"]].copy()
    fills["side"] = np.where(fills["side_sign"] == 1, "dealer_buy", "dealer_sell")
    selection_by_side = (
        fills.groupby("side")["realized_selection_points"]
        .agg(["mean", "size"])
        .rename(
            columns={"mean": "mean_realized_selection_points", "size": "n_fills"}
        )
        .reset_index()
    )
    selection_by_side["mean_realized_selection_cents"] = (
        selection_by_side["mean_realized_selection_points"] * 100.0
    )
    return {
        "win_rate_by_aggressiveness": win_rate,
        "realized_selection_by_side": selection_by_side,
    }


def rfq_increment_curves(
    market_config: ControlMarketConfig,
    models: object,
    solution: BellmanSolution,
    inventory: int,
    time_index: int,
    event: RFQEvent,
) -> pd.DataFrame:
    """Decision curves of the RFQ jump operator for one event and state.

    :param market_config: Market configuration.
    :param models: Quote model bundle (fitted, hybrid, or oracle).
    :param solution: Bellman solution supplying continuation values.
    :param inventory: Inventory at the decision.
    :param time_index: Time step of the decision.
    :param event: Incoming RFQ.
    :return: One row per candidate z with p_win, standalone reward,
        continuation delta, and the RFQ increment, all in cents.
    """

    z_grid = solution.aggressiveness_grid
    p_win = models.event_fill_probability(event, z_grid)
    post_win_value = models.event_post_win_value(event, z_grid)
    quotes = event.cp_plus + event.side_sign * z_grid * event.market_width
    standalone = float(event.size) * (
        float(event.side_sign) * (post_win_value - quotes) * 100.0
        - market_config.rfq_transaction_cost_cents
    )
    step = min(time_index, solution.episode_config.n_steps - 1)
    continuation = solution.post_rfq_value[step, :, event.regime.value]
    delta = float(
        continuation[solution.inventory_index(inventory + event.side_sign * event.size)]
        - continuation[solution.inventory_index(inventory)]
    )
    increments = p_win * (standalone + delta)
    return pd.DataFrame(
        {
            "aggressiveness": z_grid,
            "quote": quotes,
            "p_win": p_win,
            "standalone_reward_cents": standalone,
            "continuation_delta_cents": delta,
            "rfq_increment_cents": increments,
        }
    )


def rfq_decision_across_inventories(
    market_config: ControlMarketConfig,
    models: object,
    solution: BellmanSolution,
    inventories: tuple[int, ...],
    time_index: int,
    event: RFQEvent,
) -> pd.DataFrame:
    """How the optimal response to one identical RFQ changes with inventory.

    :param market_config: Market configuration.
    :param models: Quote model bundle.
    :param solution: Bellman solution supplying continuation values.
    :param inventories: Inventory levels to compare.
    :param time_index: Time step of the decision.
    :param event: The identical incoming RFQ.
    :return: One row per inventory with the chosen action and diagnostics.
    """

    rows = []
    for inventory in inventories:
        curves = rfq_increment_curves(
            market_config, models, solution, inventory, time_index, event
        )
        best = curves.loc[curves["rfq_increment_cents"].idxmax()]
        filled_inventory = inventory + event.side_sign * event.size
        breaches = abs(filled_inventory) > solution.episode_config.inventory_limit
        responds = (not breaches) and (best["rfq_increment_cents"] > 0.0)
        rows.append(
            {
                "inventory": inventory,
                "decision": "quote" if responds else "decline",
                "aggressiveness": best["aggressiveness"] if responds else float("nan"),
                "quote": best["quote"] if responds else float("nan"),
                "p_win": best["p_win"] if responds else float("nan"),
                "standalone_reward_cents": best["standalone_reward_cents"],
                "continuation_delta_cents": best["continuation_delta_cents"],
                "best_increment_cents": best["rfq_increment_cents"],
            }
        )
    return pd.DataFrame(rows)


def format_paired_differences(result: ControlEvaluationResult) -> pd.DataFrame:
    """Readable paired-difference table with explicit conclusions.

    A policy is reported as better or worse than a baseline only when the
    bootstrap confidence interval excludes zero; otherwise the comparison is
    labelled inconclusive.

    :param result: Evaluation result.
    :return: Formatted table.
    """

    rows = []
    for _, row in result.paired_differences.iterrows():
        if row["ci_lower_cents"] > 0.0:
            conclusion = "better than baseline"
        elif row["ci_upper_cents"] < 0.0:
            conclusion = "worse than baseline"
        else:
            conclusion = "inconclusive (interval includes zero)"
        rows.append(
            {
                "episode": row["episode_name"],
                "policy": row["policy"],
                "baseline": row["baseline"],
                "mean diff (c)": round(float(row["mean_difference_cents"]), 1),
                "95% CI (c)": (
                    f"[{row['ci_lower_cents']:.1f}, {row['ci_upper_cents']:.1f}]"
                ),
                "conclusion": conclusion,
            }
        )
    return pd.DataFrame(rows)
