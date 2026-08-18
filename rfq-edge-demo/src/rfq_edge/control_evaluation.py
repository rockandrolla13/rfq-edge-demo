"""Policy evaluation for the control layer with common random numbers.

Every policy runs on identical exogenous market and RFQ paths per episode
seed, so paired comparisons difference out path noise. Confidence intervals
use a block bootstrap over whole episodes (the natural dependence block).

The total objective is labelled simulated control reward; it is never real
trading PnL.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from rfq_edge.control_config import (
    ControlMarketConfig,
    EpisodeConfig,
    RegimeParameters,
    with_overrides,
)
from rfq_edge.control_pipeline import (
    ControlArtifacts,
    POLICY_ORDER,
    build_control_artifacts,
    make_policies,
    solve_episode_policies,
)
from rfq_edge.control_state import target_shortfall
from rfq_edge.event_simulator import simulate_episode
from rfq_edge.market_dynamics import simulate_exogenous_path

PREFERRED_INVENTORY_BAND = 2
BASELINE_POLICIES: tuple[str, str] = ("PlainResponder", "EdgeConsistentMyopic")
DEFAULT_BOOTSTRAP_SAMPLES = 500
CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True)
class ControlEvaluationResult:
    """Outputs of one multi-episode policy comparison.

    :param episode_summaries: One row per (episode name, policy, episode
        index) with all per-episode metrics.
    :param policy_metrics: Episode-mean metrics per (episode name, policy).
    :param regime_metrics: Step-level RFQ metrics per (policy, regime).
    :param paired_differences: Mean paired difference of the total objective
        against each baseline with block-bootstrap confidence intervals.
    :param example_logs: Full event log of episode 0 per (episode, policy).
    """

    episode_summaries: pd.DataFrame
    policy_metrics: pd.DataFrame
    regime_metrics: pd.DataFrame
    paired_differences: pd.DataFrame
    example_logs: dict[tuple[str, str], pd.DataFrame]


def evaluate_control_policies(
    policy_names: Sequence[str],
    episode_configs: Sequence[EpisodeConfig],
    artifacts: ControlArtifacts,
    n_episodes: int,
    random_state: int,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> ControlEvaluationResult:
    """Run every policy on identical episode paths and compare them.

    :param policy_names: Policies to compare (subset of POLICY_ORDER).
    :param episode_configs: Episode configurations to evaluate separately.
    :param artifacts: Control artifacts (models, market).
    :param n_episodes: Episodes per configuration.
    :param random_state: Seed controlling every exogenous path.
    :param bootstrap_samples: Episode-block bootstrap resamples.
    :return: Evaluation result.
    :raises ValueError: For unknown policy names or non-positive sizes.
    """

    unknown = set(policy_names) - set(POLICY_ORDER)
    if unknown:
        raise ValueError(f"unknown policies: {sorted(unknown)}")
    if n_episodes < 1:
        raise ValueError("n_episodes must be at least one")

    seed_rng = np.random.default_rng(random_state)
    episode_seeds = seed_rng.integers(0, 2**31 - 1, size=(len(episode_configs), n_episodes))

    summary_rows: list[dict[str, object]] = []
    step_frames: list[pd.DataFrame] = []
    example_logs: dict[tuple[str, str], pd.DataFrame] = {}
    for config_position, episode_config in enumerate(episode_configs):
        solutions = solve_episode_policies(artifacts, episode_config)
        policies = make_policies(artifacts, episode_config, solutions)
        for episode_index in range(n_episodes):
            seed = int(episode_seeds[config_position, episode_index])
            path = simulate_exogenous_path(
                artifacts.market_config, episode_config, random_state=seed
            )
            for name in policy_names:
                result = simulate_episode(
                    policies[name], episode_config, artifacts.market_config, path
                )
                summary = summarize_episode_log(result.log, episode_config)
                summary["episode_name"] = episode_config.name
                summary["policy"] = name
                summary["episode_index"] = episode_index
                summary["seed"] = seed
                summary_rows.append(summary)
                log = result.log
                step_frames.append(
                    log.assign(
                        episode_name=episode_config.name,
                        episode_index=episode_index,
                    )
                )
                if episode_index == 0:
                    example_logs[(episode_config.name, name)] = log

    episode_summaries = pd.DataFrame(summary_rows)
    policy_metrics = (
        episode_summaries.drop(columns=["episode_index", "seed"])
        .groupby(["episode_name", "policy"], sort=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    regime_metrics = _regime_metrics(pd.concat(step_frames, ignore_index=True))
    paired = _paired_differences(
        episode_summaries,
        policy_names=list(policy_names),
        bootstrap_samples=bootstrap_samples,
        random_state=random_state,
    )
    return ControlEvaluationResult(
        episode_summaries=episode_summaries,
        policy_metrics=policy_metrics,
        regime_metrics=regime_metrics,
        paired_differences=paired,
        example_logs=example_logs,
    )


def summarize_episode_log(
    log: pd.DataFrame,
    episode_config: EpisodeConfig,
) -> dict[str, float]:
    """Compute all per-episode metrics directly from the event log.

    Every quantity is reconstructed from log rows so the metrics reconcile
    exactly with the simulation.

    :param log: Event log of one episode.
    :param episode_config: Episode configuration.
    :return: Metric name to value.
    """

    target = episode_config.target_inventory
    quoted = log["action"].str.startswith("quote_rfq")
    declined = log["action"].str.startswith("decline_rfq")
    rfqs = int(log["rfq_arrived"].sum())
    responses = int(quoted.sum())
    fills = int(log["filled"].sum())
    filled_rows = log.loc[log["filled"]]
    fill_sizes = filled_rows["rfq_size"].astype(float)

    initial_shortfall = target_shortfall(episode_config.initial_inventory, target)
    terminal_shortfall = int(log["remaining_target_shortfall"].iloc[-1])
    if initial_shortfall > 0:
        completion = (initial_shortfall - terminal_shortfall) / initial_shortfall
    else:
        completion = 1.0 if terminal_shortfall == 0 else 0.0

    shortfall_before = (log["inventory_before"] - target).abs()
    shortfall_after_rfq = (log["inventory_after_rfq"] - target).abs()
    shortfall_after = (log["inventory_after"] - target).abs()
    passive_units = float((shortfall_before - shortfall_after_rfq).clip(lower=0).sum())
    active_toward_units = float(
        (shortfall_after_rfq - shortfall_after).clip(lower=0).sum()
    )
    moved_toward = passive_units + active_toward_units
    proportion_rfq = passive_units / moved_toward if moved_toward > 0 else float("nan")

    at_target = log.loc[log["inventory_after"] == target, "time_index"]
    time_to_target = float(at_target.iloc[0]) if len(at_target) > 0 else float("nan")

    inventory = log["inventory_after"].astype(float)
    running_penalty_total = float(log["running_inventory_penalty_cents"].sum())
    terminal_penalty_total = float(log["terminal_penalty_cents"].sum())
    apparent_edge_total = float(log["apparent_edge_cents"].sum())
    selection_total = float(log["realized_selection_cents"].sum())
    rfq_cost_total = float(log["rfq_cost_cents"].sum())
    rfq_reward_total = float(log["rfq_reward_cents"].sum())
    active_cost_total = float(log["active_execution_cost_cents"].sum())
    impact_total = float(log["active_impact_cents"].sum())
    total_objective = (
        rfq_reward_total
        - active_cost_total
        - running_penalty_total
        - terminal_penalty_total
    )

    predicted_edge_rows = log.loc[quoted & log["predicted_p_win"].notna()]
    return {
        # RFQ metrics.
        "rfqs_observed": float(rfqs),
        "response_rate": responses / rfqs if rfqs > 0 else float("nan"),
        "decline_rate": float(declined.sum()) / rfqs if rfqs > 0 else float("nan"),
        "fill_rate": fills / responses if responses > 0 else float("nan"),
        "n_fills": float(fills),
        "average_aggressiveness": (
            float(log.loc[quoted, "aggressiveness"].mean()) if responses else float("nan")
        ),
        "gross_apparent_edge_cents": apparent_edge_total,
        "predicted_conditional_edge_cents": (
            float(predicted_edge_rows["rfq_reward_expected_cents"].sum())
            if len(predicted_edge_rows)
            else 0.0
        ),
        "realized_clean_edge_cents": rfq_reward_total,
        "adverse_selection_cents": selection_total,
        "rfq_costs_cents": rfq_cost_total,
        "clean_reward_per_rfq_cents": rfq_reward_total / rfqs if rfqs else float("nan"),
        "clean_reward_per_fill_cents": (
            rfq_reward_total / fills if fills else float("nan")
        ),
        "filled_units": float(fill_sizes.sum()),
        # Execution metrics.
        "initial_target_shortfall": float(initial_shortfall),
        "terminal_target_shortfall": float(terminal_shortfall),
        "target_completion_pct": 100.0 * completion,
        "active_volume_units": float(log["active_execution_amount"].abs().sum()),
        "passive_internalized_units": passive_units,
        "proportion_via_rfqs": proportion_rfq,
        "active_execution_cost_cents": active_cost_total,
        "temporary_impact_cents": impact_total,
        "time_to_target": time_to_target,
        "deadline_failed": 1.0 if terminal_shortfall > 0 else 0.0,
        # Inventory metrics.
        "mean_abs_inventory": float(inventory.abs().mean()),
        "max_abs_inventory": float(inventory.abs().max()),
        "inventory_variance": float(inventory.var(ddof=0)),
        "steps_outside_band": float(
            (shortfall_after > PREFERRED_INVENTORY_BAND).sum()
        ),
        "inventory_limit_violations": float(
            (inventory.abs() > episode_config.inventory_limit).sum()
        ),
        "running_inventory_penalty_cents": running_penalty_total,
        "terminal_penalty_cents": terminal_penalty_total,
        # Objective decomposition (simulated control reward, not real PnL).
        "total_objective_cents": total_objective,
    }


def _regime_metrics(steps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (policy, regime), group in steps.groupby(["policy", "regime"], sort=False):
        rfqs = int(group["rfq_arrived"].sum())
        quoted = group["action"].str.startswith("quote_rfq")
        responses = int(quoted.sum())
        fills = int(group["filled"].sum())
        rows.append(
            {
                "policy": policy,
                "regime": regime,
                "rfqs_observed": rfqs,
                "response_rate": responses / rfqs if rfqs else float("nan"),
                "fill_rate": fills / responses if responses else float("nan"),
                "average_aggressiveness": (
                    float(group.loc[quoted, "aggressiveness"].mean())
                    if responses
                    else float("nan")
                ),
                "rfq_reward_cents_per_rfq": (
                    float(group["rfq_reward_cents"].sum()) / rfqs if rfqs else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _paired_differences(
    episode_summaries: pd.DataFrame,
    policy_names: list[str],
    bootstrap_samples: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state + 1)
    alpha = 1.0 - CONFIDENCE_LEVEL
    rows = []
    pivot = episode_summaries.pivot_table(
        index=["episode_name", "episode_index"],
        columns="policy",
        values="total_objective_cents",
    )
    for episode_name, episode_pivot in pivot.groupby(level="episode_name"):
        for baseline in BASELINE_POLICIES:
            if baseline not in episode_pivot.columns:
                continue
            for policy in policy_names:
                if policy == baseline or policy not in episode_pivot.columns:
                    continue
                differences = (
                    episode_pivot[policy] - episode_pivot[baseline]
                ).to_numpy(dtype=float)
                n = len(differences)
                means = np.empty(bootstrap_samples)
                for sample_index in range(bootstrap_samples):
                    draw = rng.integers(0, n, size=n)
                    means[sample_index] = differences[draw].mean()
                lower = float(np.quantile(means, alpha / 2.0))
                upper = float(np.quantile(means, 1.0 - alpha / 2.0))
                rows.append(
                    {
                        "episode_name": episode_name,
                        "policy": policy,
                        "baseline": baseline,
                        "mean_difference_cents": float(differences.mean()),
                        "ci_lower_cents": lower,
                        "ci_upper_cents": upper,
                        "significant": bool(lower > 0.0 or upper < 0.0),
                        "n_episodes": n,
                    }
                )
    return pd.DataFrame(rows)


def modified_market(
    market_config: ControlMarketConfig,
    rfq_cost_cents: float | None = None,
    impact_scale: float | None = None,
    selection_scale: float | None = None,
    arrival_scale: float | None = None,
) -> ControlMarketConfig:
    """Return a market with scaled cost, impact, selection, or arrivals.

    :param market_config: Base market.
    :param rfq_cost_cents: Replacement RFQ transaction cost.
    :param impact_scale: Multiplier on active impact and half-spread.
    :param selection_scale: Multiplier on client information strength
        (clipped below one).
    :param arrival_scale: Multiplier on arrival probability (clipped to one).
    :return: Modified market configuration.
    """

    new_params = []
    for params in market_config.regime_parameters:
        updates: dict[str, float] = {}
        if impact_scale is not None:
            updates["active_impact_cents"] = params.active_impact_cents * impact_scale
            updates["active_half_spread_cents"] = (
                params.active_half_spread_cents * impact_scale
            )
        if selection_scale is not None:
            updates["information_strength"] = min(
                params.information_strength * selection_scale, 0.95
            )
        if arrival_scale is not None:
            updates["arrival_probability"] = min(
                params.arrival_probability * arrival_scale, 1.0
            )
        new_params.append(
            dataclasses.replace(params, **updates) if updates else params
        )
    replacements: dict[str, object] = {
        "regime_parameters": tuple(new_params),
    }
    if rfq_cost_cents is not None:
        replacements["rfq_transaction_cost_cents"] = rfq_cost_cents
    return dataclasses.replace(market_config, **replacements)


def run_control_sensitivity(
    base_episode_config: EpisodeConfig,
    base_artifacts: ControlArtifacts,
    policy_names: Sequence[str],
    n_episodes: int,
    random_state: int,
) -> pd.DataFrame:
    """Evaluate policies under systematic parameter perturbations.

    Scenarios cover inventory penalty, terminal penalty, RFQ cost, adverse
    selection strength, active market impact, deadline length, and RFQ
    arrival intensity, each at the levels required by the study design.

    :param base_episode_config: Episode to perturb.
    :param base_artifacts: Base artifacts (models are refitted only when the
        market data-generating process changes).
    :param policy_names: Policies to compare.
    :param n_episodes: Episodes per scenario (kept moderate for cost).
    :param random_state: Seed shared across scenarios for paired paths.
    :return: One row per (scenario, episode name, policy) with key metrics.
    """

    base = base_episode_config
    episode_scenarios: dict[str, EpisodeConfig] = {
        "inventory_penalty_low": with_overrides(
            base, running_penalty_cents=base.running_penalty_cents * 0.25
        ),
        "inventory_penalty_base": base,
        "inventory_penalty_high": with_overrides(
            base, running_penalty_cents=base.running_penalty_cents * 4.0
        ),
        "terminal_penalty_low": with_overrides(
            base, terminal_penalty_cents=base.terminal_penalty_cents * 0.25
        ),
        "terminal_penalty_high": with_overrides(
            base, terminal_penalty_cents=base.terminal_penalty_cents * 4.0
        ),
        "deadline_short": with_overrides(base, n_steps=max(base.n_steps // 2, 5)),
        "deadline_long": with_overrides(base, n_steps=base.n_steps * 2),
    }
    market_scenarios: dict[str, ControlMarketConfig] = {
        "rfq_cost_5c": modified_market(base_artifacts.market_config, rfq_cost_cents=5.0),
        "rfq_cost_7_5c": modified_market(
            base_artifacts.market_config, rfq_cost_cents=7.5
        ),
        "rfq_cost_10c": modified_market(
            base_artifacts.market_config, rfq_cost_cents=10.0
        ),
        "selection_weak": modified_market(
            base_artifacts.market_config, selection_scale=0.5
        ),
        "selection_strong": modified_market(
            base_artifacts.market_config, selection_scale=1.5
        ),
        "impact_low": modified_market(base_artifacts.market_config, impact_scale=0.5),
        "impact_high": modified_market(base_artifacts.market_config, impact_scale=2.0),
        "arrival_low": modified_market(base_artifacts.market_config, arrival_scale=0.5),
        "arrival_high": modified_market(
            base_artifacts.market_config, arrival_scale=1.5
        ),
    }

    keep_columns = [
        "average_aggressiveness",
        "decline_rate",
        "active_volume_units",
        "passive_internalized_units",
        "proportion_via_rfqs",
        "target_completion_pct",
        "total_objective_cents",
    ]
    frames = []
    for scenario_name, episode_config in episode_scenarios.items():
        result = evaluate_control_policies(
            policy_names=policy_names,
            episode_configs=[episode_config],
            artifacts=base_artifacts,
            n_episodes=n_episodes,
            random_state=random_state,
            bootstrap_samples=100,
        )
        frame = result.policy_metrics[["episode_name", "policy", *keep_columns]].copy()
        frame.insert(0, "scenario", scenario_name)
        frames.append(frame)
    for scenario_name, market in market_scenarios.items():
        # The data-generating process changed: refit models on the new market.
        scenario_artifacts = build_control_artifacts(
            market_config=market,
            n_training_events=len(base_artifacts.training_history),
            random_state=random_state,
        )
        result = evaluate_control_policies(
            policy_names=policy_names,
            episode_configs=[base],
            artifacts=scenario_artifacts,
            n_episodes=n_episodes,
            random_state=random_state,
            bootstrap_samples=100,
        )
        frame = result.policy_metrics[["episode_name", "policy", *keep_columns]].copy()
        frame.insert(0, "scenario", scenario_name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
