"""Public behavior of control policy evaluation and reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rfq_edge.control_config import (
    liquidation_episode,
    market_making_episode,
)
from rfq_edge.control_evaluation import (
    evaluate_control_policies,
    modified_market,
    run_control_sensitivity,
)
from rfq_edge.control_pipeline import POLICY_ORDER
from rfq_edge.control_reporting import (
    format_paired_differences,
    policy_comparison_table,
    reconcile_episode_rewards,
    reward_decomposition_table,
)

N_EPISODES = 5


@pytest.fixture(scope="module")
def evaluation(control_artifacts):
    return evaluate_control_policies(
        policy_names=POLICY_ORDER,
        episode_configs=[market_making_episode(), liquidation_episode()],
        artifacts=control_artifacts,
        n_episodes=N_EPISODES,
        random_state=11,
        bootstrap_samples=100,
    )


def test_unknown_policy_name_is_rejected(control_artifacts) -> None:
    with pytest.raises(ValueError, match="unknown policies"):
        evaluate_control_policies(
            policy_names=["NotAPolicy"],
            episode_configs=[market_making_episode()],
            artifacts=control_artifacts,
            n_episodes=1,
            random_state=0,
        )


def test_policy_labels_are_valid(evaluation) -> None:
    assert set(evaluation.episode_summaries["policy"]) == set(POLICY_ORDER)
    assert set(evaluation.policy_metrics["policy"]) == set(POLICY_ORDER)


def test_common_random_numbers_share_exogenous_paths(evaluation) -> None:
    summaries = evaluation.episode_summaries
    for (_, _), group in summaries.groupby(["episode_name", "episode_index"]):
        assert group["seed"].nunique() == 1
        assert group["rfqs_observed"].nunique() == 1


def test_reward_components_sum_to_total_objective(evaluation) -> None:
    summaries = evaluation.episode_summaries
    reconstructed = (
        summaries["realized_clean_edge_cents"]
        - summaries["active_execution_cost_cents"]
        - summaries["running_inventory_penalty_cents"]
        - summaries["terminal_penalty_cents"]
    )
    np.testing.assert_allclose(
        reconstructed.to_numpy(), summaries["total_objective_cents"].to_numpy()
    )


def test_fill_decomposition_matches_clean_edge(evaluation) -> None:
    summaries = evaluation.episode_summaries
    reconstructed = (
        summaries["gross_apparent_edge_cents"]
        - summaries["adverse_selection_cents"]
        - summaries["rfq_costs_cents"]
    )
    np.testing.assert_allclose(
        reconstructed.to_numpy(),
        summaries["realized_clean_edge_cents"].to_numpy(),
        atol=1e-9,
    )


def test_metrics_reconcile_with_event_logs(evaluation) -> None:
    for (episode_name, policy), log in evaluation.example_logs.items():
        components = reconcile_episode_rewards(log)
        summary = evaluation.episode_summaries.loc[
            (evaluation.episode_summaries["episode_name"] == episode_name)
            & (evaluation.episode_summaries["policy"] == policy)
            & (evaluation.episode_summaries["episode_index"] == 0)
        ].iloc[0]
        assert summary["total_objective_cents"] == pytest.approx(
            components["total_reward_cents"]
        )


def test_completion_percentage_is_bounded(evaluation) -> None:
    liquidation = evaluation.episode_summaries.loc[
        evaluation.episode_summaries["episode_name"] == "liquidation"
    ]
    assert (liquidation["target_completion_pct"] <= 100.0 + 1e-9).all()


def test_no_inventory_limit_violations(evaluation) -> None:
    assert (evaluation.episode_summaries["inventory_limit_violations"] == 0).all()


def test_confidence_intervals_are_reproducible(control_artifacts, evaluation) -> None:
    repeat = evaluate_control_policies(
        policy_names=POLICY_ORDER,
        episode_configs=[market_making_episode(), liquidation_episode()],
        artifacts=control_artifacts,
        n_episodes=N_EPISODES,
        random_state=11,
        bootstrap_samples=100,
    )
    pd.testing.assert_frame_equal(
        evaluation.paired_differences, repeat.paired_differences
    )


def test_paired_differences_cover_both_baselines(evaluation) -> None:
    paired = evaluation.paired_differences
    assert set(paired["baseline"]) == {"PlainResponder", "EdgeConsistentMyopic"}
    assert (paired["ci_lower_cents"] <= paired["ci_upper_cents"]).all()


def test_no_oracle_fields_in_reporting_outputs(evaluation) -> None:
    for frame in (
        evaluation.episode_summaries,
        evaluation.policy_metrics,
        evaluation.regime_metrics,
        evaluation.paired_differences,
    ):
        assert not any(str(column).startswith("hidden") for column in frame.columns)


def test_reporting_tables_are_compact_and_labelled(evaluation) -> None:
    table = policy_comparison_table(evaluation, "liquidation")
    assert list(table.index) == list(POLICY_ORDER)
    assert "completion (%)" in table.columns

    decomposition = reward_decomposition_table(evaluation, "liquidation")
    assert "total simulated control reward (c)" in decomposition.columns

    formatted = format_paired_differences(evaluation)
    assert set(formatted["conclusion"]).issubset(
        {
            "better than baseline",
            "worse than baseline",
            "inconclusive (interval includes zero)",
        }
    )


def test_modified_market_scales_parameters(control_artifacts) -> None:
    base = control_artifacts.market_config
    market = modified_market(
        base, rfq_cost_cents=10.0, impact_scale=2.0, selection_scale=0.5,
        arrival_scale=1.5,
    )
    assert market.rfq_transaction_cost_cents == 10.0
    for original, modified in zip(base.regime_parameters, market.regime_parameters):
        assert modified.active_impact_cents == pytest.approx(
            2.0 * original.active_impact_cents
        )
        assert modified.information_strength == pytest.approx(
            0.5 * original.information_strength
        )
        assert modified.arrival_probability == pytest.approx(
            min(1.5 * original.arrival_probability, 1.0)
        )


def test_sensitivity_covers_all_scenarios(control_artifacts) -> None:
    sensitivity = run_control_sensitivity(
        base_episode_config=liquidation_episode(),
        base_artifacts=control_artifacts,
        policy_names=["EdgeConsistentMyopic", "DynamicExecution"],
        n_episodes=2,
        random_state=5,
    )
    expected = {
        "inventory_penalty_low", "inventory_penalty_base", "inventory_penalty_high",
        "terminal_penalty_low", "terminal_penalty_high",
        "deadline_short", "deadline_long",
        "rfq_cost_5c", "rfq_cost_7_5c", "rfq_cost_10c",
        "selection_weak", "selection_strong",
        "impact_low", "impact_high",
        "arrival_low", "arrival_high",
    }
    assert set(sensitivity["scenario"]) == expected
    assert {"total_objective_cents", "decline_rate"} <= set(sensitivity.columns)
