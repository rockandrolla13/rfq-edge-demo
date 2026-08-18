"""Public behavior of held-out policy evaluation and the synthetic oracle."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rfq_edge.config import OptimizerConfig
from rfq_edge.policy_evaluation import evaluate_policies, run_sensitivity_analysis
from rfq_edge.simulation_diagnostics import (
    build_oracle_context,
    oracle_fill_probability,
    oracle_selection,
    post_win_tilt_by_side,
    realized_selection_summary,
    win_rate_by_aggressiveness_bucket,
)
from tests.conftest import DEMO_SYNTHETIC_CONFIG


def test_oracle_matches_simulated_probability_at_historical_quote(demo_frame) -> None:
    context = build_oracle_context(demo_frame, DEMO_SYNTHETIC_CONFIG)
    probabilities = oracle_fill_probability(demo_frame, demo_frame["quote"], context)
    np.testing.assert_allclose(
        probabilities.to_numpy(),
        demo_frame["latent_p_win"].to_numpy(),
        atol=1e-10,
    )


def test_oracle_probability_increases_with_aggressiveness(demo_frame, demo_oracle_context) -> None:
    sample = demo_frame.iloc[:200]
    aggressive = sample["cp_plus"] + sample["side_sign"] * 1.0 * sample["market_width"]
    passive = sample["cp_plus"] - sample["side_sign"] * 1.0 * sample["market_width"]
    p_aggressive = oracle_fill_probability(sample, aggressive, demo_oracle_context)
    p_passive = oracle_fill_probability(sample, passive, demo_oracle_context)
    assert (p_aggressive > p_passive).all()


def test_oracle_selection_is_positive_on_average(demo_frame, demo_oracle_context) -> None:
    sample = demo_frame.iloc[:200]
    selection = oracle_selection(
        sample, sample["quote"], demo_oracle_context, n_draws=800, random_state=3
    )
    assert float(selection.mean()) > 0.0


def test_oracle_requires_latent_columns(demo_frame, demo_oracle_context) -> None:
    without_latents = demo_frame.drop(
        columns=[column for column in demo_frame.columns if column.startswith("latent_")]
    )
    with pytest.raises(ValueError, match="latent columns"):
        oracle_fill_probability(without_latents, 100.0, demo_oracle_context)


def test_simulation_diagnostics_show_adverse_selection(demo_frame) -> None:
    summary = realized_selection_summary(demo_frame)
    assert summary["selection_gap"] > 0.0
    tilt = post_win_tilt_by_side(demo_frame)
    buy_row = tilt.loc[tilt["side"] == "dealer_buy"].iloc[0]
    sell_row = tilt.loc[tilt["side"] == "dealer_sell"].iloc[0]
    assert buy_row["mean_residual_wins"] < buy_row["mean_residual_all"]
    assert sell_row["mean_residual_wins"] > sell_row["mean_residual_all"]


def test_win_rate_bucket_table_is_monotone(demo_frame) -> None:
    table = win_rate_by_aggressiveness_bucket(demo_frame, n_buckets=5)
    assert table["win_rate"].is_monotonic_increasing


def test_evaluate_policies_uses_same_rfqs_for_every_responder(demo_policy_result) -> None:
    decisions = demo_policy_result.decisions
    rfq_sets = decisions.groupby("responder")["rfq_id"].apply(set)
    first = rfq_sets.iloc[0]
    for rfq_set in rfq_sets:
        assert rfq_set == first


def test_evaluate_policies_summary_has_headline_metrics(demo_policy_result) -> None:
    summary = demo_policy_result.summary
    required = {
        "responder",
        "decline_rate",
        "mean_aggressiveness",
        "predicted_fill_rate",
        "simulated_fill_rate",
        "mean_apparent_edge_cents",
        "mean_conditional_edge_cents",
        "net_value_per_rfq_cents",
        "net_value_per_fill_cents",
    }
    assert required.issubset(summary.columns)
    assert len(summary) == 3
    assert summary["decline_rate"].between(0.0, 1.0).all()


def test_evaluate_policies_is_reproducible_with_common_random_numbers(
    demo_framework, demo_oracle_context
) -> None:
    config = OptimizerConfig(aggressiveness_step=0.5)
    first = evaluate_policies(
        demo_framework.test_df,
        demo_framework.models,
        demo_oracle_context,
        config,
        random_state=11,
        bootstrap_samples=0,
    )
    second = evaluate_policies(
        demo_framework.test_df,
        demo_framework.models,
        demo_oracle_context,
        config,
        random_state=11,
        bootstrap_samples=0,
    )
    pd.testing.assert_frame_equal(first.summary, second.summary)


def test_bootstrap_intervals_bracket_the_point_estimate(demo_policy_result) -> None:
    bootstrap = demo_policy_result.bootstrap
    assert (bootstrap["ci_low_cents"] <= bootstrap["ci_high_cents"]).all()


def test_segment_summaries_cover_required_dimensions(demo_policy_result) -> None:
    segments = demo_policy_result.segment_summaries
    assert set(segments) == {"side", "liquidity_bucket", "rating_bucket", "regime"}
    for table in segments.values():
        assert {"responder", "net_value_per_rfq_cents", "simulated_fill_rate"}.issubset(
            table.columns
        )


def test_sensitivity_analysis_raises_decline_rate_with_costs(
    demo_framework, demo_oracle_context
) -> None:
    scenarios = {
        "low cost": OptimizerConfig(aggressiveness_step=0.5, transaction_bps=0.5),
        "punitive cost": OptimizerConfig(aggressiveness_step=0.5, transaction_bps=200.0),
    }
    sensitivity = run_sensitivity_analysis(
        demo_framework.test_df,
        demo_framework.models,
        demo_oracle_context,
        scenarios,
        random_state=11,
    )
    assert len(sensitivity) == 6
    edge_rows = sensitivity.loc[sensitivity["responder"] == "Edge-consistent"]
    low = float(edge_rows.loc[edge_rows["scenario"] == "low cost", "decline_rate"].iloc[0])
    high = float(
        edge_rows.loc[edge_rows["scenario"] == "punitive cost", "decline_rate"].iloc[0]
    )
    assert high >= low


def test_sensitivity_analysis_rejects_empty_scenarios(
    demo_framework, demo_oracle_context
) -> None:
    with pytest.raises(ValueError, match="scenarios"):
        run_sensitivity_analysis(
            demo_framework.test_df,
            demo_framework.models,
            demo_oracle_context,
            scenarios={},
        )
