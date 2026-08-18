"""Every control plot function runs on small data and returns a Figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from rfq_edge.bellman import solve_bellman
from rfq_edge.control_config import liquidation_episode, market_making_episode
from rfq_edge.control_evaluation import evaluate_control_policies
from rfq_edge.control_plots import (
    plot_action_timeline,
    plot_active_execution_policy_heatmap,
    plot_bellman_residual,
    plot_completion_cost_frontier,
    plot_control_architecture,
    plot_event_timeline,
    plot_internalization_fraction,
    plot_inventory_path,
    plot_inventory_shadow_value,
    plot_mode_map,
    plot_mode_timeline,
    plot_policy_comparison,
    plot_quote_decision_at_event,
    plot_quote_policy_heatmap,
    plot_regime_performance,
    plot_reward_decomposition,
    plot_sensitivity_heatmap,
    plot_target_shortfall,
    plot_value_function,
    policy_color,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def solution(control_artifacts):
    return solve_bellman(
        liquidation_episode(),
        control_artifacts.market_config,
        control_artifacts.fitted_models,
    )


@pytest.fixture(scope="module")
def small_evaluation(control_artifacts):
    return evaluate_control_policies(
        policy_names=["PlainResponder", "DynamicExecution"],
        episode_configs=[market_making_episode(), liquidation_episode()],
        artifacts=control_artifacts,
        n_episodes=2,
        random_state=3,
        bootstrap_samples=50,
    )


@pytest.fixture(scope="module")
def example_log(small_evaluation):
    return small_evaluation.example_logs[("liquidation", "DynamicExecution")]


def test_policy_color_known_and_unknown() -> None:
    assert policy_color("PlainResponder") == "#8a8a8a"
    with pytest.raises(KeyError):
        policy_color("NotAPolicy")


def test_plot_control_architecture() -> None:
    figure, _ = plot_control_architecture()
    assert isinstance(figure, Figure)


def test_episode_plots(example_log) -> None:
    for function in (
        plot_event_timeline,
        plot_inventory_path,
        plot_target_shortfall,
        plot_action_timeline,
        plot_mode_timeline,
    ):
        figure, _ = function(example_log, title="test")
        assert isinstance(figure, Figure)


def test_plot_quote_decision_at_event() -> None:
    z = np.linspace(-1.5, 1.5, 13)
    figure, _ = plot_quote_decision_at_event(
        aggressiveness_grid=z,
        fill_probability=1.0 / (1.0 + np.exp(-z)),
        trade_reward_cents=-z * 10.0,
        increments_cents=np.maximum(-z * 4.0 + 1.0, -5.0),
        chosen_aggressiveness=0.25,
        title="test",
    )
    assert isinstance(figure, Figure)


def test_value_function_and_shadow_plots(solution) -> None:
    figure, _ = plot_value_function(solution, 1, steps=(0, 20, 39), title="test")
    assert isinstance(figure, Figure)
    figure, _ = plot_inventory_shadow_value(solution, 1, steps=(0, 39), title="test")
    assert isinstance(figure, Figure)


def test_policy_surface_plots(solution) -> None:
    figure, _ = plot_quote_policy_heatmap(
        solution, regime_index=1, side_sign=-1, size=1, title="test"
    )
    assert isinstance(figure, Figure)
    figure, _ = plot_active_execution_policy_heatmap(
        solution, regime_index=1, title="test"
    )
    assert isinstance(figure, Figure)
    figure, _ = plot_mode_map(
        solution, regime_index=1, side_sign=-1, size=1, title="test"
    )
    assert isinstance(figure, Figure)


def test_plot_bellman_residual(solution) -> None:
    figure, _ = plot_bellman_residual(solution, title="test")
    assert isinstance(figure, Figure)


def test_comparison_plots(small_evaluation) -> None:
    figure, _ = plot_reward_decomposition(
        small_evaluation.episode_summaries, "liquidation", title="test"
    )
    assert isinstance(figure, Figure)
    figure, _ = plot_policy_comparison(
        small_evaluation.policy_metrics,
        "liquidation",
        metric="total_objective_cents",
        ylabel="cents",
        title="test",
    )
    assert isinstance(figure, Figure)
    figure, _ = plot_completion_cost_frontier(
        small_evaluation.episode_summaries, "liquidation", title="test"
    )
    assert isinstance(figure, Figure)
    figure, _ = plot_internalization_fraction(
        small_evaluation.policy_metrics, "liquidation", title="test"
    )
    assert isinstance(figure, Figure)
    figure, _ = plot_regime_performance(
        small_evaluation.regime_metrics,
        metric="response_rate",
        ylabel="response rate",
        title="test",
    )
    assert isinstance(figure, Figure)


def test_plot_sensitivity_heatmap() -> None:
    sensitivity = pd.DataFrame(
        {
            "scenario": ["a", "a", "b", "b"],
            "episode_name": ["liquidation"] * 4,
            "policy": ["PlainResponder", "DynamicExecution"] * 2,
            "total_objective_cents": [1.0, 2.0, 3.0, 4.0],
        }
    )
    figure, _ = plot_sensitivity_heatmap(
        sensitivity, metric="total_objective_cents", title="test"
    )
    assert isinstance(figure, Figure)
