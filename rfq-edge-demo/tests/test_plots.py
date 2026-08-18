"""Every public plot function returns a Matplotlib figure without showing it."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from rfq_edge import plots
from rfq_edge.config import OptimizerConfig
from rfq_edge.responders import compare_responders, scan_responder_grid
from rfq_edge.simulation_diagnostics import win_rate_by_aggressiveness_bucket


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _assert_figure(result) -> None:
    figure = result[0]
    assert isinstance(figure, Figure)


def test_plot_simulation_workflow() -> None:
    _assert_figure(plots.plot_simulation_workflow())


def test_simulation_data_plots(demo_frame) -> None:
    _assert_figure(plots.plot_bond_activity_distribution(demo_frame))
    _assert_figure(plots.plot_synthetic_price_paths(demo_frame, n_bonds=3))
    _assert_figure(plots.plot_internal_signal_vs_future_value(demo_frame))
    _assert_figure(plots.plot_hidden_information_mechanism(demo_frame))
    buckets = win_rate_by_aggressiveness_bucket(demo_frame, n_buckets=5)
    _assert_figure(plots.plot_empirical_win_rate_by_aggressiveness(buckets))


def test_value_model_plots(demo_frame) -> None:
    metrics = {
        "CP+": {"mae_cents": 20.0, "weighted_mae_cents": 21.0},
        "V0": {"mae_cents": 15.0, "weighted_mae_cents": 15.5},
    }
    _assert_figure(plots.plot_value_model_comparison(metrics))
    predicted = (demo_frame["internal_mid"] - demo_frame["cp_plus"]).astype(float)
    actual = (demo_frame["y5"] - demo_frame["cp_plus"]).astype(float)
    _assert_figure(plots.plot_value_prediction_calibration(predicted, actual))
    _assert_figure(
        plots.plot_value_residuals_over_time(demo_frame["timestamp"], predicted - actual)
    )
    by_regime = {
        "CP+": {"calm": 18.0, "normal": 20.0, "volatile": 26.0},
        "V0": {"calm": 13.0, "normal": 15.0, "volatile": 20.0},
    }
    _assert_figure(plots.plot_value_performance_by_regime(by_regime))
    boundary = pd.to_datetime(demo_frame["timestamp"]).quantile(0.8)
    _assert_figure(plots.plot_chronological_split(demo_frame["timestamp"], boundary))


def test_fill_model_plots(demo_frame) -> None:
    curve = {
        "mean_predicted": [0.1, 0.3, 0.5, 0.7],
        "fraction_positives": [0.12, 0.28, 0.52, 0.66],
    }
    _assert_figure(plots.plot_fill_calibration(curve, brier_score=0.15, log_loss_value=0.45))
    predicted_curve = pd.DataFrame(
        {"aggressiveness": [-1.0, 0.0, 1.0], "p_win": [0.1, 0.3, 0.6]}
    )
    buckets = win_rate_by_aggressiveness_bucket(demo_frame, n_buckets=5)
    _assert_figure(
        plots.plot_fill_probability_by_aggressiveness(predicted_curve, buckets)
    )
    curves_by_side = {
        "dealer_buy": predicted_curve,
        "dealer_sell": predicted_curve.assign(p_win=[0.15, 0.35, 0.65]),
    }
    _assert_figure(plots.plot_fill_probability_by_side(curves_by_side))
    curves_by_tier = {"retail": predicted_curve, "professional": predicted_curve}
    _assert_figure(plots.plot_fill_probability_by_client_tier(curves_by_tier))


def test_selection_plots(demo_frame) -> None:
    predicted = (demo_frame["internal_mid"] - demo_frame["y5"]) * demo_frame["side_sign"]
    realized = predicted * 0.8
    _assert_figure(plots.plot_selection_calibration(predicted, realized))
    curve = pd.DataFrame(
        {"aggressiveness": [-1.0, 0.0, 1.0], "selection_cents": [9.0, 6.0, 4.0]}
    )
    _assert_figure(plots.plot_selection_by_aggressiveness(curve))
    by_bucket = pd.DataFrame(
        {"bucket": ["low", "medium", "high"], "selection_cents": [8.0, 5.0, 3.0]}
    )
    _assert_figure(plots.plot_selection_by_liquidity(by_bucket))
    _assert_figure(plots.plot_predicted_vs_oracle_selection(predicted, realized))


def test_quote_decision_plots(demo_framework, demo_policy_result) -> None:
    rfq = demo_framework.test_df.loc[demo_framework.test_df["v0_oof"].notna()].iloc[[0]]
    config = OptimizerConfig(aggressiveness_step=0.5)
    grid = scan_responder_grid(rfq, demo_framework.models, config)
    comparison = compare_responders(rfq, demo_framework.models, config)
    _assert_figure(plots.plot_quote_surface(grid, comparison, side_label="(test RFQ)"))
    _assert_figure(plots.plot_responder_comparison(comparison))
    _assert_figure(plots.plot_selected_quotes_distribution(demo_policy_result.decisions))
    _assert_figure(plots.plot_quote_frontier(demo_policy_result.summary))
    _assert_figure(
        plots.plot_policy_performance(
            demo_policy_result.summary, demo_policy_result.bootstrap
        )
    )
    _assert_figure(plots.plot_cumulative_clean_value(demo_policy_result.decisions))
    _assert_figure(plots.plot_policy_heatmap(demo_policy_result.decisions))


def test_sensitivity_plot() -> None:
    sensitivity = pd.DataFrame(
        {
            "scenario": ["low", "low", "high", "high"],
            "responder": ["Plain V0", "Edge-consistent", "Plain V0", "Edge-consistent"],
            "mean_aggressiveness": [-0.4, -0.6, -0.8, -1.0],
            "decline_rate": [0.05, 0.06, 0.30, 0.35],
        }
    )
    _assert_figure(plots.plot_sensitivity_analysis(sensitivity))
