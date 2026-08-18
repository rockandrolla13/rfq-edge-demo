"""Reusable Matplotlib figures for the RFQ edge-consistent demo notebook.

Every function returns ``(figure, axes)`` and never calls ``plt.show()``.
Price differences are displayed in cents; price levels stay in points.

Color conventions used throughout:

* grey — plain CP+ responder
* blue — plain V0 responder
* red/orange — edge-consistent responder
* black dashed — oracle truth
* green — selected quote
* horizontal zero line — the decline option
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

COLOR_PLAIN_CP_PLUS = "grey"
COLOR_PLAIN_V0 = "tab:blue"
COLOR_EDGE_CONSISTENT = "tab:red"
COLOR_ORACLE = "black"
COLOR_SELECTED = "tab:green"

RESPONDER_COLORS: dict[str, str] = {
    "Plain CP+": COLOR_PLAIN_CP_PLUS,
    "Plain V0": COLOR_PLAIN_V0,
    "Edge-consistent": COLOR_EDGE_CONSISTENT,
    "Oracle": COLOR_ORACLE,
}

POINTS_TO_CENTS = 100.0
TITLE_SIZE = 13
LABEL_SIZE = 11


def plot_simulation_workflow() -> tuple[Figure, Axes]:
    """Draw the observable and hidden paths of the synthetic RFQ market.

    :return: Figure and axes with the workflow diagram.
    """

    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    observable_boxes = [
        (0.03, "Observable RFQ\nstate X"),
        (0.23, "Dealer chooses\nquote q"),
        (0.43, "Client decides\nwhether to trade"),
        (0.63, "Win or loss"),
        (0.83, "Independent t+5\nclean mark Y5"),
    ]
    for x_position, label in observable_boxes:
        ax.annotate(
            label,
            xy=(x_position + 0.07, 0.72),
            ha="center",
            va="center",
            fontsize=LABEL_SIZE,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#dbe9f6", "edgecolor": "#3c6e9f"},
        )
    for x_position in (0.155, 0.355, 0.555, 0.755):
        ax.annotate(
            "",
            xy=(x_position + 0.045, 0.72),
            xytext=(x_position, 0.72),
            arrowprops={"arrowstyle": "->", "color": "#3c6e9f", "lw": 1.6},
        )

    hidden_boxes = [
        (0.10, "Hidden client\ninformation"),
        (0.36, "Affects the client's\ndecision"),
        (0.62, "Correlated\nwith Y5"),
        (0.86, "Winning becomes\ninformative"),
    ]
    for x_position, label in hidden_boxes:
        ax.annotate(
            label,
            xy=(x_position, 0.22),
            ha="center",
            va="center",
            fontsize=LABEL_SIZE,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f9e0d9", "edgecolor": "#b3402a"},
        )
    for x_start, x_end in ((0.17, 0.27), (0.45, 0.53), (0.69, 0.77)):
        ax.annotate(
            "",
            xy=(x_end, 0.22),
            xytext=(x_start, 0.22),
            arrowprops={"arrowstyle": "->", "color": "#b3402a", "lw": 1.6},
        )
    ax.annotate(
        "",
        xy=(0.50, 0.60),
        xytext=(0.38, 0.32),
        arrowprops={"arrowstyle": "->", "color": "#b3402a", "lw": 1.4, "linestyle": "dashed"},
    )
    ax.text(
        0.02,
        0.94,
        "Observable path (available to fitted models)",
        fontsize=LABEL_SIZE,
        color="#3c6e9f",
    )
    ax.text(
        0.02,
        0.02,
        "Hidden path (latent simulation variables, never shown to models)",
        fontsize=LABEL_SIZE,
        color="#b3402a",
    )
    ax.set_title("How the synthetic RFQ market works", fontsize=TITLE_SIZE)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    return fig, ax


def plot_distribution(
    values: pd.Series,
    title: str,
    xlabel: str,
    bins: int = 40,
    color: str = COLOR_PLAIN_V0,
    log_x: bool = False,
) -> tuple[Figure, Axes]:
    """Histogram of one observable RFQ quantity.

    :param values: Series to plot.
    :param title: Plot title.
    :param xlabel: X-axis label including units.
    :param bins: Number of histogram bins.
    :param color: Bar color.
    :param log_x: Whether to use a logarithmic x scale.
    :return: Figure and axes.
    """

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.hist(values.astype(float), bins=bins, color=color, edgecolor="white")
    if log_x:
        ax.set_xscale("log")
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel("Count", fontsize=LABEL_SIZE)
    return fig, ax


def plot_bond_activity_distribution(df: pd.DataFrame) -> tuple[Figure, Axes]:
    """Histogram of RFQ counts per bond, showing sparse bond coverage.

    :param df: RFQ dataframe.
    :return: Figure and axes.
    """

    counts = df.groupby("bond_id").size()
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.hist(counts, bins=40, color=COLOR_PLAIN_V0, edgecolor="white")
    ax.axvline(counts.median(), color=COLOR_ORACLE, linestyle="--", label=f"median = {counts.median():.0f}")
    ax.set_title("RFQs per bond (most bonds are sparse)", fontsize=TITLE_SIZE)
    ax.set_xlabel("RFQ count per bond", fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of bonds", fontsize=LABEL_SIZE)
    ax.legend()
    return fig, ax


def plot_synthetic_price_paths(df: pd.DataFrame, n_bonds: int = 5) -> tuple[Figure, Axes]:
    """CP+ price paths for the most active bonds.

    :param df: RFQ dataframe.
    :param n_bonds: Number of active bonds to display.
    :return: Figure and axes.
    """

    active_bonds = df.groupby("bond_id").size().nlargest(n_bonds).index
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    for bond_id in active_bonds:
        bond_rows = df.loc[df["bond_id"] == bond_id].sort_values("timestamp")
        ax.plot(
            pd.to_datetime(bond_rows["timestamp"]),
            bond_rows["cp_plus"],
            marker=".",
            markersize=3,
            linewidth=1.0,
            label=bond_id,
        )
    ax.set_title("Example CP+ clean price paths (most active bonds)", fontsize=TITLE_SIZE)
    ax.set_xlabel("Date", fontsize=LABEL_SIZE)
    ax.set_ylabel("CP+ clean price (points)", fontsize=LABEL_SIZE)
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    return fig, ax


def plot_internal_signal_vs_future_value(df: pd.DataFrame) -> tuple[Figure, Axes]:
    """Scatter of internal alpha against the realized future residual.

    :param df: RFQ dataframe.
    :return: Figure and axes.
    """

    internal_alpha = (df["internal_mid"] - df["cp_plus"]).astype(float)
    future_residual = (df["y5"] - df["cp_plus"]).astype(float)
    correlation = float(internal_alpha.corr(future_residual))
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(internal_alpha, future_residual, s=6, alpha=0.25, color=COLOR_PLAIN_V0)
    slope = correlation * future_residual.std() / internal_alpha.std()
    x_line = np.linspace(internal_alpha.min(), internal_alpha.max(), 50)
    ax.plot(
        x_line,
        slope * x_line,
        color=COLOR_ORACLE,
        linestyle="--",
        label=f"correlation = {correlation:.2f}",
    )
    ax.set_title("Internal mid signal is informative but imperfect", fontsize=TITLE_SIZE)
    ax.set_xlabel("Internal mid minus CP+ (points)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Realized Y5 minus CP+ (points)", fontsize=LABEL_SIZE)
    ax.legend()
    return fig, ax


def plot_hidden_information_mechanism(df: pd.DataFrame) -> tuple[Figure, np.ndarray]:
    """Show that winning selects different future outcomes per dealer side.

    :param df: RFQ dataframe with realized outcomes.
    :return: Figure and axes array (one panel per side).
    """

    residual = (df["y5"] - df["cp_plus"]).astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for ax, side in zip(axes, ("dealer_buy", "dealer_sell")):
        side_mask = df["side"] == side
        all_residuals = residual.loc[side_mask]
        win_residuals = residual.loc[side_mask & df["won"]]
        ax.hist(all_residuals, bins=40, alpha=0.45, color=COLOR_PLAIN_CP_PLUS, density=True, label="all RFQs")
        ax.hist(win_residuals, bins=40, alpha=0.55, color=COLOR_EDGE_CONSISTENT, density=True, label="wins")
        ax.axvline(all_residuals.mean(), color=COLOR_PLAIN_CP_PLUS, linestyle="--")
        ax.axvline(win_residuals.mean(), color=COLOR_EDGE_CONSISTENT, linestyle="--")
        gap_cents = (win_residuals.mean() - all_residuals.mean()) * POINTS_TO_CENTS
        ax.set_title(f"{side} (win tilt {gap_cents:+.1f}c)", fontsize=TITLE_SIZE)
        ax.set_xlabel("Y5 minus CP+ (points)", fontsize=LABEL_SIZE)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Density", fontsize=LABEL_SIZE)
    fig.suptitle("Winning is informative: fills tilt future value against the dealer", fontsize=TITLE_SIZE)
    fig.tight_layout()
    return fig, axes


def plot_empirical_win_rate_by_aggressiveness(bucket_table: pd.DataFrame) -> tuple[Figure, Axes]:
    """Empirical win rate against historical quote aggressiveness.

    :param bucket_table: Output of ``win_rate_by_aggressiveness_bucket``.
    :return: Figure and axes.
    """

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(
        bucket_table["mean_aggressiveness"],
        bucket_table["win_rate"],
        marker="o",
        color=COLOR_PLAIN_V0,
    )
    ax.set_title("Win rate rises with normalized aggressiveness", fontsize=TITLE_SIZE)
    ax.set_xlabel("Normalized aggressiveness z", fontsize=LABEL_SIZE)
    ax.set_ylabel("Empirical win rate", fontsize=LABEL_SIZE)
    ax.set_ylim(0.0, 1.0)
    return fig, ax


def plot_value_model_comparison(metrics_by_model: dict[str, dict[str, float]]) -> tuple[Figure, Axes]:
    """Bar chart of test MAE and weighted MAE per V0 forecast, in cents.

    :param metrics_by_model: Mapping from model name to metric dictionary.
    :return: Figure and axes.
    """

    names = list(metrics_by_model.keys())
    mae = [metrics_by_model[name]["mae_cents"] for name in names]
    weighted = [metrics_by_model[name]["weighted_mae_cents"] for name in names]
    positions = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.bar(positions - width / 2, mae, width, label="MAE", color=COLOR_PLAIN_V0)
    ax.bar(positions + width / 2, weighted, width, label="size-weighted MAE", color=COLOR_EDGE_CONSISTENT)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, fontsize=LABEL_SIZE)
    ax.set_title("Held-out t+5 forecast error by model", fontsize=TITLE_SIZE)
    ax.set_ylabel("Error (cents)", fontsize=LABEL_SIZE)
    ax.legend()
    return fig, ax


def plot_value_prediction_calibration(
    predicted_residual: pd.Series,
    actual_residual: pd.Series,
    n_buckets: int = 10,
) -> tuple[Figure, Axes]:
    """Binned predicted-versus-realized residual calibration.

    :param predicted_residual: Predicted V0 minus CP+ in points.
    :param actual_residual: Realized Y5 minus CP+ in points.
    :param n_buckets: Number of quantile buckets.
    :return: Figure and axes.
    """

    frame = pd.DataFrame(
        {
            "predicted": predicted_residual.astype(float) * POINTS_TO_CENTS,
            "actual": actual_residual.astype(float) * POINTS_TO_CENTS,
        }
    )
    frame["bucket"] = pd.qcut(frame["predicted"], q=n_buckets, duplicates="drop")
    grouped = frame.groupby("bucket", observed=False).mean()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(grouped["predicted"], grouped["actual"], color=COLOR_PLAIN_V0, zorder=3)
    limits = [
        min(grouped["predicted"].min(), grouped["actual"].min()),
        max(grouped["predicted"].max(), grouped["actual"].max()),
    ]
    ax.plot(limits, limits, color=COLOR_ORACLE, linestyle="--", label="perfect calibration")
    ax.set_title("V0 residual calibration (bucket means)", fontsize=TITLE_SIZE)
    ax.set_xlabel("Predicted residual (cents)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Realized residual (cents)", fontsize=LABEL_SIZE)
    ax.legend()
    return fig, ax


def plot_value_residuals_over_time(
    timestamps: pd.Series,
    errors_points: pd.Series,
) -> tuple[Figure, Axes]:
    """Monthly MAE of the V0 forecast through the test period.

    :param timestamps: RFQ timestamps.
    :param errors_points: Forecast errors (V0 minus Y5) in points.
    :return: Figure and axes.
    """

    frame = pd.DataFrame(
        {
            "month": pd.to_datetime(timestamps).dt.to_period("M").dt.to_timestamp(),
            "abs_error_cents": errors_points.abs().astype(float) * POINTS_TO_CENTS,
        }
    )
    monthly = frame.groupby("month")["abs_error_cents"].mean()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(monthly.index, monthly.to_numpy(), marker="o", color=COLOR_PLAIN_V0)
    ax.set_title("V0 forecast MAE through time", fontsize=TITLE_SIZE)
    ax.set_xlabel("Month", fontsize=LABEL_SIZE)
    ax.set_ylabel("MAE (cents)", fontsize=LABEL_SIZE)
    fig.autofmt_xdate()
    return fig, ax


def plot_value_performance_by_regime(
    mae_by_regime_by_model: dict[str, dict[str, float]],
) -> tuple[Figure, Axes]:
    """Grouped bars of forecast MAE per volatility regime and model.

    :param mae_by_regime_by_model: model name -> regime -> MAE in cents.
    :return: Figure and axes.
    """

    models = list(mae_by_regime_by_model.keys())
    regimes = sorted({regime for values in mae_by_regime_by_model.values() for regime in values})
    positions = np.arange(len(regimes))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    palette = [COLOR_PLAIN_CP_PLUS, COLOR_PLAIN_V0, COLOR_EDGE_CONSISTENT, COLOR_SELECTED]
    for model_index, model_name in enumerate(models):
        values = [mae_by_regime_by_model[model_name].get(regime, np.nan) for regime in regimes]
        ax.bar(
            positions + model_index * width,
            values,
            width,
            label=model_name,
            color=palette[model_index % len(palette)],
        )
    ax.set_xticks(positions + width * (len(models) - 1) / 2)
    ax.set_xticklabels(regimes, fontsize=LABEL_SIZE)
    ax.set_title("Forecast MAE by volatility regime", fontsize=TITLE_SIZE)
    ax.set_xlabel("Regime", fontsize=LABEL_SIZE)
    ax.set_ylabel("MAE (cents)", fontsize=LABEL_SIZE)
    ax.legend()
    return fig, ax


def plot_chronological_split(
    timestamps: pd.Series,
    train_end: pd.Timestamp,
    label_train: str = "train",
    label_test: str = "held-out test",
) -> tuple[Figure, Axes]:
    """Visualize the chronological train/test periods on a timeline.

    :param timestamps: All RFQ timestamps.
    :param train_end: Boundary timestamp between train and test.
    :param label_train: Label for the training period.
    :param label_test: Label for the test period.
    :return: Figure and axes.
    """

    parsed = pd.to_datetime(timestamps)
    monthly_counts = parsed.dt.to_period("W").dt.to_timestamp().value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    boundary = pd.to_datetime(train_end)
    colors = [
        COLOR_PLAIN_V0 if index <= boundary else COLOR_EDGE_CONSISTENT
        for index in monthly_counts.index
    ]
    ax.bar(monthly_counts.index, monthly_counts.to_numpy(), width=6.0, color=colors)
    ax.axvline(boundary, color=COLOR_ORACLE, linestyle="--", linewidth=1.6)
    ax.text(boundary, ax.get_ylim()[1] * 0.9, "  split", fontsize=LABEL_SIZE)
    ax.set_title(f"Chronological split: {label_train} (blue) then {label_test} (red)", fontsize=TITLE_SIZE)
    ax.set_xlabel("Week", fontsize=LABEL_SIZE)
    ax.set_ylabel("RFQs per week", fontsize=LABEL_SIZE)
    fig.autofmt_xdate()
    return fig, ax


def plot_fill_calibration(
    calibration_curve: dict[str, list[float]],
    brier_score: float,
    log_loss_value: float,
) -> tuple[Figure, Axes]:
    """Reliability curve for the fill-probability model.

    :param calibration_curve: Dict with mean_predicted and fraction_positives.
    :param brier_score: Held-out Brier score.
    :param log_loss_value: Held-out log loss.
    :return: Figure and axes.
    """

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.plot(
        calibration_curve["mean_predicted"],
        calibration_curve["fraction_positives"],
        marker="o",
        color=COLOR_PLAIN_V0,
        label="model",
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], color=COLOR_ORACLE, linestyle="--", label="perfect")
    ax.set_title(
        f"Fill-probability calibration (Brier {brier_score:.3f}, log loss {log_loss_value:.3f})",
        fontsize=TITLE_SIZE,
    )
    ax.set_xlabel("Mean predicted win probability", fontsize=LABEL_SIZE)
    ax.set_ylabel("Empirical win rate", fontsize=LABEL_SIZE)
    ax.legend()
    return fig, ax


def plot_fill_probability_by_aggressiveness(
    predicted_curve: pd.DataFrame,
    empirical_buckets: pd.DataFrame,
) -> tuple[Figure, Axes]:
    """Predicted counterfactual fill curve with empirical bucket win rates.

    :param predicted_curve: Columns aggressiveness and p_win (grid means).
    :param empirical_buckets: Output of ``win_rate_by_aggressiveness_bucket``.
    :return: Figure and axes.
    """

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(
        predicted_curve["aggressiveness"],
        predicted_curve["p_win"],
        color=COLOR_PLAIN_V0,
        linewidth=2.0,
        label="model p(q, X) mean",
    )
    ax.scatter(
        empirical_buckets["mean_aggressiveness"],
        empirical_buckets["win_rate"],
        color=COLOR_ORACLE,
        zorder=3,
        label="empirical bucket win rate",
    )
    ax.set_title("Fill probability against aggressiveness", fontsize=TITLE_SIZE)
    ax.set_xlabel("Normalized aggressiveness z", fontsize=LABEL_SIZE)
    ax.set_ylabel("Win probability", fontsize=LABEL_SIZE)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    return fig, ax


def plot_fill_probability_by_side(curves_by_side: dict[str, pd.DataFrame]) -> tuple[Figure, Axes]:
    """Counterfactual fill curves for dealer-buy and dealer-sell RFQs.

    :param curves_by_side: side label -> frame with aggressiveness and p_win.
    :return: Figure and axes.
    """

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    colors = {"dealer_buy": COLOR_PLAIN_V0, "dealer_sell": COLOR_EDGE_CONSISTENT}
    for side, curve in curves_by_side.items():
        ax.plot(
            curve["aggressiveness"],
            curve["p_win"],
            marker="o",
            markersize=4,
            label=side,
            color=colors.get(side, COLOR_PLAIN_CP_PLUS),
        )
    ax.set_title("Fill probability by dealer side", fontsize=TITLE_SIZE)
    ax.set_xlabel("Normalized aggressiveness z", fontsize=LABEL_SIZE)
    ax.set_ylabel("Mean predicted win probability", fontsize=LABEL_SIZE)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    return fig, ax


def plot_fill_probability_by_client_tier(
    curves_by_tier: dict[str, pd.DataFrame],
    title: str = "Fill probability by client tier",
) -> tuple[Figure, Axes]:
    """Counterfactual fill curves per category (client tier by default).

    :param curves_by_tier: category label -> frame with aggressiveness and p_win.
    :param title: Plot title, overridable for other categorical splits.
    :return: Figure and axes.
    """

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for tier, curve in sorted(curves_by_tier.items()):
        ax.plot(curve["aggressiveness"], curve["p_win"], marker="o", markersize=4, label=tier)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.set_xlabel("Normalized aggressiveness z", fontsize=LABEL_SIZE)
    ax.set_ylabel("Mean predicted win probability", fontsize=LABEL_SIZE)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    return fig, ax


def plot_selection_calibration(
    predicted_selection: pd.Series,
    realized_selection: pd.Series,
    n_buckets: int = 8,
) -> tuple[Figure, Axes]:
    """Binned predicted-versus-realized adverse selection on fills.

    :param predicted_selection: Predicted A(q, X) in points.
    :param realized_selection: Realized D in points.
    :param n_buckets: Number of quantile buckets.
    :return: Figure and axes.
    """

    frame = pd.DataFrame(
        {
            "predicted": predicted_selection.astype(float) * POINTS_TO_CENTS,
            "realized": realized_selection.astype(float) * POINTS_TO_CENTS,
        }
    )
    frame["bucket"] = pd.qcut(frame["predicted"], q=n_buckets, duplicates="drop")
    grouped = frame.groupby("bucket", observed=False).mean()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(grouped["predicted"], grouped["realized"], color=COLOR_EDGE_CONSISTENT, zorder=3)
    limits = [
        min(grouped["predicted"].min(), grouped["realized"].min()),
        max(grouped["predicted"].max(), grouped["realized"].max()),
    ]
    ax.plot(limits, limits, color=COLOR_ORACLE, linestyle="--", label="perfect calibration")
    ax.set_title("Adverse-selection calibration on fills (bucket means)", fontsize=TITLE_SIZE)
    ax.set_xlabel("Predicted selection (cents)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Realized selection (cents)", fontsize=LABEL_SIZE)
    ax.legend()
    return fig, ax


def plot_selection_by_aggressiveness(selection_curve: pd.DataFrame) -> tuple[Figure, Axes]:
    """Predicted adverse selection against candidate aggressiveness.

    :param selection_curve: Columns aggressiveness and selection_cents.
    :return: Figure and axes.
    """

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(
        selection_curve["aggressiveness"],
        selection_curve["selection_cents"],
        marker="o",
        color=COLOR_EDGE_CONSISTENT,
    )
    ax.axhline(0.0, color=COLOR_PLAIN_CP_PLUS, linewidth=1.0)
    ax.set_title("Predicted adverse selection against aggressiveness", fontsize=TITLE_SIZE)
    ax.set_xlabel("Normalized aggressiveness z", fontsize=LABEL_SIZE)
    ax.set_ylabel("Predicted selection (cents)", fontsize=LABEL_SIZE)
    return fig, ax


def plot_selection_by_liquidity(selection_by_bucket: pd.DataFrame) -> tuple[Figure, Axes]:
    """Bar chart of predicted selection per liquidity bucket.

    :param selection_by_bucket: Columns bucket and selection_cents.
    :return: Figure and axes.
    """

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(
        selection_by_bucket["bucket"].astype(str),
        selection_by_bucket["selection_cents"],
        color=COLOR_EDGE_CONSISTENT,
    )
    ax.set_title("Predicted adverse selection by liquidity", fontsize=TITLE_SIZE)
    ax.set_xlabel("Liquidity bucket", fontsize=LABEL_SIZE)
    ax.set_ylabel("Predicted selection (cents)", fontsize=LABEL_SIZE)
    return fig, ax


def plot_predicted_vs_oracle_selection(
    predicted_selection: pd.Series,
    oracle_selection: pd.Series,
    n_buckets: int = 8,
) -> tuple[Figure, Axes]:
    """Compare fitted selection with the synthetic-oracle counterfactual.

    :param predicted_selection: Model A(q, X) in points.
    :param oracle_selection: Oracle selection in points.
    :param n_buckets: Number of quantile buckets.
    :return: Figure and axes.
    """

    frame = pd.DataFrame(
        {
            "predicted": predicted_selection.astype(float) * POINTS_TO_CENTS,
            "oracle": oracle_selection.astype(float) * POINTS_TO_CENTS,
        }
    )
    frame["bucket"] = pd.qcut(frame["predicted"], q=n_buckets, duplicates="drop")
    grouped = frame.groupby("bucket", observed=False).mean()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.scatter(grouped["predicted"], grouped["oracle"], color=COLOR_EDGE_CONSISTENT, zorder=3, label="bucket means")
    limits = [
        min(grouped["predicted"].min(), grouped["oracle"].min()),
        max(grouped["predicted"].max(), grouped["oracle"].max()),
    ]
    ax.plot(limits, limits, color=COLOR_ORACLE, linestyle="--", label="oracle truth")
    ax.set_title("Predicted versus oracle adverse selection", fontsize=TITLE_SIZE)
    ax.set_xlabel("Predicted selection (cents)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Oracle selection (cents)", fontsize=LABEL_SIZE)
    ax.legend()
    return fig, ax


def plot_quote_surface(
    grid: pd.DataFrame,
    decisions: pd.DataFrame | None = None,
    oracle_grid: pd.DataFrame | None = None,
    side_label: str = "",
) -> tuple[Figure, np.ndarray]:
    """Four-panel quote surface for one RFQ.

    Panels: fill probability; V0 and m(q, X); apparent versus conditional
    edge; expected objective for all three responders with selected quotes.

    :param grid: Output of ``scan_responder_grid`` for one RFQ.
    :param decisions: Optional output of ``compare_responders`` for markers.
    :param oracle_grid: Optional output of ``oracle_optimal_quote``.
    :param side_label: Text appended to the figure title.
    :return: Figure and 2x2 axes array.
    """

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6))
    quotes = grid["quote"]

    ax = axes[0, 0]
    ax.plot(quotes, grid["p_win"], color=COLOR_PLAIN_V0, linewidth=2.0, label="model p(q, X)")
    if oracle_grid is not None:
        ax.plot(
            oracle_grid["quote"],
            oracle_grid["oracle_p_win"],
            color=COLOR_ORACLE,
            linestyle="--",
            label="oracle truth",
        )
    ax.set_title("Fill probability", fontsize=TITLE_SIZE)
    ax.set_ylabel("p(win)", fontsize=LABEL_SIZE)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.plot(
        quotes,
        grid["post_win_value_plain_v0"],
        color=COLOR_PLAIN_V0,
        linewidth=2.0,
        label="V0 (unconditional)",
    )
    ax.plot(
        quotes,
        grid["post_win_value_edge_consistent"],
        color=COLOR_EDGE_CONSISTENT,
        linewidth=2.0,
        label="m(q, X) conditional",
    )
    if oracle_grid is not None:
        ax.plot(
            oracle_grid["quote"],
            oracle_grid["oracle_post_win_value"],
            color=COLOR_ORACLE,
            linestyle="--",
            label="oracle truth",
        )
    ax.set_title("Post-win clean value", fontsize=TITLE_SIZE)
    ax.set_ylabel("Clean price (points)", fontsize=LABEL_SIZE)
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.plot(
        quotes,
        grid["apparent_edge_cents"],
        color=COLOR_PLAIN_V0,
        linewidth=2.0,
        label="apparent edge (V0 - q)",
    )
    ax.plot(
        quotes,
        grid["edge_cents_edge_consistent"],
        color=COLOR_EDGE_CONSISTENT,
        linewidth=2.0,
        label="conditional edge (m - q)",
    )
    if oracle_grid is not None:
        ax.plot(
            oracle_grid["quote"],
            oracle_grid["oracle_edge_cents"],
            color=COLOR_ORACLE,
            linestyle="--",
            label="oracle truth",
        )
    ax.axhline(0.0, color=COLOR_PLAIN_CP_PLUS, linewidth=1.0)
    ax.set_title("Clean edge", fontsize=TITLE_SIZE)
    ax.set_xlabel("Candidate quote (points)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Edge (cents)", fontsize=LABEL_SIZE)
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    objective_columns = {
        "Plain CP+": ("expected_value_cents_plain_cp_plus", COLOR_PLAIN_CP_PLUS),
        "Plain V0": ("expected_value_cents_plain_v0", COLOR_PLAIN_V0),
        "Edge-consistent": ("expected_value_cents_edge_consistent", COLOR_EDGE_CONSISTENT),
    }
    for label, (column, color) in objective_columns.items():
        ax.plot(quotes, grid[column], color=color, linewidth=2.0, label=label)
    if oracle_grid is not None:
        ax.plot(
            oracle_grid["quote"],
            oracle_grid["oracle_expected_objective_cents"],
            color=COLOR_ORACLE,
            linestyle="--",
            label="oracle truth",
        )
    ax.axhline(0.0, color=COLOR_PLAIN_CP_PLUS, linewidth=1.0, linestyle=":")
    if decisions is not None:
        for _, decision in decisions.iterrows():
            if bool(decision["accepted"]) and decision["quote"] is not None:
                ax.axvline(
                    float(decision["quote"]),
                    color=RESPONDER_COLORS.get(str(decision["responder"]), COLOR_SELECTED),
                    linestyle=":",
                    linewidth=1.6,
                )
                ax.scatter(
                    [float(decision["quote"])],
                    [float(decision["expected_value_cents"])],
                    color=COLOR_SELECTED,
                    zorder=4,
                    s=45,
                )
    ax.set_title("Expected objective J(q, X)", fontsize=TITLE_SIZE)
    ax.set_xlabel("Candidate quote (points)", fontsize=LABEL_SIZE)
    ax.set_ylabel("J (cents)", fontsize=LABEL_SIZE)
    ax.legend(fontsize=9)

    fig.suptitle(f"Quote surface {side_label}".strip(), fontsize=TITLE_SIZE + 1)
    fig.tight_layout()
    return fig, axes


def plot_responder_comparison(comparison: pd.DataFrame) -> tuple[Figure, Axes]:
    """Bar chart of the responder objectives for one RFQ.

    :param comparison: Output of ``compare_responders`` (optionally with an
        oracle row appended).
    :return: Figure and axes.
    """

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    colors = [
        RESPONDER_COLORS.get(str(name), COLOR_SELECTED)
        for name in comparison["responder"]
    ]
    ax.bar(comparison["responder"], comparison["expected_value_cents"], color=colors)
    ax.axhline(0.0, color=COLOR_PLAIN_CP_PLUS, linewidth=1.0)
    ax.set_title("Expected objective at each responder's selected quote", fontsize=TITLE_SIZE)
    ax.set_ylabel("Expected objective (cents)", fontsize=LABEL_SIZE)
    return fig, ax


def plot_selected_quotes_distribution(decisions: pd.DataFrame) -> tuple[Figure, Axes]:
    """Distribution of selected aggressiveness per responder on the test set.

    :param decisions: Long-format decisions from ``evaluate_policies``.
    :return: Figure and axes.
    """

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for responder, group in decisions.groupby("responder", sort=False):
        accepted = group.loc[group["accepted"], "aggressiveness"].dropna()
        ax.hist(
            accepted,
            bins=np.arange(-1.75, 1.80, 0.25),
            alpha=0.5,
            label=str(responder),
            color=RESPONDER_COLORS.get(str(responder), COLOR_SELECTED),
        )
    ax.set_title("Selected quote aggressiveness by responder", fontsize=TITLE_SIZE)
    ax.set_xlabel("Selected normalized aggressiveness z", fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of RFQs", fontsize=LABEL_SIZE)
    ax.legend()
    return fig, ax


def plot_quote_frontier(summary: pd.DataFrame) -> tuple[Figure, Axes]:
    """Fill-rate versus net-clean-value frontier across responders.

    :param summary: Summary table from ``evaluate_policies``.
    :return: Figure and axes.
    """

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for _, row in summary.iterrows():
        color = RESPONDER_COLORS.get(str(row["responder"]), COLOR_SELECTED)
        ax.scatter(
            row["simulated_fill_rate"],
            row["net_value_per_fill_cents"],
            s=90,
            color=color,
            zorder=3,
        )
        ax.annotate(
            str(row["responder"]),
            (row["simulated_fill_rate"], row["net_value_per_fill_cents"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=LABEL_SIZE,
        )
    ax.axhline(0.0, color=COLOR_PLAIN_CP_PLUS, linewidth=1.0, linestyle=":")
    ax.set_title("Fill rate versus net clean value per fill", fontsize=TITLE_SIZE)
    ax.set_xlabel("Simulated fill rate", fontsize=LABEL_SIZE)
    ax.set_ylabel("Net clean value per fill (cents)", fontsize=LABEL_SIZE)
    return fig, ax


def plot_policy_performance(
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame | None = None,
) -> tuple[Figure, Axes]:
    """Net clean value per RFQ per responder with bootstrap intervals.

    :param summary: Summary table from ``evaluate_policies``.
    :param bootstrap: Bootstrap table from ``evaluate_policies``.
    :return: Figure and axes.
    """

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    responders = summary["responder"].tolist()
    values = summary["net_value_per_rfq_cents"].to_numpy()
    colors = [RESPONDER_COLORS.get(str(name), COLOR_SELECTED) for name in responders]
    error_bars = None
    if bootstrap is not None and bootstrap["ci_low_cents"].notna().any():
        aligned = bootstrap.set_index("responder").loc[responders]
        lower = values - aligned["ci_low_cents"].to_numpy()
        upper = aligned["ci_high_cents"].to_numpy() - values
        error_bars = np.vstack([lower, upper])
    ax.bar(responders, values, color=colors, yerr=error_bars, capsize=6)
    ax.axhline(0.0, color=COLOR_PLAIN_CP_PLUS, linewidth=1.0)
    ax.set_title("Held-out net clean value per RFQ (95% date-block bootstrap)", fontsize=TITLE_SIZE)
    ax.set_ylabel("Net clean value per RFQ (cents)", fontsize=LABEL_SIZE)
    return fig, ax


def plot_cumulative_clean_value(decisions: pd.DataFrame) -> tuple[Figure, Axes]:
    """Cumulative simulated clean value through the held-out period.

    :param decisions: Long-format decisions from ``evaluate_policies``.
    :return: Figure and axes.
    """

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for responder, group in decisions.groupby("responder", sort=False):
        daily = group.groupby("date")["realized_value_cents"].sum().sort_index()
        ax.plot(
            daily.index,
            daily.cumsum().to_numpy(),
            label=str(responder),
            color=RESPONDER_COLORS.get(str(responder), COLOR_SELECTED),
            linewidth=2.0,
        )
    ax.axhline(0.0, color=COLOR_PLAIN_CP_PLUS, linewidth=1.0, linestyle=":")
    ax.set_title("Cumulative simulated clean value on the held-out test set", fontsize=TITLE_SIZE)
    ax.set_xlabel("Date", fontsize=LABEL_SIZE)
    ax.set_ylabel("Cumulative clean value (cents)", fontsize=LABEL_SIZE)
    ax.legend()
    fig.autofmt_xdate()
    return fig, ax


def plot_policy_heatmap(
    decisions: pd.DataFrame,
    responder_label: str = "Edge-consistent",
) -> tuple[Figure, Axes]:
    """Heatmap of net clean value per RFQ by liquidity bucket and regime.

    :param decisions: Long-format decisions from ``evaluate_policies``.
    :param responder_label: Responder to display.
    :return: Figure and axes.
    """

    subset = decisions.loc[decisions["responder"] == responder_label]
    pivot = subset.pivot_table(
        index="liquidity_bucket",
        columns="regime",
        values="realized_value_cents",
        aggfunc="mean",
        observed=True,
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    image = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=LABEL_SIZE)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=LABEL_SIZE)
    for row_index in range(pivot.shape[0]):
        for column_index in range(pivot.shape[1]):
            value = pivot.iat[row_index, column_index]
            if np.isfinite(value):
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.1f}c",
                    ha="center",
                    va="center",
                    fontsize=LABEL_SIZE,
                )
    fig.colorbar(image, ax=ax, label="Net clean value per RFQ (cents)")
    ax.set_title(f"{responder_label}: value by liquidity and regime", fontsize=TITLE_SIZE)
    ax.set_xlabel("Regime", fontsize=LABEL_SIZE)
    ax.set_ylabel("Liquidity bucket", fontsize=LABEL_SIZE)
    return fig, ax


def plot_sensitivity_analysis(sensitivity: pd.DataFrame) -> tuple[Figure, np.ndarray]:
    """Selected aggressiveness and decline rate across cost scenarios.

    :param sensitivity: Output of ``run_sensitivity_analysis``.
    :return: Figure and two axes.
    """

    scenarios = sensitivity["scenario"].unique().tolist()
    positions = np.arange(len(scenarios))
    responders = sensitivity["responder"].unique().tolist()
    width = 0.8 / max(len(responders), 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for metric_ax, metric, title, unit in (
        (axes[0], "mean_aggressiveness", "Mean selected aggressiveness", "z"),
        (axes[1], "decline_rate", "Decline rate", "fraction"),
    ):
        for responder_index, responder in enumerate(responders):
            rows = sensitivity.loc[sensitivity["responder"] == responder]
            values = [
                float(rows.loc[rows["scenario"] == scenario, metric].iloc[0])
                for scenario in scenarios
            ]
            metric_ax.bar(
                positions + responder_index * width,
                values,
                width,
                label=str(responder),
                color=RESPONDER_COLORS.get(str(responder), COLOR_SELECTED),
            )
        metric_ax.set_xticks(positions + width * (len(responders) - 1) / 2)
        metric_ax.set_xticklabels(scenarios, rotation=20, ha="right", fontsize=9)
        metric_ax.set_title(title, fontsize=TITLE_SIZE)
        metric_ax.set_ylabel(unit, fontsize=LABEL_SIZE)
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    return fig, axes
