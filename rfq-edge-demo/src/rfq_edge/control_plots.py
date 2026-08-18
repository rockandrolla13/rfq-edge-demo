"""Plotting functions for the control layer.

Every function returns (Figure, Axes) and never calls plt.show().

Consistent policy colors:

* grey — PlainResponder;
* blue — EdgeConsistentMyopic;
* purple — DynamicMarketMaker;
* orange — DynamicExecution;
* black dashed — OracleDynamic.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow, FancyBboxPatch

from rfq_edge.bellman import (
    BellmanSolution,
    SIDE_SIGNS,
    bellman_residual,
    inventory_shadow_value,
)
from rfq_edge.control_state import EconomicMode

POLICY_COLORS: dict[str, str] = {
    "PlainResponder": "#8a8a8a",
    "EdgeConsistentMyopic": "#1f77b4",
    "DynamicMarketMaker": "#7b3fb3",
    "DynamicExecution": "#ff8c00",
    "OracleDynamic": "#000000",
}
ORACLE_LINESTYLE = "--"
MODE_COLORS: dict[str, str] = {
    EconomicMode.MARKET_MAKING.value: "#1f77b4",
    EconomicMode.PASSIVE_EXECUTION.value: "#2ca02c",
    EconomicMode.DEFENSIVE_MARKET_MAKING.value: "#d62728",
    EconomicMode.ACTIVE_EXECUTION.value: "#ff8c00",
    EconomicMode.WAIT.value: "#c7c7c7",
    EconomicMode.DECLINE.value: "#7f7f7f",
}
MODE_ORDER: tuple[str, ...] = tuple(MODE_COLORS)
_TITLE_SIZE = 12
_LABEL_SIZE = 10


def policy_color(policy: str) -> str:
    """Consistent display color of one policy.

    :param policy: Policy name.
    :return: Hex color.
    :raises KeyError: For unknown policies.
    """

    return POLICY_COLORS[policy]


def _new_axes(figsize: tuple[float, float] = (8.0, 4.5)) -> tuple[Figure, plt.Axes]:
    figure, axes = plt.subplots(figsize=figsize)
    axes.tick_params(labelsize=_LABEL_SIZE)
    return figure, axes


def plot_control_architecture() -> tuple[Figure, plt.Axes]:
    """Diagram of the two controls acting on the shared state.

    :return: Figure and axes.
    """

    figure, axes = plt.subplots(figsize=(10.0, 5.0))
    axes.set_xlim(0, 10)
    axes.set_ylim(0, 6)
    axes.axis("off")

    def _box(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        axes.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.12",
                facecolor=color, edgecolor="black", linewidth=1.0,
            )
        )
        axes.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=9.5,
        )

    _box(0.3, 2.3, 2.4, 1.4, "State S_t\n(X, I, I*, T-t, R)", "#dce6f2")
    _box(3.9, 4.0, 2.6, 1.3, "RFQ response\nquote q or decline", "#e2f0d9")
    _box(3.9, 0.6, 2.6, 1.3, "Active execution\nu in {-2,...,+2}", "#fbe5d6")
    _box(7.4, 2.3, 2.3, 1.4, "Reward\nr_rfq - C_active\n- penalties", "#f2f2f2")
    for start, end in (
        ((2.7, 3.3), (3.9, 4.6)),
        ((2.7, 2.7), (3.9, 1.3)),
        ((6.5, 4.6), (7.4, 3.4)),
        ((6.5, 1.3), (7.4, 2.6)),
    ):
        axes.add_patch(
            FancyArrow(
                start[0], start[1], end[0] - start[0], end[1] - start[1],
                width=0.01, head_width=0.14, length_includes_head=True,
                color="black",
            )
        )
    axes.set_title(
        "Control architecture: one state, two controls (q on RFQs, u actively)",
        fontsize=_TITLE_SIZE,
    )
    return figure, axes


def plot_event_timeline(log: pd.DataFrame, title: str) -> tuple[Figure, plt.Axes]:
    """RFQ arrivals, quotes, declines, fills, and active trades over time.

    :param log: Episode event log.
    :param title: Plot title.
    :return: Figure and axes.
    """

    figure, axes = _new_axes(figsize=(10.0, 4.5))
    arrivals = log.loc[log["rfq_arrived"]]
    buys = arrivals.loc[arrivals["rfq_side_sign"] == 1]
    sells = arrivals.loc[arrivals["rfq_side_sign"] == -1]
    axes.scatter(
        buys["time_index"], buys["rfq_size"],
        marker="^", color="#2ca02c", label="client sells (dealer buys)", s=45,
    )
    axes.scatter(
        sells["time_index"], -sells["rfq_size"],
        marker="v", color="#d62728", label="client buys (dealer sells)", s=45,
    )
    filled = arrivals.loc[arrivals["filled"]]
    axes.scatter(
        filled["time_index"],
        filled["rfq_side_sign"] * filled["rfq_size"],
        facecolors="none", edgecolors="black", s=130, label="filled",
    )
    active = log.loc[log["active_execution_amount"] != 0]
    axes.bar(
        active["time_index"], active["active_execution_amount"],
        color="#ff8c00", alpha=0.6, label="active execution (units)", width=0.8,
    )
    axes.axhline(0.0, color="black", linewidth=0.8)
    axes.set_xlabel("time step", fontsize=_LABEL_SIZE)
    axes.set_ylabel("signed size (inventory units)", fontsize=_LABEL_SIZE)
    axes.set_title(title, fontsize=_TITLE_SIZE)
    axes.legend(fontsize=8.5, loc="best")
    return figure, axes


def plot_inventory_path(log: pd.DataFrame, title: str) -> tuple[Figure, plt.Axes]:
    """Inventory against time with the target and hard limits.

    :param log: Episode event log.
    :param title: Plot title.
    :return: Figure and axes.
    """

    figure, axes = _new_axes(figsize=(10.0, 4.0))
    axes.step(
        log["time_index"], log["inventory_after"],
        where="post", color="#1f77b4", label="inventory (units)",
    )
    axes.axhline(
        float(log["target_inventory"].iloc[0]),
        color="black", linestyle="--", linewidth=1.0, label="target",
    )
    limit = float(log["inventory_after"].abs().max())
    axes.set_xlabel("time step", fontsize=_LABEL_SIZE)
    axes.set_ylabel("inventory (units of $100k notional)", fontsize=_LABEL_SIZE)
    axes.set_title(title, fontsize=_TITLE_SIZE)
    axes.legend(fontsize=9)
    axes.set_ylim(min(-1.0, -1.2 * limit), max(1.0, 1.2 * limit))
    return figure, axes


def plot_target_shortfall(log: pd.DataFrame, title: str) -> tuple[Figure, plt.Axes]:
    """Remaining target shortfall against time.

    :param log: Episode event log.
    :param title: Plot title.
    :return: Figure and axes.
    """

    figure, axes = _new_axes(figsize=(10.0, 3.5))
    axes.step(
        log["time_index"], log["remaining_target_shortfall"],
        where="post", color="#d62728",
    )
    axes.set_xlabel("time step", fontsize=_LABEL_SIZE)
    axes.set_ylabel("|inventory - target| (units)", fontsize=_LABEL_SIZE)
    axes.set_title(title, fontsize=_TITLE_SIZE)
    axes.set_ylim(bottom=-0.3)
    return figure, axes


def plot_action_timeline(log: pd.DataFrame, title: str) -> tuple[Figure, plt.Axes]:
    """Primary action per step (wait / decline / quote / active).

    :param log: Episode event log.
    :param title: Plot title.
    :return: Figure and axes.
    """

    labels = ["wait", "decline_rfq", "quote_rfq", "active_execution"]
    figure, axes = _new_axes(figsize=(10.0, 3.2))
    for position, label in enumerate(labels):
        if label == "active_execution":
            mask = log["active_execution_amount"] != 0
        else:
            mask = log["action"].str.startswith(label)
        rows = log.loc[mask]
        axes.scatter(
            rows["time_index"], np.full(len(rows), position),
            s=40, label=label.replace("_", " "),
        )
    axes.set_yticks(range(len(labels)))
    axes.set_yticklabels([label.replace("_", " ") for label in labels], fontsize=9)
    axes.set_xlabel("time step", fontsize=_LABEL_SIZE)
    axes.set_title(title, fontsize=_TITLE_SIZE)
    return figure, axes


def plot_mode_timeline(log: pd.DataFrame, title: str) -> tuple[Figure, plt.Axes]:
    """Economic mode label of every step as a colored band.

    :param log: Episode event log.
    :param title: Plot title.
    :return: Figure and axes.
    """

    figure, axes = _new_axes(figsize=(10.0, 2.6))
    for _, row in log.iterrows():
        axes.axvspan(
            row["time_index"] - 0.5,
            row["time_index"] + 0.5,
            color=MODE_COLORS[str(row["economic_mode"])],
            alpha=0.85,
        )
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=color,
               markersize=10, label=mode.replace("_", " "))
        for mode, color in MODE_COLORS.items()
    ]
    axes.legend(handles=handles, fontsize=7.5, ncol=3, loc="upper center",
                bbox_to_anchor=(0.5, -0.35))
    axes.set_xlim(-0.5, float(log["time_index"].max()) + 0.5)
    axes.set_yticks([])
    axes.set_xlabel("time step", fontsize=_LABEL_SIZE)
    axes.set_title(title, fontsize=_TITLE_SIZE)
    return figure, axes


def plot_quote_decision_at_event(
    aggressiveness_grid: np.ndarray,
    fill_probability: np.ndarray,
    trade_reward_cents: np.ndarray,
    increments_cents: np.ndarray,
    chosen_aggressiveness: float | None,
    title: str,
) -> tuple[Figure, np.ndarray]:
    """Three-panel decision picture for one RFQ.

    :param aggressiveness_grid: Candidate z values.
    :param fill_probability: p(z).
    :param trade_reward_cents: Standalone r_rfq(z), in cents.
    :param increments_cents: RFQIncrement(z), in cents.
    :param chosen_aggressiveness: Selected z, or None for decline.
    :param title: Figure title.
    :return: Figure and array of axes.
    """

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    axes[0].plot(aggressiveness_grid, fill_probability, color="#1f77b4")
    axes[0].set_ylabel("fill probability", fontsize=_LABEL_SIZE)
    axes[1].plot(aggressiveness_grid, trade_reward_cents, color="#2ca02c")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("standalone RFQ reward (cents)", fontsize=_LABEL_SIZE)
    axes[2].plot(aggressiveness_grid, increments_cents, color="#ff8c00")
    axes[2].axhline(0.0, color="black", linewidth=0.8, label="decline value")
    axes[2].set_ylabel("RFQ increment (cents)", fontsize=_LABEL_SIZE)
    for panel in axes:
        panel.set_xlabel("normalized aggressiveness z", fontsize=_LABEL_SIZE)
        if chosen_aggressiveness is not None:
            panel.axvline(
                chosen_aggressiveness, color="#2ca02c", linestyle=":",
                linewidth=1.5, label="selected quote",
            )
        panel.tick_params(labelsize=9)
    axes[2].legend(fontsize=8)
    figure.suptitle(title, fontsize=_TITLE_SIZE)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return figure, axes


def plot_value_function(
    solution: BellmanSolution,
    regime_index: int,
    steps: tuple[int, ...],
    title: str,
) -> tuple[Figure, plt.Axes]:
    """V_k(I, r) against inventory for several time steps.

    :param solution: Solved Bellman problem.
    :param regime_index: Regime to display.
    :param steps: Time steps to overlay.
    :param title: Plot title.
    :return: Figure and axes.
    """

    figure, axes = _new_axes()
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, len(steps)))
    for color, step in zip(colors, steps):
        axes.plot(
            solution.inventory_grid,
            solution.value[step, :, regime_index],
            color=color,
            label=f"k = {step}",
        )
    axes.axvline(
        solution.episode_config.target_inventory,
        color="black", linestyle="--", linewidth=1.0, label="target",
    )
    axes.set_xlabel("inventory (units)", fontsize=_LABEL_SIZE)
    axes.set_ylabel("value V_k(I, r) (cents)", fontsize=_LABEL_SIZE)
    axes.set_title(title, fontsize=_TITLE_SIZE)
    axes.legend(fontsize=8.5)
    return figure, axes


def plot_inventory_shadow_value(
    solution: BellmanSolution,
    regime_index: int,
    steps: tuple[int, ...],
    title: str,
) -> tuple[Figure, plt.Axes]:
    """Central-difference shadow value dV/dI against inventory.

    :param solution: Solved Bellman problem.
    :param regime_index: Regime to display.
    :param steps: Time steps to overlay.
    :param title: Plot title.
    :return: Figure and axes.
    """

    figure, axes = _new_axes()
    colors = plt.get_cmap("plasma")(np.linspace(0.15, 0.85, len(steps)))
    inventories = solution.inventory_grid
    for color, step in zip(colors, steps):
        shadow = [
            inventory_shadow_value(solution, step, int(inventory), regime_index)
            for inventory in inventories
        ]
        axes.plot(inventories, shadow, color=color, label=f"k = {step}")
    axes.axhline(0.0, color="black", linewidth=0.8)
    axes.axvline(
        solution.episode_config.target_inventory,
        color="black", linestyle="--", linewidth=1.0, label="target",
    )
    axes.set_xlabel("inventory (units)", fontsize=_LABEL_SIZE)
    axes.set_ylabel("shadow value dV/dI (cents per unit)", fontsize=_LABEL_SIZE)
    axes.set_title(title, fontsize=_TITLE_SIZE)
    axes.legend(fontsize=8.5)
    return figure, axes


def plot_quote_policy_heatmap(
    solution: BellmanSolution,
    regime_index: int,
    side_sign: int,
    size: int,
    title: str,
) -> tuple[Figure, plt.Axes]:
    """Optimal quote aggressiveness over (time, inventory); declines masked.

    :param solution: Solved Bellman problem.
    :param regime_index: Regime to display.
    :param side_sign: +1 dealer buy, -1 dealer sell.
    :param size: RFQ size in units.
    :param title: Plot title.
    :return: Figure and axes.
    """

    side_position = SIDE_SIGNS.index(side_sign)
    size_position = solution.size_index(size)
    indices = solution.quote_policy_z_index[
        :, :, regime_index, side_position, size_position
    ]
    z_grid = solution.aggressiveness_grid
    surface = np.where(indices >= 0, z_grid[np.clip(indices, 0, None)], np.nan)
    figure, axes = _new_axes(figsize=(9.0, 4.5))
    mesh = axes.pcolormesh(
        np.arange(surface.shape[0] + 1) - 0.5,
        np.append(solution.inventory_grid - 0.5, solution.inventory_grid[-1] + 0.5),
        surface.T,
        cmap="RdYlGn",
        vmin=float(z_grid.min()),
        vmax=float(z_grid.max()),
    )
    colorbar = figure.colorbar(mesh, ax=axes)
    colorbar.set_label("optimal aggressiveness z (blank = decline)", fontsize=9)
    axes.set_xlabel("time step", fontsize=_LABEL_SIZE)
    axes.set_ylabel("inventory (units)", fontsize=_LABEL_SIZE)
    axes.set_title(title, fontsize=_TITLE_SIZE)
    return figure, axes


def plot_active_execution_policy_heatmap(
    solution: BellmanSolution,
    regime_index: int,
    title: str,
) -> tuple[Figure, plt.Axes]:
    """Optimal active execution amount over (time, inventory).

    :param solution: Solved Bellman problem.
    :param regime_index: Regime to display.
    :param title: Plot title.
    :return: Figure and axes.
    """

    surface = solution.active_policy[:, :, regime_index].astype(float)
    amounts = sorted(
        {int(amount) for amount in np.unique(surface)}
        | set(solution.episode_config.active_action_grid)
    )
    bounds = np.array(amounts, dtype=float)
    figure, axes = _new_axes(figsize=(9.0, 4.5))
    mesh = axes.pcolormesh(
        np.arange(surface.shape[0] + 1) - 0.5,
        np.append(solution.inventory_grid - 0.5, solution.inventory_grid[-1] + 0.5),
        surface.T,
        cmap="coolwarm_r",
        vmin=float(bounds.min()),
        vmax=float(bounds.max()),
    )
    colorbar = figure.colorbar(mesh, ax=axes, ticks=amounts)
    colorbar.set_label("active execution u (units; u < 0 sells)", fontsize=9)
    axes.set_xlabel("time step", fontsize=_LABEL_SIZE)
    axes.set_ylabel("inventory (units)", fontsize=_LABEL_SIZE)
    axes.set_title(title, fontsize=_TITLE_SIZE)
    return figure, axes


def plot_mode_map(
    solution: BellmanSolution,
    regime_index: int,
    side_sign: int,
    size: int,
    title: str,
) -> tuple[Figure, plt.Axes]:
    """Economic mode implied by the solved policies over (time, inventory).

    For each grid state, a quoted RFQ is labelled passive execution
    (toward target), defensive market making (away from target, quoted
    passively), or market making; declined RFQs fall back to active
    execution when u is nonzero, otherwise decline.

    :param solution: Solved Bellman problem.
    :param regime_index: Regime to display.
    :param side_sign: RFQ side used for the map.
    :param size: RFQ size used for the map.
    :param title: Plot title.
    :return: Figure and axes.
    """

    side_position = SIDE_SIGNS.index(side_sign)
    size_position = solution.size_index(size)
    z_grid = solution.aggressiveness_grid
    quote_indices = solution.quote_policy_z_index[
        :, :, regime_index, side_position, size_position
    ]
    active = solution.active_policy[:, :, regime_index]
    target = solution.episode_config.target_inventory
    n_steps, n_inventory = quote_indices.shape
    mode_codes = np.zeros((n_steps, n_inventory), dtype=int)
    for step in range(n_steps):
        for position in range(n_inventory):
            inventory = int(solution.inventory_grid[position])
            code = _mode_code(
                quote_index=int(quote_indices[step, position]),
                active_amount=int(active[step, position]),
                inventory=inventory,
                target=target,
                side_sign=side_sign,
                size=size,
                z_grid=z_grid,
            )
            mode_codes[step, position] = code
    cmap = ListedColormap([MODE_COLORS[mode] for mode in MODE_ORDER])
    norm = BoundaryNorm(np.arange(len(MODE_ORDER) + 1) - 0.5, len(MODE_ORDER))
    figure, axes = _new_axes(figsize=(9.5, 4.8))
    axes.pcolormesh(
        np.arange(n_steps + 1) - 0.5,
        np.append(solution.inventory_grid - 0.5, solution.inventory_grid[-1] + 0.5),
        mode_codes.T,
        cmap=cmap,
        norm=norm,
    )
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=MODE_COLORS[mode],
               markersize=10, label=mode.replace("_", " "))
        for mode in MODE_ORDER
    ]
    axes.legend(handles=handles, fontsize=7.5, ncol=3, loc="upper center",
                bbox_to_anchor=(0.5, -0.18))
    axes.set_xlabel("time step", fontsize=_LABEL_SIZE)
    axes.set_ylabel("inventory (units)", fontsize=_LABEL_SIZE)
    axes.set_title(title, fontsize=_TITLE_SIZE)
    return figure, axes


def _mode_code(
    quote_index: int,
    active_amount: int,
    inventory: int,
    target: int,
    side_sign: int,
    size: int,
    z_grid: np.ndarray,
) -> int:
    from rfq_edge.control_state import DEFENSIVE_AGGRESSIVENESS_THRESHOLD

    if quote_index >= 0:
        shortfall_before = abs(inventory - target)
        shortfall_after = abs(inventory + side_sign * size - target)
        if shortfall_after < shortfall_before:
            return MODE_ORDER.index(EconomicMode.PASSIVE_EXECUTION.value)
        if (
            shortfall_after > shortfall_before
            and float(z_grid[quote_index]) <= DEFENSIVE_AGGRESSIVENESS_THRESHOLD
        ):
            return MODE_ORDER.index(EconomicMode.DEFENSIVE_MARKET_MAKING.value)
        return MODE_ORDER.index(EconomicMode.MARKET_MAKING.value)
    if active_amount != 0:
        return MODE_ORDER.index(EconomicMode.ACTIVE_EXECUTION.value)
    return MODE_ORDER.index(EconomicMode.DECLINE.value)


def plot_bellman_residual(
    solution: BellmanSolution,
    title: str,
) -> tuple[Figure, plt.Axes]:
    """Maximum absolute Bellman residual per time step, log scale.

    :param solution: Solved Bellman problem.
    :param title: Plot title.
    :return: Figure and axes.
    """

    stats = bellman_residual(solution)
    per_step = np.asarray(stats["per_step_max_abs_residual_cents"], dtype=float)
    figure, axes = _new_axes(figsize=(8.0, 3.5))
    floored = np.maximum(per_step, 1e-18)
    axes.semilogy(np.arange(len(per_step)), floored, color="#1f77b4")
    axes.axhline(
        stats["tolerance_cents"], color="#d62728", linestyle="--",
        label=f"tolerance = {stats['tolerance_cents']:.0e} cents",
    )
    axes.set_xlabel("time step", fontsize=_LABEL_SIZE)
    axes.set_ylabel("max |residual| (cents, log scale)", fontsize=_LABEL_SIZE)
    axes.set_title(title, fontsize=_TITLE_SIZE)
    axes.legend(fontsize=9)
    return figure, axes


def plot_reward_decomposition(
    episode_summaries: pd.DataFrame,
    episode_name: str,
    title: str,
) -> tuple[Figure, plt.Axes]:
    """Mean reward decomposition per policy as stacked bars.

    :param episode_summaries: Per-episode summaries from evaluation.
    :param episode_name: Episode configuration to display.
    :param title: Plot title.
    :return: Figure and axes.
    """

    subset = episode_summaries.loc[episode_summaries["episode_name"] == episode_name]
    means = subset.groupby("policy", sort=False).mean(numeric_only=True)
    components = pd.DataFrame(
        {
            "apparent edge": means["gross_apparent_edge_cents"],
            "adverse selection": -means["adverse_selection_cents"],
            "RFQ costs": -means["rfq_costs_cents"],
            "active costs": -means["active_execution_cost_cents"],
            "running penalty": -means["running_inventory_penalty_cents"],
            "terminal penalty": -means["terminal_penalty_cents"],
        }
    )
    figure, axes = _new_axes(figsize=(9.5, 4.8))
    positions = np.arange(len(components.index))
    bottom_positive = np.zeros(len(components.index))
    bottom_negative = np.zeros(len(components.index))
    palette = ("#2ca02c", "#d62728", "#9467bd", "#ff8c00", "#8c564b", "#7f7f7f")
    for color, column in zip(palette, components.columns):
        values = components[column].to_numpy(dtype=float)
        bottom = np.where(values >= 0.0, bottom_positive, bottom_negative)
        axes.bar(positions, values, bottom=bottom, color=color, label=column, width=0.6)
        bottom_positive = bottom_positive + np.clip(values, 0.0, None)
        bottom_negative = bottom_negative + np.clip(values, None, 0.0)
    totals = means["total_objective_cents"].to_numpy(dtype=float)
    axes.scatter(
        positions, totals, color="black", zorder=5, marker="D",
        label="total simulated control reward",
    )
    axes.axhline(0.0, color="black", linewidth=0.8)
    axes.set_xticks(positions)
    axes.set_xticklabels(components.index, fontsize=8.5, rotation=12)
    axes.set_ylabel("cents per episode", fontsize=_LABEL_SIZE)
    axes.set_title(title, fontsize=_TITLE_SIZE)
    axes.legend(fontsize=8, ncol=2)
    return figure, axes


def plot_policy_comparison(
    policy_metrics: pd.DataFrame,
    episode_name: str,
    metric: str,
    ylabel: str,
    title: str,
) -> tuple[Figure, plt.Axes]:
    """Bar chart of one metric across policies for one episode type.

    :param policy_metrics: Aggregated policy metrics.
    :param episode_name: Episode configuration to display.
    :param metric: Metric column to plot.
    :param ylabel: Axis label including units.
    :param title: Plot title.
    :return: Figure and axes.
    """

    subset = policy_metrics.loc[policy_metrics["episode_name"] == episode_name]
    figure, axes = _new_axes(figsize=(8.5, 4.2))
    positions = np.arange(len(subset))
    colors = [POLICY_COLORS.get(policy, "#333333") for policy in subset["policy"]]
    axes.bar(positions, subset[metric].to_numpy(dtype=float), color=colors, width=0.6)
    axes.set_xticks(positions)
    axes.set_xticklabels(subset["policy"], fontsize=8.5, rotation=12)
    axes.set_ylabel(ylabel, fontsize=_LABEL_SIZE)
    axes.set_title(title, fontsize=_TITLE_SIZE)
    axes.axhline(0.0, color="black", linewidth=0.8)
    return figure, axes


def plot_completion_cost_frontier(
    episode_summaries: pd.DataFrame,
    episode_name: str,
    title: str,
) -> tuple[Figure, plt.Axes]:
    """Target completion against total execution cost per policy.

    :param episode_summaries: Per-episode summaries.
    :param episode_name: Episode configuration to display.
    :param title: Plot title.
    :return: Figure and axes.
    """

    subset = episode_summaries.loc[episode_summaries["episode_name"] == episode_name]
    means = subset.groupby("policy", sort=False).mean(numeric_only=True)
    figure, axes = _new_axes(figsize=(7.5, 4.5))
    for policy, row in means.iterrows():
        cost = row["active_execution_cost_cents"] + row["rfq_costs_cents"]
        axes.scatter(
            cost, row["target_completion_pct"],
            color=POLICY_COLORS.get(str(policy), "#333333"),
            s=90, label=str(policy),
            marker="o" if policy != "OracleDynamic" else "s",
        )
    axes.set_xlabel("execution cost per episode (cents)", fontsize=_LABEL_SIZE)
    axes.set_ylabel("target completion (%)", fontsize=_LABEL_SIZE)
    axes.set_title(title, fontsize=_TITLE_SIZE)
    axes.legend(fontsize=8)
    return figure, axes


def plot_internalization_fraction(
    policy_metrics: pd.DataFrame,
    episode_name: str,
    title: str,
) -> tuple[Figure, plt.Axes]:
    """Share of target-directed volume internalized through RFQs.

    :param policy_metrics: Aggregated policy metrics.
    :param episode_name: Episode configuration to display.
    :param title: Plot title.
    :return: Figure and axes.
    """

    figure, axes = plot_policy_comparison(
        policy_metrics=policy_metrics,
        episode_name=episode_name,
        metric="proportion_via_rfqs",
        ylabel="fraction of target-directed volume via RFQs",
        title=title,
    )
    axes.set_ylim(0.0, 1.05)
    return figure, axes


def plot_regime_performance(
    regime_metrics: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
) -> tuple[Figure, plt.Axes]:
    """One RFQ metric per regime, grouped by policy.

    :param regime_metrics: Step-level regime metrics from evaluation.
    :param metric: Metric column to plot.
    :param ylabel: Axis label including units.
    :param title: Plot title.
    :return: Figure and axes.
    """

    regimes = list(dict.fromkeys(regime_metrics["regime"]))
    policies = list(dict.fromkeys(regime_metrics["policy"]))
    figure, axes = _new_axes(figsize=(9.0, 4.2))
    width = 0.8 / max(len(policies), 1)
    for offset, policy in enumerate(policies):
        rows = regime_metrics.loc[regime_metrics["policy"] == policy]
        values = [
            float(rows.loc[rows["regime"] == regime, metric].iloc[0])
            if (rows["regime"] == regime).any()
            else float("nan")
            for regime in regimes
        ]
        axes.bar(
            np.arange(len(regimes)) + offset * width,
            values,
            width=width,
            color=POLICY_COLORS.get(policy, "#333333"),
            label=policy,
        )
    axes.set_xticks(np.arange(len(regimes)) + 0.4 - width / 2)
    axes.set_xticklabels(regimes, fontsize=9)
    axes.set_ylabel(ylabel, fontsize=_LABEL_SIZE)
    axes.set_title(title, fontsize=_TITLE_SIZE)
    axes.legend(fontsize=8)
    return figure, axes


def plot_sensitivity_heatmap(
    sensitivity: pd.DataFrame,
    metric: str,
    title: str,
) -> tuple[Figure, plt.Axes]:
    """Scenario-by-policy heatmap of one sensitivity metric.

    :param sensitivity: Output of run_control_sensitivity.
    :param metric: Metric column to display.
    :param title: Plot title.
    :return: Figure and axes.
    """

    table = sensitivity.pivot_table(
        index="scenario", columns="policy", values=metric, sort=False
    )
    figure, axes = _new_axes(figsize=(8.5, 0.5 * len(table.index) + 2.0))
    mesh = axes.pcolormesh(table.to_numpy(dtype=float), cmap="RdYlGn")
    axes.set_xticks(np.arange(len(table.columns)) + 0.5)
    axes.set_xticklabels(table.columns, fontsize=8.5, rotation=15)
    axes.set_yticks(np.arange(len(table.index)) + 0.5)
    axes.set_yticklabels(table.index, fontsize=8.5)
    colorbar = figure.colorbar(mesh, ax=axes)
    colorbar.set_label(metric, fontsize=9)
    axes.set_title(title, fontsize=_TITLE_SIZE)
    return figure, axes
