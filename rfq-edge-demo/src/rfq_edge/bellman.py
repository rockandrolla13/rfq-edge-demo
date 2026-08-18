"""Finite-horizon Bellman solver for the event-driven control problem.

The solver performs exact backward induction on a finite grid of
(time step k, integer inventory I, market regime r). It is a discrete
Bellman approximation to a finite-horizon jump-HJB — not an exact
continuous-time PDE solution.

Event ordering inside one step matches the forward simulator exactly:

1. Post-RFQ stage value U_k(I, r) folds in the optimal active execution,
   the running inventory penalty on end-of-step inventory, and the regime
   transition into V_{k+1}:

       U_k(I, r) = max_u [ -C_active(u, r)
                           - phi * (I + u - I_target)^2
                           + sum_r' P(r, r') V_{k+1}(I + u, r') ].

2. Start-of-step value V_k(I, r) adds the RFQ jump term:

       V_k(I, r) = U_k(I, r)
                   + lambda_r * E_{side, size}[ max(0, max_q RFQIncrement(q)) ]

       RFQIncrement(q) = p(q, X) * [ r_rfq(q, X, n)
                                     + U_k(I + sigma * n, r)
                                     - U_k(I, r) ].

   ``U_k`` plays the role of "V_next" in the RFQ jump operator: it is the
   continuation value immediately after the RFQ stage of step k. The
   continuation difference is the entire dynamic inventory value; no
   separate inventory adjustment is layered on top.

3. Terminal condition: V_T(I, r) = -eta * (I - I_target)^2.

Fills or active trades that would breach the inventory limit are excluded
explicitly (forced decline / masked action), never silently clipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rfq_edge.control_config import (
    ControlMarketConfig,
    EpisodeConfig,
    POINTS_TO_CENTS,
    REGIME_ORDER,
)
from rfq_edge.execution_costs import active_execution_cost_cents

SIDE_SIGNS: tuple[int, int] = (1, -1)
RESIDUAL_TOLERANCE_CENTS = 1e-6


class PlanningQuoteModel(Protocol):
    """Marginal quote model used inside the Bellman recursion."""

    def fill_probability(
        self, regime_index: int, aggressiveness: np.ndarray, size: int
    ) -> np.ndarray:
        """Marginal fill probability per candidate aggressiveness."""

    def selection_points(
        self, regime_index: int, aggressiveness: np.ndarray
    ) -> np.ndarray:
        """Marginal adverse selection per candidate, in points."""


@dataclass(frozen=True)
class PlanningTables:
    """Precomputed per-regime quantities used by the backward recursion.

    :param episode_config: Episode configuration.
    :param inventory_grid: Integer inventories from -limit to +limit.
    :param aggressiveness_grid: Candidate normalized aggressiveness values.
    :param size_values: Union of RFQ sizes across regimes, sorted.
    :param transition: Regime transition matrix.
    :param arrival_probability: RFQ arrival probability per regime.
    :param size_weights: P(size | regime), aligned with ``size_values``.
    :param fill_probability: p(z) per (regime, size, z).
    :param trade_reward_cents: r_rfq per (regime, size, z), in cents.
    :param active_grid: Admissible active execution amounts, zero first.
    :param active_costs: C_active per (regime, action), in cents.
    """

    episode_config: EpisodeConfig
    inventory_grid: np.ndarray
    aggressiveness_grid: np.ndarray
    size_values: tuple[int, ...]
    transition: np.ndarray
    arrival_probability: np.ndarray
    size_weights: np.ndarray
    fill_probability: np.ndarray
    trade_reward_cents: np.ndarray
    active_grid: tuple[int, ...]
    active_costs: np.ndarray


@dataclass(frozen=True)
class BellmanSolution:
    """Solved value function and greedy policies on the grid.

    :param episode_config: Episode the solution was solved for.
    :param market_config: Market configuration used.
    :param planning_tables: Tables the recursion was evaluated on.
    :param value: V_k(I, r), shape (T + 1, n_inventory, n_regimes), cents.
    :param post_rfq_value: U_k(I, r), shape (T, n_inventory, n_regimes).
    :param active_policy: Optimal signed active execution at the active
        stage of each step, shape (T, n_inventory, n_regimes).
    :param quote_policy_z_index: Optimal aggressiveness index per
        (k, I, r, side, size); -1 encodes decline.
    """

    episode_config: EpisodeConfig
    market_config: ControlMarketConfig
    planning_tables: PlanningTables
    value: np.ndarray
    post_rfq_value: np.ndarray
    active_policy: np.ndarray
    quote_policy_z_index: np.ndarray

    @property
    def inventory_grid(self) -> np.ndarray:
        """Integer inventory grid of the solution."""

        return self.planning_tables.inventory_grid

    @property
    def aggressiveness_grid(self) -> np.ndarray:
        """Candidate aggressiveness grid of the solution."""

        return self.planning_tables.aggressiveness_grid

    @property
    def size_values(self) -> tuple[int, ...]:
        """RFQ sizes covered by the quote policy."""

        return self.planning_tables.size_values

    def inventory_index(self, inventory: int) -> int:
        """Map an inventory level to its grid index.

        :param inventory: Inventory in units.
        :return: Grid index.
        :raises ValueError: If the inventory is outside the grid.
        """

        index = inventory + self.episode_config.inventory_limit
        if index < 0 or index >= len(self.inventory_grid):
            raise ValueError(f"inventory {inventory} is outside the solved grid")
        return index

    def size_index(self, size: int) -> int:
        """Map an RFQ size to its policy-array index.

        :param size: RFQ size in units.
        :return: Index into the size axis.
        :raises ValueError: If the size was not part of the planning grid.
        """

        if size not in self.size_values:
            raise ValueError(f"size {size} was not part of the planning grid")
        return self.size_values.index(size)


def build_planning_tables(
    episode_config: EpisodeConfig,
    market_config: ControlMarketConfig,
    planning_models: PlanningQuoteModel,
) -> PlanningTables:
    """Precompute all regime-level quantities the recursion needs.

    :param episode_config: Episode configuration.
    :param market_config: Market configuration.
    :param planning_models: Marginal fill and selection model.
    :return: Planning tables.
    """

    limit = episode_config.inventory_limit
    inventory_grid = np.arange(-limit, limit + 1, dtype=int)
    z_grid = np.asarray(market_config.aggressiveness_grid, dtype=float)
    all_sizes = sorted(
        {
            size
            for params in market_config.regime_parameters
            for size in params.size_values
        }
    )
    n_regimes = len(REGIME_ORDER)
    n_sizes = len(all_sizes)
    size_weights = np.zeros((n_regimes, n_sizes))
    fill_probability = np.zeros((n_regimes, n_sizes, len(z_grid)))
    trade_reward = np.zeros((n_regimes, n_sizes, len(z_grid)))
    arrival = np.zeros(n_regimes)
    if episode_config.active_execution_allowed:
        # Zero first so exact ties resolve to waiting.
        active_grid = tuple(
            sorted(episode_config.active_action_grid, key=lambda u: (abs(u), u))
        )
    else:
        active_grid = (0,)
    active_costs = np.zeros((n_regimes, len(active_grid)))

    for regime_index, regime in enumerate(REGIME_ORDER):
        params = market_config.parameters_for(regime)
        arrival[regime_index] = params.arrival_probability
        selection = planning_models.selection_points(regime_index, z_grid)
        # side_sign * (m - q) = -(A + z * width) when m = V0 - side_sign * A.
        edge_cents = -(selection + z_grid * params.market_width) * POINTS_TO_CENTS
        for size_value, probability in zip(
            params.size_values, params.size_probabilities
        ):
            size_position = all_sizes.index(size_value)
            size_weights[regime_index, size_position] = probability
            fill_probability[regime_index, size_position] = (
                planning_models.fill_probability(regime_index, z_grid, size_value)
            )
            trade_reward[regime_index, size_position] = float(size_value) * (
                edge_cents - market_config.rfq_transaction_cost_cents
            )
        for action_position, amount in enumerate(active_grid):
            active_costs[regime_index, action_position] = active_execution_cost_cents(
                amount, params
            )

    return PlanningTables(
        episode_config=episode_config,
        inventory_grid=inventory_grid,
        aggressiveness_grid=z_grid,
        size_values=tuple(all_sizes),
        transition=np.asarray(market_config.transition_matrix, dtype=float),
        arrival_probability=arrival,
        size_weights=size_weights,
        fill_probability=fill_probability,
        trade_reward_cents=trade_reward,
        active_grid=active_grid,
        active_costs=active_costs,
    )


def solve_bellman(
    episode_config: EpisodeConfig,
    market_config: ControlMarketConfig,
    planning_models: PlanningQuoteModel,
) -> BellmanSolution:
    """Solve the finite-horizon control problem by exact backward induction.

    :param episode_config: Episode configuration (horizon, penalties, limits).
    :param market_config: Market configuration (regimes, arrivals, costs).
    :param planning_models: Marginal fill and selection model (fitted or
        oracle-marginal); it never receives hidden per-event variables.
    :return: Solved value function and greedy policies.
    """

    tables = build_planning_tables(episode_config, market_config, planning_models)
    n_steps = episode_config.n_steps
    n_inventory = len(tables.inventory_grid)
    n_regimes = len(REGIME_ORDER)

    value = np.zeros((n_steps + 1, n_inventory, n_regimes))
    post_rfq_value = np.zeros((n_steps, n_inventory, n_regimes))
    active_policy = np.zeros((n_steps, n_inventory, n_regimes), dtype=int)
    quote_policy = np.full(
        (n_steps, n_inventory, n_regimes, len(SIDE_SIGNS), len(tables.size_values)),
        -1,
        dtype=int,
    )

    deviation = tables.inventory_grid.astype(float) - float(
        episode_config.target_inventory
    )
    value[n_steps] = np.repeat(
        (-episode_config.terminal_penalty_cents * deviation**2)[:, None],
        n_regimes,
        axis=1,
    )

    for step in range(n_steps - 1, -1, -1):
        step_value, step_u, step_active, step_quote = _backward_step(
            value_next=value[step + 1],
            tables=tables,
        )
        value[step] = step_value
        post_rfq_value[step] = step_u
        active_policy[step] = step_active
        quote_policy[step] = step_quote

    return BellmanSolution(
        episode_config=episode_config,
        market_config=market_config,
        planning_tables=tables,
        value=value,
        post_rfq_value=post_rfq_value,
        active_policy=active_policy,
        quote_policy_z_index=quote_policy,
    )


def bellman_residual(solution: BellmanSolution) -> dict[str, float]:
    """Re-evaluate the recursion at the solved values and report residuals.

    :param solution: Solved Bellman problem.
    :return: Maximum and mean absolute residual in cents, the count of grid
        states above tolerance, and the tolerance used.
    """

    residuals = []
    for step in range(solution.episode_config.n_steps):
        recomputed, _, _, _ = _backward_step(
            value_next=solution.value[step + 1],
            tables=solution.planning_tables,
        )
        residuals.append(np.abs(recomputed - solution.value[step]))
    stacked = np.stack(residuals)
    return {
        "max_abs_residual_cents": float(stacked.max()),
        "mean_abs_residual_cents": float(stacked.mean()),
        "n_states_violating_tolerance": int(
            (stacked > RESIDUAL_TOLERANCE_CENTS).sum()
        ),
        "tolerance_cents": RESIDUAL_TOLERANCE_CENTS,
    }


def inventory_shadow_value(
    solution: BellmanSolution,
    step: int,
    inventory: int,
    regime_index: int,
) -> float:
    """Marginal value of one extra unit of inventory, in cents.

    Central finite difference where possible, one-sided at the grid bounds:
    dV/dI ~ [V(k, I + 1, r) - V(k, I - 1, r)] / 2.

    :param solution: Solved Bellman problem.
    :param step: Time index k.
    :param inventory: Inventory level I.
    :param regime_index: Regime index r.
    :return: Shadow value in cents per unit.
    """

    limit = solution.episode_config.inventory_limit
    values = solution.value[step, :, regime_index]
    index = solution.inventory_index(inventory)
    if inventory == limit:
        return float(values[index] - values[index - 1])
    if inventory == -limit:
        return float(values[index + 1] - values[index])
    return float((values[index + 1] - values[index - 1]) / 2.0)


def fill_continuation_delta(
    solution: BellmanSolution,
    step: int,
    inventory: int,
    side_sign: int,
    size: int,
    regime_index: int,
) -> float:
    """Discrete fill continuation value DeltaV_fill, in cents.

    DeltaV_fill = U_k(I + side_sign * size, r) - U_k(I, r), where U_k is the
    post-RFQ-stage continuation used by the RFQ jump operator.

    :param solution: Solved Bellman problem.
    :param step: Time index k.
    :param inventory: Inventory before the fill.
    :param side_sign: +1 dealer buy, -1 dealer sell.
    :param size: RFQ size in units.
    :param regime_index: Regime index r.
    :return: Continuation difference in cents.
    :raises ValueError: If the fill would leave the solved grid.
    """

    index_before = solution.inventory_index(inventory)
    index_after = solution.inventory_index(inventory + side_sign * size)
    values = solution.post_rfq_value[step, :, regime_index]
    return float(values[index_after] - values[index_before])


def _backward_step(
    value_next: np.ndarray,
    tables: PlanningTables,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One exact backward induction step following the documented ordering.

    :param value_next: V_{k+1}, shape (n_inventory, n_regimes).
    :param tables: Precomputed planning tables.
    :return: (V_k, U_k, active policy, quote policy indices).
    """

    config = tables.episode_config
    n_inventory, n_regimes = value_next.shape
    limit = config.inventory_limit
    target = float(config.target_inventory)
    inventories = tables.inventory_grid.astype(float)

    # Expectation over the next regime: EV[i, r] = sum_r' P(r, r') V_next[i, r'].
    expected_next = value_next @ tables.transition.T

    # Active execution stage: U_k(I, r) = max_u ActiveValue(u).
    post_rfq = np.full((n_inventory, n_regimes), -np.inf)
    active_policy = np.zeros((n_inventory, n_regimes), dtype=int)
    for action_position, amount in enumerate(tables.active_grid):
        shifted_inventory = inventories + amount
        valid = np.abs(shifted_inventory) <= limit
        source = np.clip(np.arange(n_inventory) + amount, 0, n_inventory - 1)
        penalty = config.running_penalty_cents * (shifted_inventory - target) ** 2
        candidate = (
            expected_next[source]
            - tables.active_costs[:, action_position][None, :]
            - penalty[:, None]
        )
        candidate[~valid] = -np.inf
        better = candidate > post_rfq
        post_rfq = np.where(better, candidate, post_rfq)
        active_policy = np.where(better, amount, active_policy)

    # RFQ jump stage: expected gain over side, size, and best quote.
    n_sides = len(SIDE_SIGNS)
    n_sizes = len(tables.size_values)
    quote_policy = np.full((n_inventory, n_regimes, n_sides, n_sizes), -1, dtype=int)
    expected_gain = np.zeros((n_inventory, n_regimes))
    for side_position, side_sign in enumerate(SIDE_SIGNS):
        for size_position, size_value in enumerate(tables.size_values):
            shift = side_sign * size_value
            valid = np.abs(tables.inventory_grid + shift) <= limit
            source = np.clip(np.arange(n_inventory) + shift, 0, n_inventory - 1)
            continuation_delta = post_rfq[source] - post_rfq
            for regime_index in range(n_regimes):
                weight = tables.size_weights[regime_index, size_position]
                if weight == 0.0:
                    continue
                p_vector = tables.fill_probability[regime_index, size_position]
                reward_vector = tables.trade_reward_cents[regime_index, size_position]
                increments = p_vector[None, :] * (
                    reward_vector[None, :]
                    + continuation_delta[:, regime_index][:, None]
                )
                best_index = np.argmax(increments, axis=1)
                best_gain = increments[np.arange(n_inventory), best_index]
                respond = valid & (best_gain > 0.0)
                gain = np.where(respond, best_gain, 0.0)
                quote_policy[:, regime_index, side_position, size_position] = np.where(
                    respond, best_index, -1
                )
                # Sides are equally likely: weight = 0.5 * P(size | regime).
                expected_gain[:, regime_index] += 0.5 * weight * gain

    value_now = post_rfq + tables.arrival_probability[None, :] * expected_gain
    return value_now, post_rfq, active_policy, quote_policy
