"""Hostile-audit tests: reconciliation, formula consistency, and ordering.

These tests encode the audit obligations directly:

* every reward component appears exactly once and sums to the total;
* the dynamic controller's decision equals a brute-force evaluation of the
  published RFQ jump formula;
* probabilities are bounded and value functions finite;
* the event ordering of the forward simulator matches the documented one;
* the controller waits when future RFQ flow is cheaper than immediate
  active execution.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfq_edge.bellman import solve_bellman
from rfq_edge.control_config import (
    MarketRegime,
    liquidation_episode,
    market_making_episode,
)
from rfq_edge.control_pipeline import POLICY_ORDER, make_policies, solve_episode_policies
from rfq_edge.control_reporting import reconcile_episode_rewards
from rfq_edge.control_state import ActionType, ControlState, RFQEvent
from rfq_edge.controllers import DynamicExecutionController
from rfq_edge.dynamic_objective import (
    expected_rfq_reward_cents,
    rfq_increment_cents,
)
from rfq_edge.event_simulator import simulate_episode
from rfq_edge.market_dynamics import ExogenousPath, simulate_exogenous_path


def test_reward_reconciliation_holds_for_every_policy(control_artifacts) -> None:
    # total reward = RFQ clean reward - active cost - running penalty
    #                - terminal penalty, exactly, for all five policies.
    config = liquidation_episode()
    solutions = solve_episode_policies(control_artifacts, config)
    policies = make_policies(control_artifacts, config, solutions)
    for seed in (0, 1, 2, 3):
        path = simulate_exogenous_path(
            control_artifacts.market_config, config, random_state=seed
        )
        for name in POLICY_ORDER:
            result = simulate_episode(
                policies[name], config, control_artifacts.market_config, path
            )
            components = reconcile_episode_rewards(result.log)
            assert result.total_reward_cents == pytest.approx(
                components["total_reward_cents"]
            ), name


def test_dynamic_decision_matches_published_jump_formula(control_artifacts) -> None:
    config = liquidation_episode()
    solution = solve_bellman(
        config, control_artifacts.market_config, control_artifacts.fitted_models
    )
    policy = DynamicExecutionController(
        control_artifacts.market_config, control_artifacts.fitted_models, solution
    )
    event = RFQEvent(
        event_id=0, time_index=15, side="dealer_sell", side_sign=-1, size=2,
        client_tier="professional", liquidity_score=0.55, market_width=0.12,
        regime=MarketRegime.NORMAL, cp_plus=100.0,
        hidden_client_signal=0.0, hidden_future_residual=0.0,
    )
    state = ControlState(
        time_index=15, time_remaining=config.n_steps - 15,
        market_regime=MarketRegime.NORMAL, inventory=6, target_inventory=0,
        initial_inventory=8, inventory_limit=config.inventory_limit,
        current_cp_plus=100.0, volatility=0.08, liquidity_score=0.55,
        market_width=0.12, active_execution_available=True,
    )
    action = policy.respond_to_rfq(state, event)

    # Brute force using only public reward functions and solution arrays.
    z_grid = solution.aggressiveness_grid
    p_win = control_artifacts.fitted_models.event_fill_probability(event, z_grid)
    post_win = control_artifacts.fitted_models.event_post_win_value(event, z_grid)
    quotes = event.cp_plus + event.side_sign * z_grid * event.market_width
    rewards = expected_rfq_reward_cents(
        post_win_value=post_win, quote=quotes, side_sign=event.side_sign,
        size=event.size,
        transaction_cost_cents=(
            control_artifacts.market_config.rfq_transaction_cost_cents
        ),
    )
    continuation = solution.post_rfq_value[15, :, MarketRegime.NORMAL.value]
    increments = rfq_increment_cents(
        fill_probability=p_win,
        trade_reward_cents=rewards,
        continuation_after_fill=float(continuation[solution.inventory_index(4)]),
        continuation_without_fill=float(continuation[solution.inventory_index(6)]),
    )
    best = int(np.argmax(increments))
    assert action.action_type is ActionType.QUOTE_RFQ
    assert action.normalized_aggressiveness == pytest.approx(float(z_grid[best]))


def test_probabilities_are_bounded_on_the_full_grid(control_artifacts) -> None:
    z_grid = np.asarray(control_artifacts.market_config.aggressiveness_grid)
    for regime_index in range(3):
        for size in (1, 2, 3):
            for models in (
                control_artifacts.fitted_models, control_artifacts.oracle_models
            ):
                p = models.fill_probability(regime_index, z_grid, size)
                assert (p > 0.0).all() and (p < 1.0).all()
        assert (
            control_artifacts.oracle_models.selection_points(regime_index, z_grid)
            > 0.0
        ).all()


def test_controller_waits_when_future_rfqs_are_cheaper(control_artifacts) -> None:
    # A one-unit shortfall with a long horizon should be left to RFQ flow;
    # active execution appears only once the deadline threatens.
    solution = solve_bellman(
        liquidation_episode(),
        control_artifacts.market_config,
        control_artifacts.fitted_models,
    )
    one_unit = solution.inventory_index(1)
    for regime_index in range(3):
        assert (solution.active_policy[:5, one_unit, regime_index] == 0).all()
    final = solution.episode_config.n_steps - 1
    assert int(solution.active_policy[final, one_unit, 0]) < 0


def test_simulator_event_ordering_is_rfq_then_active_then_penalty(
    control_artifacts,
) -> None:
    # One step: fill (+2) is applied before active (-1); the running penalty
    # is charged on the end-of-step inventory (+1), not on any intermediate.
    from rfq_edge.control_config import with_overrides
    from rfq_edge.control_state import ControlAction

    config = with_overrides(
        market_making_episode(), n_steps=1, running_penalty_cents=1.0
    )
    event = RFQEvent(
        event_id=0, time_index=0, side="dealer_buy", side_sign=1, size=2,
        client_tier="retail", liquidity_score=0.55, market_width=0.12,
        regime=MarketRegime.NORMAL, cp_plus=100.0,
        hidden_client_signal=0.0, hidden_future_residual=0.0,
    )
    path = ExogenousPath(
        regimes=np.array([MarketRegime.NORMAL.value]),
        cp_plus=np.array([100.0]),
        events=(event,),
        fill_uniforms=np.array([0.0]),
    )

    class _Scripted:
        name = "scripted"

        def respond_to_rfq(self, state, rfq):
            return ControlAction(
                action_type=ActionType.QUOTE_RFQ, quote=100.0,
                normalized_aggressiveness=0.0, active_execution_amount=0,
                respond_or_decline=True, policy_name=self.name,
            )

        def choose_active_execution(self, state):
            # The active stage must see the post-fill inventory.
            assert state.inventory == 2
            return ControlAction(
                action_type=ActionType.ACTIVE_EXECUTION, quote=None,
                normalized_aggressiveness=None, active_execution_amount=-1,
                respond_or_decline=False, policy_name=self.name,
            )

    result = simulate_episode(
        _Scripted(), config, control_artifacts.market_config, path
    )
    row = result.log.iloc[0]
    assert row["inventory_before"] == 0
    assert row["inventory_after_rfq"] == 2
    assert row["inventory_after"] == 1
    assert row["running_inventory_penalty_cents"] == pytest.approx(1.0)


def test_value_functions_are_finite_for_all_episode_solutions(
    control_artifacts,
) -> None:
    for config in (market_making_episode(), liquidation_episode()):
        for models in (
            control_artifacts.fitted_models, control_artifacts.oracle_models
        ):
            solution = solve_bellman(
                config, control_artifacts.market_config, models
            )
            assert np.isfinite(solution.value).all()
            assert np.isfinite(solution.post_rfq_value).all()
