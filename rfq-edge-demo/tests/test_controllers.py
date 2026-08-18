"""Behavioral tests for the five control policies."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rfq_edge.bellman import solve_bellman
from rfq_edge.control_config import (
    EpisodeConfig,
    MarketRegime,
    liquidation_episode,
    with_overrides,
)
from rfq_edge.control_state import ActionType, ControlState, RFQEvent
from rfq_edge.control_pipeline import make_policies, solve_episode_policies
from rfq_edge.controllers import (
    DynamicExecutionController,
    EdgeConsistentMyopicResponder,
    OracleDynamicController,
    PlainResponder,
)
from rfq_edge.event_simulator import simulate_episode
from rfq_edge.market_dynamics import simulate_exogenous_path


def _event(
    side_sign: int,
    size: int = 1,
    regime: MarketRegime = MarketRegime.NORMAL,
    hidden_client_signal: float = 0.0,
    market_width: float = 0.12,
) -> RFQEvent:
    return RFQEvent(
        event_id=0,
        time_index=0,
        side="dealer_buy" if side_sign == 1 else "dealer_sell",
        side_sign=side_sign,
        size=size,
        client_tier="professional",
        liquidity_score=0.55,
        market_width=market_width,
        regime=regime,
        cp_plus=100.0,
        hidden_client_signal=hidden_client_signal,
        hidden_future_residual=0.0,
    )


def _state(
    config: EpisodeConfig,
    inventory: int = 0,
    time_index: int = 0,
    regime: MarketRegime = MarketRegime.NORMAL,
) -> ControlState:
    return ControlState(
        time_index=time_index,
        time_remaining=config.n_steps - time_index,
        market_regime=regime,
        inventory=inventory,
        target_inventory=config.target_inventory,
        initial_inventory=config.initial_inventory,
        inventory_limit=config.inventory_limit,
        current_cp_plus=100.0,
        volatility=0.08,
        liquidity_score=0.55,
        market_width=0.12,
        active_execution_available=config.active_execution_allowed,
        current_rfq=None,
    )


def test_dynamic_reduces_to_myopic_without_penalties(control_artifacts) -> None:
    # No running penalty, no terminal urgency, no active execution, and a
    # limit far from reach: the continuation delta is exactly zero, so the
    # dynamic quote must coincide with the myopic edge-consistent quote.
    flat_config = EpisodeConfig(
        name="flat",
        n_steps=3,
        initial_inventory=0,
        target_inventory=0,
        inventory_limit=20,
        running_penalty_cents=0.0,
        terminal_penalty_cents=0.0,
        myopic_inventory_penalty_cents=0.0,
        active_execution_allowed=False,
    )
    solution = solve_bellman(
        flat_config, control_artifacts.market_config, control_artifacts.fitted_models
    )
    dynamic = DynamicExecutionController(
        control_artifacts.market_config, control_artifacts.fitted_models, solution
    )
    myopic = EdgeConsistentMyopicResponder(
        control_artifacts.market_config, flat_config, control_artifacts.fitted_models
    )
    state = _state(flat_config)
    for regime in MarketRegime:
        for side_sign in (1, -1):
            for size in (1, 2):
                event = _event(side_sign, size=size, regime=regime)
                dynamic_action = dynamic.respond_to_rfq(state, event)
                myopic_action = myopic.respond_to_rfq(state, event)
                assert dynamic_action.action_type == myopic_action.action_type
                if dynamic_action.action_type is ActionType.QUOTE_RFQ:
                    assert dynamic_action.normalized_aggressiveness == pytest.approx(
                        myopic_action.normalized_aggressiveness
                    )


def test_helpful_rfq_gets_no_less_aggressive_quote(control_artifacts) -> None:
    config = liquidation_episode()
    solution = solve_bellman(
        config, control_artifacts.market_config, control_artifacts.fitted_models
    )
    dynamic = DynamicExecutionController(
        control_artifacts.market_config, control_artifacts.fitted_models, solution
    )
    event = _event(side_sign=-1, size=1)
    long_action = dynamic.respond_to_rfq(_state(config, inventory=8), event)
    flat_action = dynamic.respond_to_rfq(_state(config, inventory=0), event)
    assert long_action.action_type is ActionType.QUOTE_RFQ
    if flat_action.action_type is ActionType.QUOTE_RFQ:
        assert (
            long_action.normalized_aggressiveness
            >= flat_action.normalized_aggressiveness
        )


def test_harmful_rfq_gets_no_more_aggressive_quote(control_artifacts) -> None:
    config = liquidation_episode()
    solution = solve_bellman(
        config, control_artifacts.market_config, control_artifacts.fitted_models
    )
    dynamic = DynamicExecutionController(
        control_artifacts.market_config, control_artifacts.fitted_models, solution
    )
    event = _event(side_sign=1, size=1)
    long_action = dynamic.respond_to_rfq(_state(config, inventory=8), event)
    flat_action = dynamic.respond_to_rfq(_state(config, inventory=0), event)
    if long_action.action_type is ActionType.QUOTE_RFQ:
        assert flat_action.action_type is ActionType.QUOTE_RFQ
        assert (
            long_action.normalized_aggressiveness
            <= flat_action.normalized_aggressiveness
        )


def test_all_policies_decline_when_costs_swamp_every_increment(control_artifacts) -> None:
    costly_market = dataclasses.replace(
        control_artifacts.market_config, rfq_transaction_cost_cents=500.0
    )
    config = with_overrides(liquidation_episode(), terminal_penalty_cents=0.0,
                            running_penalty_cents=0.0)
    solution = solve_bellman(
        config, costly_market, control_artifacts.fitted_models
    )
    plain = PlainResponder(costly_market, control_artifacts.fitted_models)
    dynamic = DynamicExecutionController(
        costly_market, control_artifacts.fitted_models, solution
    )
    state = _state(config)
    event = _event(side_sign=1)
    assert plain.respond_to_rfq(state, event).action_type is ActionType.DECLINE_RFQ
    assert dynamic.respond_to_rfq(state, event).action_type is ActionType.DECLINE_RFQ


def test_dynamic_accepts_negative_standalone_edge_to_cut_shortfall(
    control_artifacts,
) -> None:
    config = liquidation_episode()
    solution = solve_bellman(
        config, control_artifacts.market_config, control_artifacts.fitted_models
    )
    dynamic = DynamicExecutionController(
        control_artifacts.market_config, control_artifacts.fitted_models, solution
    )
    state = _state(config, inventory=8, time_index=config.n_steps - 1)
    event = _event(side_sign=-1, size=2)
    action = dynamic.respond_to_rfq(state, event)
    assert action.action_type is ActionType.QUOTE_RFQ
    # Standalone selection-adjusted edge at the chosen quote is negative;
    # the trade is taken for its continuation value toward the target.
    z = action.normalized_aggressiveness
    assert z is not None
    standalone = float(event.size) * (
        float(event.side_sign)
        * (action.predicted_post_win_value - action.quote)
        * 100.0
        - control_artifacts.market_config.rfq_transaction_cost_cents
    )
    assert standalone < 0.0


def test_oracle_controller_rejects_fitted_models(control_artifacts) -> None:
    config = liquidation_episode()
    solution = solve_bellman(
        config, control_artifacts.market_config, control_artifacts.oracle_models
    )
    with pytest.raises(TypeError, match="oracle models"):
        OracleDynamicController(
            control_artifacts.market_config,
            control_artifacts.fitted_models,  # type: ignore[arg-type]
            solution,
        )


def test_hidden_signal_moves_oracle_but_not_fitted_policies(control_artifacts) -> None:
    config = liquidation_episode()
    solutions = solve_episode_policies(control_artifacts, config)
    policies = make_policies(control_artifacts, config, solutions)
    state = _state(config, inventory=4)
    neutral = _event(side_sign=1, hidden_client_signal=0.0)
    informed = _event(side_sign=1, hidden_client_signal=2.5)

    for name in ("PlainResponder", "EdgeConsistentMyopic", "DynamicMarketMaker",
                 "DynamicExecution"):
        first = policies[name].respond_to_rfq(state, neutral)
        second = policies[name].respond_to_rfq(state, informed)
        assert first.action_type == second.action_type
        assert first.normalized_aggressiveness == second.normalized_aggressiveness

    oracle_first = policies["OracleDynamic"].respond_to_rfq(state, neutral)
    oracle_second = policies["OracleDynamic"].respond_to_rfq(state, informed)
    changed = (
        oracle_first.action_type != oracle_second.action_type
        or oracle_first.normalized_aggressiveness
        != oracle_second.normalized_aggressiveness
    )
    assert changed


def test_plain_and_edge_consistent_differ_under_adverse_selection(
    control_artifacts,
) -> None:
    config = liquidation_episode()
    plain = PlainResponder(
        control_artifacts.market_config, control_artifacts.fitted_models
    )
    myopic = EdgeConsistentMyopicResponder(
        control_artifacts.market_config, config, control_artifacts.fitted_models
    )
    state = _state(config, inventory=0)
    differing = 0
    for regime in MarketRegime:
        for side_sign in (1, -1):
            event = _event(side_sign, regime=regime, market_width=0.25)
            plain_action = plain.respond_to_rfq(state, event)
            myopic_action = myopic.respond_to_rfq(state, event)
            if (
                plain_action.action_type != myopic_action.action_type
                or plain_action.normalized_aggressiveness
                != myopic_action.normalized_aggressiveness
            ):
                differing += 1
    assert differing > 0


def test_higher_eta_reduces_terminal_shortfall(control_artifacts) -> None:
    def _mean_shortfall(terminal_penalty: float) -> float:
        config = with_overrides(
            liquidation_episode(), terminal_penalty_cents=terminal_penalty
        )
        solution = solve_bellman(
            config, control_artifacts.market_config, control_artifacts.fitted_models
        )
        policy = DynamicExecutionController(
            control_artifacts.market_config, control_artifacts.fitted_models, solution
        )
        shortfalls = []
        for seed in range(15):
            path = simulate_exogenous_path(
                control_artifacts.market_config, config, random_state=seed
            )
            result = simulate_episode(
                policy, config, control_artifacts.market_config, path
            )
            shortfalls.append(abs(result.terminal_inventory - config.target_inventory))
        return float(np.mean(shortfalls))

    assert _mean_shortfall(300.0) < _mean_shortfall(2.0)


def test_policies_respect_inventory_limits_in_simulation(control_artifacts) -> None:
    config = liquidation_episode()
    solutions = solve_episode_policies(control_artifacts, config)
    policies = make_policies(control_artifacts, config, solutions)
    for name, policy in policies.items():
        for seed in (0, 1, 2):
            path = simulate_exogenous_path(
                control_artifacts.market_config, config, random_state=seed
            )
            result = simulate_episode(
                policy, config, control_artifacts.market_config, path
            )
            assert (
                result.log["inventory_after"].abs().max() <= config.inventory_limit
            ), name
