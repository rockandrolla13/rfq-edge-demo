"""Public behavior of the event simulator, exogenous paths, and models."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rfq_edge.control_config import (
    MarketRegime,
    default_control_market,
    liquidation_episode,
    market_making_episode,
    with_overrides,
)
from rfq_edge.control_models import fit_control_models
from rfq_edge.control_state import ActionType, ControlAction, ControlState, RFQEvent
from rfq_edge.event_simulator import simulate_episode
from rfq_edge.market_dynamics import (
    ExogenousPath,
    generate_training_history,
    simulate_exogenous_path,
    true_fill_probability,
)
from rfq_edge.oracle_control import OracleControlModels

MARKET = default_control_market()


class _FixedPolicy:
    """Quotes a fixed aggressiveness on every admissible RFQ; fixed active u."""

    def __init__(
        self,
        name: str = "fixed",
        aggressiveness: float | None = 0.0,
        active_amount: int = 0,
    ) -> None:
        self.name = name
        self.aggressiveness = aggressiveness
        self.active_amount = active_amount

    def respond_to_rfq(self, state: ControlState, event: RFQEvent) -> ControlAction:
        filled_inventory = state.inventory + event.side_sign * event.size
        if self.aggressiveness is None or abs(filled_inventory) > state.inventory_limit:
            return ControlAction(
                action_type=ActionType.DECLINE_RFQ,
                quote=None,
                normalized_aggressiveness=None,
                active_execution_amount=0,
                respond_or_decline=False,
                policy_name=self.name,
            )
        quote = event.cp_plus + event.side_sign * self.aggressiveness * event.market_width
        return ControlAction(
            action_type=ActionType.QUOTE_RFQ,
            quote=quote,
            normalized_aggressiveness=self.aggressiveness,
            active_execution_amount=0,
            respond_or_decline=True,
            policy_name=self.name,
            predicted_p_win=0.5,
            predicted_selection=0.0,
            predicted_post_win_value=event.cp_plus,
        )

    def choose_active_execution(self, state: ControlState) -> ControlAction:
        amount = self.active_amount
        if abs(state.inventory + amount) > state.inventory_limit:
            amount = 0
        if amount == 0:
            return ControlAction(
                action_type=ActionType.WAIT,
                quote=None,
                normalized_aggressiveness=None,
                active_execution_amount=0,
                respond_or_decline=False,
                policy_name=self.name,
            )
        return ControlAction(
            action_type=ActionType.ACTIVE_EXECUTION,
            quote=None,
            normalized_aggressiveness=None,
            active_execution_amount=amount,
            respond_or_decline=False,
            policy_name=self.name,
        )


def _manual_event(step: int, side_sign: int, size: int = 2) -> RFQEvent:
    return RFQEvent(
        event_id=step,
        time_index=step,
        side="dealer_buy" if side_sign == 1 else "dealer_sell",
        side_sign=side_sign,
        size=size,
        client_tier="professional",
        liquidity_score=0.55,
        market_width=0.12,
        regime=MarketRegime.NORMAL,
        cp_plus=100.0,
        hidden_client_signal=0.0,
        hidden_future_residual=0.05,
    )


def _manual_path(events: list[RFQEvent | None], always_fill: bool) -> ExogenousPath:
    n_steps = len(events)
    uniform = 0.0 if always_fill else 1.0
    return ExogenousPath(
        regimes=np.full(n_steps, MarketRegime.NORMAL.value, dtype=int),
        cp_plus=np.full(n_steps, 100.0),
        events=tuple(events),
        fill_uniforms=np.full(n_steps, uniform),
    )


def _short_episode(n_steps: int, **overrides: object):
    base = with_overrides(market_making_episode(), n_steps=n_steps)
    if overrides:
        base = with_overrides(base, **overrides)
    return base


def test_dealer_buy_fill_increases_inventory_in_simulation() -> None:
    path = _manual_path([_manual_event(0, side_sign=1)], always_fill=True)
    result = simulate_episode(_FixedPolicy(), _short_episode(1), MARKET, path)
    row = result.log.iloc[0]
    assert row["filled"]
    assert row["inventory_after_rfq"] == row["inventory_before"] + 2


def test_dealer_sell_fill_decreases_inventory_in_simulation() -> None:
    path = _manual_path([_manual_event(0, side_sign=-1)], always_fill=True)
    result = simulate_episode(_FixedPolicy(), _short_episode(1), MARKET, path)
    row = result.log.iloc[0]
    assert row["filled"]
    assert row["inventory_after_rfq"] == row["inventory_before"] - 2


def test_active_buy_and_sell_move_inventory() -> None:
    path = _manual_path([None, None], always_fill=False)
    buy = simulate_episode(
        _FixedPolicy(active_amount=1), _short_episode(2), MARKET, path
    )
    sell = simulate_episode(
        _FixedPolicy(active_amount=-1), _short_episode(2), MARKET, path
    )
    assert buy.terminal_inventory == 2
    assert sell.terminal_inventory == -2
    assert (buy.log["active_execution_cost_cents"] > 0.0).all()


def test_helpful_rfq_reduces_target_shortfall() -> None:
    # Liquidation: long 8 to target 0; a dealer-sell fill helps.
    config = with_overrides(liquidation_episode(), n_steps=1)
    path = _manual_path([_manual_event(0, side_sign=-1)], always_fill=True)
    result = simulate_episode(_FixedPolicy(), config, MARKET, path)
    row = result.log.iloc[0]
    assert row["remaining_target_shortfall"] == 6
    assert row["economic_mode"] == "passive_execution"


def test_harmful_rfq_increases_target_shortfall() -> None:
    config = with_overrides(liquidation_episode(), n_steps=1)
    path = _manual_path([_manual_event(0, side_sign=1)], always_fill=True)
    result = simulate_episode(_FixedPolicy(), config, MARKET, path)
    assert result.log.iloc[0]["remaining_target_shortfall"] == 10


def test_costs_and_penalties_in_log_are_non_negative() -> None:
    path = simulate_exogenous_path(MARKET, market_making_episode(), random_state=7)
    result = simulate_episode(
        _FixedPolicy(active_amount=1), market_making_episode(), MARKET, path
    )
    assert (result.log["active_execution_cost_cents"] >= 0.0).all()
    assert (result.log["running_inventory_penalty_cents"] >= 0.0).all()
    assert result.terminal_penalty_cents >= 0.0


def test_inventory_limit_violation_is_rejected() -> None:
    config = _short_episode(1, inventory_limit=1)
    path = _manual_path([None], always_fill=False)

    class _Violator(_FixedPolicy):
        def choose_active_execution(self, state: ControlState) -> ControlAction:
            return ControlAction(
                action_type=ActionType.ACTIVE_EXECUTION,
                quote=None,
                normalized_aggressiveness=None,
                active_execution_amount=2,
                respond_or_decline=False,
                policy_name=self.name,
            )

    with pytest.raises(ValueError, match="limit"):
        simulate_episode(_Violator(), config, MARKET, path)


def test_quoting_beyond_limit_is_rejected() -> None:
    config = _short_episode(1, inventory_limit=1)
    path = _manual_path([_manual_event(0, side_sign=1, size=2)], always_fill=True)

    class _BlindQuoter(_FixedPolicy):
        def respond_to_rfq(self, state: ControlState, event: RFQEvent) -> ControlAction:
            return ControlAction(
                action_type=ActionType.QUOTE_RFQ,
                quote=event.cp_plus,
                normalized_aggressiveness=0.0,
                active_execution_amount=0,
                respond_or_decline=True,
                policy_name=self.name,
            )

    with pytest.raises(ValueError, match="limit"):
        simulate_episode(_BlindQuoter(), config, MARKET, path)


def test_reward_components_reconcile_with_total() -> None:
    path = simulate_exogenous_path(MARKET, market_making_episode(), random_state=11)
    result = simulate_episode(_FixedPolicy(), market_making_episode(), MARKET, path)
    log = result.log
    reconstructed = (
        log["rfq_reward_cents"].sum()
        - log["active_execution_cost_cents"].sum()
        - log["running_inventory_penalty_cents"].sum()
        - log["terminal_penalty_cents"].sum()
    )
    assert result.total_reward_cents == pytest.approx(reconstructed)
    assert log["cumulative_reward_cents"].iloc[-1] == pytest.approx(reconstructed)


def test_identical_seeds_produce_identical_paths() -> None:
    first = simulate_exogenous_path(MARKET, market_making_episode(), random_state=42)
    second = simulate_exogenous_path(MARKET, market_making_episode(), random_state=42)
    np.testing.assert_array_equal(first.regimes, second.regimes)
    np.testing.assert_array_equal(first.cp_plus, second.cp_plus)
    np.testing.assert_array_equal(first.fill_uniforms, second.fill_uniforms)
    assert first.events == second.events


def test_common_random_numbers_expose_same_exogenous_path_to_each_policy() -> None:
    path = simulate_exogenous_path(MARKET, market_making_episode(), random_state=3)
    passive = simulate_episode(
        _FixedPolicy(name="passive", aggressiveness=-0.5),
        market_making_episode(),
        MARKET,
        path,
    )
    aggressive = simulate_episode(
        _FixedPolicy(name="aggressive", aggressiveness=1.0),
        market_making_episode(),
        MARKET,
        path,
    )
    shared = ["rfq_arrived", "rfq_side", "rfq_size", "regime"]
    assert passive.log[shared].equals(aggressive.log[shared])


def test_more_aggressive_quotes_fill_more_often_on_shared_paths() -> None:
    config = market_making_episode()
    fills_passive = 0
    fills_aggressive = 0
    for seed in range(20):
        path = simulate_exogenous_path(MARKET, config, random_state=seed)
        passive = simulate_episode(
            _FixedPolicy(name="p", aggressiveness=-0.75), config, MARKET, path
        )
        aggressive = simulate_episode(
            _FixedPolicy(name="a", aggressiveness=0.75), config, MARKET, path
        )
        fills_passive += int(passive.log["filled"].sum())
        fills_aggressive += int(aggressive.log["filled"].sum())
    assert fills_aggressive > fills_passive


def test_fitted_models_ignore_hidden_oracle_fields() -> None:
    history = generate_training_history(MARKET, n_events=4000, random_state=0)
    assert not any(column.startswith("hidden") for column in history.columns)
    fitted = fit_control_models(history)
    event = _manual_event(0, side_sign=1)
    perturbed = dataclasses.replace(
        event, hidden_client_signal=3.0, hidden_future_residual=-1.0
    )
    grid = np.asarray(MARKET.aggressiveness_grid)
    np.testing.assert_array_equal(
        fitted.event_fill_probability(event, grid),
        fitted.event_fill_probability(perturbed, grid),
    )
    np.testing.assert_array_equal(
        fitted.event_post_win_value(event, grid),
        fitted.event_post_win_value(perturbed, grid),
    )


def test_oracle_models_react_to_hidden_signal() -> None:
    oracle = OracleControlModels(MARKET)
    event = _manual_event(0, side_sign=1)
    informed = dataclasses.replace(event, hidden_client_signal=2.0)
    grid = np.asarray(MARKET.aggressiveness_grid)
    # A strongly positive client signal on a dealer buy lowers fill odds
    # (the client keeps the bond when future value is strong).
    assert (
        oracle.event_fill_probability(informed, grid)
        < oracle.event_fill_probability(event, grid)
    ).all()
    assert oracle.event_post_win_value(informed, grid)[0] > event.cp_plus


def test_fitted_models_recover_dgp_shape() -> None:
    history = generate_training_history(MARKET, n_events=12000, random_state=1)
    fitted = fit_control_models(history)
    oracle = OracleControlModels(MARKET)
    grid = np.linspace(-1.5, 1.5, 7)
    for regime_index in range(3):
        fitted_p = fitted.fill_probability(regime_index, grid, size=1)
        oracle_p = oracle.fill_probability(regime_index, grid, size=1)
        assert (np.diff(fitted_p) > 0.0).all()
        assert np.max(np.abs(fitted_p - oracle_p)) < 0.08
        fitted_a = fitted.selection_points(regime_index, grid)
        oracle_a = oracle.selection_points(regime_index, grid)
        assert (fitted_a > 0.0).all()
        assert (np.diff(fitted_a) < 0.0).all()
        assert np.max(np.abs(fitted_a - oracle_a)) < 0.05


def test_true_fill_probability_matches_event_mechanism() -> None:
    event = _manual_event(0, side_sign=1)
    oracle = OracleControlModels(MARKET)
    direct = true_fill_probability(MARKET, event, aggressiveness=0.5)
    via_oracle = oracle.event_fill_probability(event, np.asarray([0.5]))[0]
    assert direct == pytest.approx(via_oracle)
