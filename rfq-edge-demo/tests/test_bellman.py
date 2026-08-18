"""Public behavior of the Bellman solver on the control grid."""

from __future__ import annotations

import numpy as np
import pytest

from rfq_edge.bellman import (
    RESIDUAL_TOLERANCE_CENTS,
    bellman_residual,
    fill_continuation_delta,
    inventory_shadow_value,
    solve_bellman,
)
from rfq_edge.control_config import (
    MarketRegime,
    liquidation_episode,
    market_making_episode,
    with_overrides,
)

DEALER_BUY_INDEX = 0
DEALER_SELL_INDEX = 1


@pytest.fixture(scope="module")
def mm_solution(control_artifacts):
    return solve_bellman(
        market_making_episode(),
        control_artifacts.market_config,
        control_artifacts.fitted_models,
    )


@pytest.fixture(scope="module")
def liquidation_solution(control_artifacts):
    return solve_bellman(
        liquidation_episode(),
        control_artifacts.market_config,
        control_artifacts.fitted_models,
    )


def test_terminal_condition_is_quadratic_shortfall_penalty(liquidation_solution) -> None:
    config = liquidation_solution.episode_config
    grid = liquidation_solution.inventory_grid
    expected = -config.terminal_penalty_cents * (
        grid.astype(float) - config.target_inventory
    ) ** 2
    for regime_index in range(3):
        np.testing.assert_allclose(
            liquidation_solution.value[config.n_steps, :, regime_index], expected
        )


def test_value_function_is_finite_everywhere(mm_solution, liquidation_solution) -> None:
    assert np.isfinite(mm_solution.value).all()
    assert np.isfinite(mm_solution.post_rfq_value).all()
    assert np.isfinite(liquidation_solution.value).all()


def test_bellman_residual_is_within_tolerance(mm_solution, liquidation_solution) -> None:
    for solution in (mm_solution, liquidation_solution):
        stats = bellman_residual(solution)
        assert stats["max_abs_residual_cents"] <= RESIDUAL_TOLERANCE_CENTS
        assert stats["n_states_violating_tolerance"] == 0


def test_shadow_value_signs_around_target(mm_solution) -> None:
    # With target zero, extra long inventory destroys value and extra short
    # inventory below target gains from moving up.
    regime = MarketRegime.NORMAL.value
    assert inventory_shadow_value(mm_solution, 0, 5, regime) < 0.0
    assert inventory_shadow_value(mm_solution, 0, -5, regime) > 0.0


def test_fill_continuation_delta_matches_post_rfq_values(liquidation_solution) -> None:
    step, inventory, regime = 10, 8, MarketRegime.NORMAL.value
    delta = fill_continuation_delta(
        liquidation_solution, step, inventory, side_sign=-1, size=2, regime_index=regime
    )
    values = liquidation_solution.post_rfq_value[step, :, regime]
    manual = (
        values[liquidation_solution.inventory_index(6)]
        - values[liquidation_solution.inventory_index(8)]
    )
    assert delta == pytest.approx(manual)
    # Selling toward the target from a long position gains continuation value.
    assert delta > 0.0


def test_active_execution_is_zero_at_target_without_alpha(mm_solution) -> None:
    config = mm_solution.episode_config
    at_target = mm_solution.inventory_index(config.target_inventory)
    assert (mm_solution.active_policy[:, at_target, :] == 0).all()


def test_execution_urgency_increases_near_deadline(liquidation_solution) -> None:
    regime = MarketRegime.NORMAL.value
    long_index = liquidation_solution.inventory_index(8)
    early = int(liquidation_solution.active_policy[0, long_index, regime])
    late = int(
        liquidation_solution.active_policy[
            liquidation_solution.episode_config.n_steps - 1, long_index, regime
        ]
    )
    assert late < 0
    assert abs(late) >= abs(early)


def test_quotes_that_would_breach_the_limit_are_declined(mm_solution) -> None:
    limit = mm_solution.episode_config.inventory_limit
    at_limit = mm_solution.inventory_index(limit)
    # Any dealer-buy fill at +limit would breach; policy must decline.
    assert (
        mm_solution.quote_policy_z_index[:, at_limit, :, DEALER_BUY_INDEX, :] == -1
    ).all()


def test_helpful_side_is_quoted_no_less_aggressively(liquidation_solution) -> None:
    # Long +8 versus target 0 in the middle of the episode: dealer-sell RFQs
    # (which reduce the position) should be quoted at least as aggressively
    # as under neutral inventory, dealer-buy RFQs at most as aggressively.
    step = liquidation_solution.episode_config.n_steps // 2
    regime = MarketRegime.NORMAL.value
    size_position = liquidation_solution.size_index(1)
    long_index = liquidation_solution.inventory_index(8)
    flat_index = liquidation_solution.inventory_index(0)
    policy = liquidation_solution.quote_policy_z_index

    sell_long = policy[step, long_index, regime, DEALER_SELL_INDEX, size_position]
    sell_flat = policy[step, flat_index, regime, DEALER_SELL_INDEX, size_position]
    assert sell_long >= sell_flat

    buy_long = policy[step, long_index, regime, DEALER_BUY_INDEX, size_position]
    buy_flat = policy[step, flat_index, regime, DEALER_BUY_INDEX, size_position]
    assert buy_long <= buy_flat


def test_higher_terminal_penalty_raises_urgency(control_artifacts) -> None:
    lax = solve_bellman(
        with_overrides(liquidation_episode(), terminal_penalty_cents=2.0),
        control_artifacts.market_config,
        control_artifacts.fitted_models,
    )
    strict = solve_bellman(
        with_overrides(liquidation_episode(), terminal_penalty_cents=300.0),
        control_artifacts.market_config,
        control_artifacts.fitted_models,
    )
    regime = MarketRegime.NORMAL.value
    long_index = lax.inventory_index(8)
    final = lax.episode_config.n_steps - 1
    lax_sales = int(lax.active_policy[final, long_index, regime])
    strict_sales = int(strict.active_policy[final, long_index, regime])
    assert strict_sales <= lax_sales
    assert strict_sales < 0
