"""Public behavior of the reward assembly functions."""

from __future__ import annotations

import numpy as np
import pytest

from rfq_edge.dynamic_objective import (
    expected_rfq_reward_cents,
    realized_rfq_reward_cents,
    rfq_increment_cents,
)


def test_dealer_buy_below_post_win_value_earns_positive_edge() -> None:
    # Dealer buys at 99.90 a bond worth 100.00 post win: +10 cents, minus cost.
    reward = expected_rfq_reward_cents(
        post_win_value=100.0,
        quote=99.90,
        side_sign=1,
        size=1,
        transaction_cost_cents=1.0,
    )
    assert reward == pytest.approx(9.0)


def test_dealer_sell_above_post_win_value_earns_positive_edge() -> None:
    reward = expected_rfq_reward_cents(
        post_win_value=100.0,
        quote=100.10,
        side_sign=-1,
        size=1,
        transaction_cost_cents=1.0,
    )
    assert reward == pytest.approx(9.0)


def test_reward_scales_linearly_with_size() -> None:
    small = expected_rfq_reward_cents(100.0, 99.95, 1, 1, 1.0)
    large = expected_rfq_reward_cents(100.0, 99.95, 1, 3, 1.0)
    assert large == pytest.approx(3.0 * small)


def test_expected_reward_is_vectorized_over_candidates() -> None:
    quotes = np.array([99.90, 100.00, 100.10])
    rewards = expected_rfq_reward_cents(100.0, quotes, 1, 1, 0.0)
    np.testing.assert_allclose(rewards, [10.0, 0.0, -10.0])


def test_realized_reward_uses_realized_future_value() -> None:
    # Dealer buys at 99.95; the bond ends at 100.00 - 0.20 = 99.80.
    reward = realized_rfq_reward_cents(
        quote=99.95,
        cp_plus=100.0,
        future_residual=-0.20,
        side_sign=1,
        size=2,
        transaction_cost_cents=1.0,
    )
    assert reward == pytest.approx(2.0 * (-15.0 - 1.0))


def test_rfq_increment_matches_definition() -> None:
    increment = rfq_increment_cents(
        fill_probability=0.4,
        trade_reward_cents=5.0,
        continuation_after_fill=-2.0,
        continuation_without_fill=-8.0,
    )
    assert increment == pytest.approx(0.4 * (5.0 + 6.0))


def test_rfq_increment_can_be_positive_with_negative_standalone_reward() -> None:
    # A trade that loses 4 cents standalone but frees 10 cents of
    # continuation value is worth taking.
    increment = rfq_increment_cents(
        fill_probability=0.5,
        trade_reward_cents=-4.0,
        continuation_after_fill=0.0,
        continuation_without_fill=-10.0,
    )
    assert increment == pytest.approx(3.0)
