"""Reward assembly shared by the Bellman solver, controllers, and simulator.

Reward components and where each appears exactly once per step:

* RFQ trade reward       r_rfq = n * [side_sign * (m - q) * 100 - cost_cents]
* active execution cost  r_active = -C_active(u)
* running penalty        r_inv = -phi * (I_end - I_target)^2
* terminal penalty       r_T = -eta * (I_T - I_target)^2

The t+5 conditional clean edge enters only through r_rfq. The continuation
value V represents inventory risk, target completion, future RFQ
opportunities, and future execution costs; it never re-counts the same
expected price move.
"""

from __future__ import annotations

import numpy as np

from rfq_edge.control_config import POINTS_TO_CENTS


def expected_rfq_reward_cents(
    post_win_value: np.ndarray | float,
    quote: np.ndarray | float,
    side_sign: int,
    size: int,
    transaction_cost_cents: float,
) -> np.ndarray | float:
    """Expected RFQ trade reward conditional on a fill, in cents.

    r_rfq(q, X, n) = n * [side_sign * (m(q, X) - q) * 100 - cost_cents].

    :param post_win_value: Post-win clean value m in points.
    :param quote: Clean quote q in points.
    :param side_sign: +1 dealer buy, -1 dealer sell.
    :param size: RFQ size n in units.
    :param transaction_cost_cents: RFQ transaction cost per unit, cents.
    :return: Reward per candidate, in cents.
    """

    edge_cents = float(side_sign) * (np.asarray(post_win_value) - np.asarray(quote)) * POINTS_TO_CENTS
    return float(size) * (edge_cents - transaction_cost_cents)


def realized_rfq_reward_cents(
    quote: float,
    cp_plus: float,
    future_residual: float,
    side_sign: int,
    size: int,
    transaction_cost_cents: float,
) -> float:
    """Realized RFQ trade reward after a fill, in cents.

    Uses the realized future clean value y5 = cp_plus + future_residual.

    :param quote: Clean quote in points.
    :param cp_plus: CP+ at the RFQ, in points.
    :param future_residual: Realized future residual in points.
    :param side_sign: +1 dealer buy, -1 dealer sell.
    :param size: RFQ size in units.
    :param transaction_cost_cents: RFQ transaction cost per unit, cents.
    :return: Realized reward in cents.
    """

    realized_value = cp_plus + future_residual
    edge_cents = float(side_sign) * (realized_value - quote) * POINTS_TO_CENTS
    return float(size) * (edge_cents - transaction_cost_cents)


def rfq_increment_cents(
    fill_probability: np.ndarray | float,
    trade_reward_cents: np.ndarray | float,
    continuation_after_fill: float,
    continuation_without_fill: float,
) -> np.ndarray | float:
    """The RFQ jump increment used by the Bellman recursion and controllers.

    RFQIncrement(q) = p(q, X) * [r_rfq + V_next(I_fill, r) - V_next(I, r)].

    The continuation difference is the dynamic inventory value; no separate
    inventory adjustment is added on top of it.

    :param fill_probability: p(q, X) per candidate.
    :param trade_reward_cents: r_rfq per candidate, in cents.
    :param continuation_after_fill: V_next at the post-fill inventory.
    :param continuation_without_fill: V_next at the unchanged inventory.
    :return: Increment per candidate, in cents.
    """

    continuation_delta = continuation_after_fill - continuation_without_fill
    return np.asarray(fill_probability) * (
        np.asarray(trade_reward_cents) + continuation_delta
    )
