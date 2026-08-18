"""Cost and penalty functions for the control layer, in cents.

Each reward component appears in exactly one function so that reward
accounting can be reconciled line by line:

* RFQ trade reward — :mod:`rfq_edge.dynamic_objective`;
* active execution cost — :func:`active_execution_cost_cents`;
* running inventory penalty — :func:`running_inventory_penalty_cents`;
* terminal penalty — :func:`terminal_penalty_cents`.
"""

from __future__ import annotations

from rfq_edge.control_config import EpisodeConfig, RegimeParameters


def active_execution_cost_cents(amount: int, regime_params: RegimeParameters) -> float:
    """Cost of an active execution of ``amount`` units, in cents.

    C_active(u) = half_spread * |u| + impact * u^2 + fixed_fee * 1{u != 0}.
    Stressed regimes carry wider spreads and greater impact through their
    parameters. The cost is zero when u = 0 and strictly positive otherwise.

    :param amount: Signed active execution in units (u > 0 buys).
    :param regime_params: Parameters of the current regime.
    :return: Non-negative cost in cents.
    """

    if amount == 0:
        return 0.0
    absolute = abs(float(amount))
    spread_cost = regime_params.active_half_spread_cents * absolute
    impact_cost = regime_params.active_impact_cents * absolute * absolute
    return spread_cost + impact_cost + regime_params.active_fixed_fee_cents


def running_inventory_penalty_cents(
    inventory: int,
    config: EpisodeConfig,
) -> float:
    """Per-step inventory penalty phi * (I - I_target)^2, in cents.

    Applied once per step on end-of-step inventory (delta_t = 1 step).

    :param inventory: End-of-step inventory in units.
    :param config: Episode configuration providing phi and the target.
    :return: Non-negative penalty in cents.
    """

    deviation = float(inventory - config.target_inventory)
    return config.running_penalty_cents * deviation * deviation


def terminal_penalty_cents(inventory: int, config: EpisodeConfig) -> float:
    """Terminal shortfall penalty eta * (I_T - I_target)^2, in cents.

    Zero exactly at target and quadratic in the shortfall.

    :param inventory: Inventory at the horizon in units.
    :param config: Episode configuration providing eta and the target.
    :return: Non-negative penalty in cents.
    """

    deviation = float(inventory - config.target_inventory)
    return config.terminal_penalty_cents * deviation * deviation
