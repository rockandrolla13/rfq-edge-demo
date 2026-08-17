"""Trading, inventory, and hedge-risk costs for an RFQ fill."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from rfq_edge.config import OptimizerConfig
from rfq_edge.synthetic import RfqRequest, Side

PRICE_POINTS_TO_CENTS = 100.0


@dataclass(frozen=True)
class CostParams:
    """Cost components expressed in basis points of mid.

    :param transaction_bps: Hedge and operational cost paid on every fill.
    :param inventory_bps_per_unit: Extra bps per unit of inventory increase.
    :param risk_aversion: Scales volatility and hedge-horizon risk cost.
    """

    transaction_bps: float
    inventory_bps_per_unit: float
    risk_aversion: float


def trading_cost_bps(request: RfqRequest, params: CostParams) -> float:
    """Return fill costs that do not depend on the quoted edge.

    :param request: Incoming RFQ.
    :param params: Cost calibration.
    :return: Non-negative cost in basis points of mid.
    :raises ValueError: If parameters are invalid.
    """

    _validate_cost_params(params)
    transaction = params.transaction_bps
    inventory = _inventory_increase_cost_bps(request, params)
    risk = params.risk_aversion * request.volatility * math.sqrt(request.time_to_hedge)
    return transaction + inventory + risk


def rfq_inventory_delta(side: str, size: float) -> float:
    """Return the dealer inventory change if an RFQ fills.

    :param side: Dealer side label.
    :param size: RFQ size in notional units.
    :return: Signed inventory change.
    :raises ValueError: If the side is unknown.
    """

    if side == "dealer_buy":
        return size
    if side == "dealer_sell":
        return -size
    raise ValueError(f"unsupported side: {side}")


def rfq_trading_cost_points(row: pd.Series, config: OptimizerConfig) -> float:
    """Return quote-independent trading cost in clean price points.

    :param row: Single RFQ row with ``cp_plus`` and ``volatility``.
    :param config: Optimizer configuration.
    :return: Non-negative trading cost in price points.
    """

    mid = float(row["cp_plus"])
    transaction = config.transaction_bps / 10_000.0 * mid
    risk = (
        config.risk_aversion
        * float(row["volatility"])
        * math.sqrt(config.hedge_horizon_years)
        / 10_000.0
        * mid
    )
    return transaction + risk


def rfq_inventory_value_points(row: pd.Series, config: OptimizerConfig) -> float:
    """Return inventory preference value in clean price points.

    Reducing absolute inventory earns a positive value; increasing it applies
    a penalty. This term is separate from quote-independent trading cost.

    :param row: Single RFQ row with ``side``, ``size``, and ``inventory``.
    :param config: Optimizer configuration.
    :return: Signed inventory value in price points.
    """

    inventory_delta = rfq_inventory_delta(str(row["side"]), float(row["size"]))
    projected_inventory = float(row["inventory"]) + inventory_delta
    current_abs = abs(float(row["inventory"]))
    projected_abs = abs(projected_inventory)
    if projected_abs < current_abs:
        reduction = current_abs - projected_abs
        return reduction * config.inventory_value_per_unit
    if projected_abs > current_abs:
        increase = projected_abs - current_abs
        return -increase * config.inventory_penalty_per_unit
    return 0.0


def points_to_cents(value_points: float) -> float:
    """Convert clean price points to cents of par.

    :param value_points: Value expressed in price points.
    :return: Value expressed in cents.
    """

    return value_points * PRICE_POINTS_TO_CENTS


def _inventory_increase_cost_bps(request: RfqRequest, params: CostParams) -> float:
    inventory_delta = _legacy_inventory_delta(request)
    projected_inventory = request.inventory + inventory_delta
    current_abs = abs(request.inventory)
    projected_abs = abs(projected_inventory)
    if projected_abs <= current_abs:
        return 0.0
    increase = projected_abs - current_abs
    return params.inventory_bps_per_unit * increase


def _legacy_inventory_delta(request: RfqRequest) -> float:
    if request.side is Side.BUY:
        return -request.quantity
    if request.side is Side.SELL:
        return request.quantity
    raise ValueError(f"unsupported side: {request.side}")


def _validate_cost_params(params: CostParams) -> None:
    if params.transaction_bps < 0.0:
        raise ValueError("transaction_bps must be non-negative")
    if params.inventory_bps_per_unit < 0.0:
        raise ValueError("inventory_bps_per_unit must be non-negative")
    if params.risk_aversion < 0.0:
        raise ValueError("risk_aversion must be non-negative")
