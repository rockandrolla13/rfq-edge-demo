"""Trading, inventory, and hedge-risk costs for an RFQ fill."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rfq_edge.synthetic import RfqRequest, Side


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

    Inventory cost is charged only when the fill would increase absolute
    inventory. Reducing a preexisting position does not earn a rebate.

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


def _inventory_increase_cost_bps(request: RfqRequest, params: CostParams) -> float:
    inventory_delta = _inventory_delta(request)
    projected_inventory = request.inventory + inventory_delta
    current_abs = abs(request.inventory)
    projected_abs = abs(projected_inventory)
    if projected_abs <= current_abs:
        return 0.0
    increase = projected_abs - current_abs
    return params.inventory_bps_per_unit * increase


def _inventory_delta(request: RfqRequest) -> float:
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
