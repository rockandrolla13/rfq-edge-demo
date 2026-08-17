"""Adverse-selection model conditional on winning the RFQ."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rfq_edge.synthetic import RfqRequest


@dataclass(frozen=True)
class SelectionModelParams:
    """Winner's-curse calibration in basis points of mid.

    Tighter quotes (lower edge) attract more informed flow, so selection
    decays as the quoted edge widens.

    :param base_bps: Floor selection paid even on defensive quotes.
    :param scale: Strength of the tightness-driven selection term.
    :param decay: Positive rate at which selection fades as edge rises.
    """

    base_bps: float
    scale: float
    decay: float


def adverse_selection_bps(
    edge_bps: float,
    request: RfqRequest,
    params: SelectionModelParams,
) -> float:
    """Return expected adverse selection in basis points if the quote fills.

    :param edge_bps: Quoted dealer markup in basis points of mid.
    :param request: Incoming RFQ.
    :param params: Selection calibration.
    :return: Non-negative selection cost in basis points.
    :raises ValueError: If inputs or parameters are invalid.
    """

    _validate_selection_inputs(edge_bps, params)
    tightness = math.exp(-params.decay * edge_bps)
    size_factor = math.sqrt(request.quantity)
    informed_component = params.scale * tightness * request.volatility * size_factor
    return params.base_bps + informed_component


def _validate_selection_inputs(edge_bps: float, params: SelectionModelParams) -> None:
    if edge_bps < 0.0:
        raise ValueError("edge_bps must be non-negative")
    if params.base_bps < 0.0:
        raise ValueError("base_bps must be non-negative")
    if params.scale < 0.0:
        raise ValueError("scale must be non-negative")
    if params.decay <= 0.0:
        raise ValueError("decay must be positive")
