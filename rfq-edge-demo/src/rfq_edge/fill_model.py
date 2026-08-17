"""Fill-probability model for a quoted RFQ edge."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rfq_edge.synthetic import RfqRequest


@dataclass(frozen=True)
class FillModelParams:
    """Logistic fill curve in quoted edge, size, and competition.

    :param intercept: Logit at zero edge with no extra penalties.
    :param edge_coef: Positive coefficient. Larger quoted edge lowers fill odds.
    :param competition_coef: Positive coefficient on competing responders.
    :param size_coef: Positive coefficient on log quantity.
    """

    intercept: float
    edge_coef: float
    competition_coef: float
    size_coef: float


def fill_probability(
    edge_bps: float,
    request: RfqRequest,
    params: FillModelParams,
) -> float:
    """Return P(fill) for a candidate dealer edge.

    :param edge_bps: Quoted dealer markup in basis points of mid.
    :param request: Incoming RFQ.
    :param params: Logistic calibration.
    :return: Fill probability in (0, 1).
    :raises ValueError: If coefficients or inputs are invalid.
    """

    _validate_fill_inputs(edge_bps, params)
    logit = (
        params.intercept
        - params.edge_coef * edge_bps
        - params.competition_coef * float(request.competition_count)
        - params.size_coef * math.log(request.quantity)
    )
    return _sigmoid(logit)


def _validate_fill_inputs(edge_bps: float, params: FillModelParams) -> None:
    if edge_bps < 0.0:
        raise ValueError("edge_bps must be non-negative")
    if params.edge_coef <= 0.0:
        raise ValueError("edge_coef must be positive")
    if params.competition_coef < 0.0:
        raise ValueError("competition_coef must be non-negative")
    if params.size_coef < 0.0:
        raise ValueError("size_coef must be non-negative")


def _sigmoid(logit: float) -> float:
    if logit >= 0.0:
        exp_neg = math.exp(-logit)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(logit)
    return exp_pos / (1.0 + exp_pos)
