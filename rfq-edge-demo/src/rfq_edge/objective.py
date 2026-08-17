"""Expected-edge objective for a candidate RFQ quote."""

from __future__ import annotations

from dataclasses import dataclass

from rfq_edge.costs import CostParams, trading_cost_bps
from rfq_edge.fill_model import FillModelParams, fill_probability
from rfq_edge.selection_model import SelectionModelParams, adverse_selection_bps
from rfq_edge.synthetic import RfqRequest
from rfq_edge.value_model import ValueModelParams


BPS_SCALE = 10_000.0


@dataclass(frozen=True)
class ResponderModels:
    """Calibrated models and the target net edge for a consistent quote.

    :param value: Reservation-value parameters.
    :param fill: Fill-probability parameters.
    :param selection: Adverse-selection parameters.
    :param costs: Trading-cost parameters.
    :param target_edge_bps: Net edge the consistent quote must earn if filled.
    """

    value: ValueModelParams
    fill: FillModelParams
    selection: SelectionModelParams
    costs: CostParams
    target_edge_bps: float


@dataclass(frozen=True)
class EdgeComponents:
    """Observable decomposition of a candidate quote.

    :param quoted_edge_bps: Dealer markup in basis points of mid.
    :param fill_probability: Probability the quote is hit.
    :param selection_bps: Expected adverse selection if filled.
    :param cost_bps: Inventory, hedge, and transaction costs if filled.
    :param required_edge_bps: Selection plus costs plus target edge.
    :param net_edge_bps: Quoted edge minus selection and costs.
    :param expected_pnl: Fill probability times dollar net edge.
    """

    quoted_edge_bps: float
    fill_probability: float
    selection_bps: float
    cost_bps: float
    required_edge_bps: float
    net_edge_bps: float
    expected_pnl: float


def evaluate_quote(
    request: RfqRequest,
    models: ResponderModels,
    edge_bps: float,
) -> EdgeComponents:
    """Score one quoted edge against fill, selection, and cost models.

    :param request: Incoming RFQ.
    :param models: Calibrated responder models.
    :param edge_bps: Candidate dealer markup in basis points of mid.
    :return: Component breakdown and expected PnL.
    :raises ValueError: If the target edge is negative.
    """

    if models.target_edge_bps < 0.0:
        raise ValueError("target_edge_bps must be non-negative")
    fill = fill_probability(edge_bps, request, models.fill)
    selection = adverse_selection_bps(edge_bps, request, models.selection)
    cost = trading_cost_bps(request, models.costs)
    required = selection + cost + models.target_edge_bps
    net_edge = edge_bps - selection - cost
    expected_pnl = fill * _bps_to_dollars(net_edge, request)
    return EdgeComponents(
        quoted_edge_bps=edge_bps,
        fill_probability=fill,
        selection_bps=selection,
        cost_bps=cost,
        required_edge_bps=required,
        net_edge_bps=net_edge,
        expected_pnl=expected_pnl,
    )


def _bps_to_dollars(edge_bps: float, request: RfqRequest) -> float:
    return edge_bps / BPS_SCALE * request.mid_price * request.quantity
