"""Legacy quote search helpers used by the responder pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from rfq_edge.objective import EdgeComponents, ResponderModels, evaluate_quote
from rfq_edge.synthetic import RfqRequest


@dataclass(frozen=True)
class OptimizerParams:
    """One-dimensional search bounds over quoted edge.

    :param min_edge_bps: Inclusive lower bound. Must be non-negative.
    :param max_edge_bps: Inclusive upper bound. Must exceed the lower bound.
    :param step_bps: Grid step used by the expected-PnL maximizer.
    """

    min_edge_bps: float
    max_edge_bps: float
    step_bps: float


@dataclass(frozen=True)
class QuoteSolution:
    """A selected quote and its objective decomposition.

    :param rule: Name of the rule that produced the quote.
    :param components: Evaluated fill, selection, cost, and PnL terms.
    """

    rule: str
    components: EdgeComponents


def solve_consistent_edge(
    request: RfqRequest,
    models: ResponderModels,
    search: OptimizerParams,
) -> QuoteSolution:
    """Solve quoted_edge = selection + costs + target_edge.

    :param request: Incoming RFQ.
    :param models: Calibrated responder models.
    :param search: Bounds for the one-dimensional solve.
    :return: Edge-consistent quote and its components.
    :raises ValueError: If bounds are invalid or no root exists inside them.
    """

    _validate_search(search)
    edge_bps = _bisect_consistent_edge(request, models, search)
    components = evaluate_quote(request, models, edge_bps)
    return QuoteSolution(rule="edge_consistent", components=components)


def maximize_expected_pnl(
    request: RfqRequest,
    models: ResponderModels,
    search: OptimizerParams,
) -> QuoteSolution:
    """Pick the grid edge with the highest expected PnL.

    :param request: Incoming RFQ.
    :param models: Calibrated responder models.
    :param search: Bounds for the one-dimensional solve.
    :return: Expected-PnL maximizing quote and its components.
    :raises ValueError: If the search grid is empty or invalid.
    """

    _validate_search(search)
    best_components = _scan_expected_pnl_grid(request, models, search)
    return QuoteSolution(rule="expected_pnl", components=best_components)


def _validate_search(search: OptimizerParams) -> None:
    if search.min_edge_bps < 0.0:
        raise ValueError("min_edge_bps must be non-negative")
    if search.max_edge_bps <= search.min_edge_bps:
        raise ValueError("max_edge_bps must exceed min_edge_bps")
    if search.step_bps <= 0.0:
        raise ValueError("step_bps must be positive")


def _bisect_consistent_edge(
    request: RfqRequest,
    models: ResponderModels,
    search: OptimizerParams,
) -> float:
    low = search.min_edge_bps
    high = search.max_edge_bps
    gap_low = _consistency_gap(request, models, low)
    gap_high = _consistency_gap(request, models, high)
    if gap_low > 0.0:
        raise ValueError("consistent edge is below min_edge_bps")
    if gap_high < 0.0:
        raise ValueError("consistent edge is above max_edge_bps")
    return _bisect_root(request, models, low, high)


def _consistency_gap(
    request: RfqRequest,
    models: ResponderModels,
    edge_bps: float,
) -> float:
    components = evaluate_quote(request, models, edge_bps)
    return components.quoted_edge_bps - components.required_edge_bps


def _bisect_root(
    request: RfqRequest,
    models: ResponderModels,
    low: float,
    high: float,
) -> float:
    current_low = low
    current_high = high
    midpoint = current_low
    for _ in range(80):
        midpoint = 0.5 * (current_low + current_high)
        gap = _consistency_gap(request, models, midpoint)
        if abs(gap) <= 1e-8:
            return midpoint
        if gap < 0.0:
            current_low = midpoint
        else:
            current_high = midpoint
    return midpoint


def _scan_expected_pnl_grid(
    request: RfqRequest,
    models: ResponderModels,
    search: OptimizerParams,
) -> EdgeComponents:
    grid = _edge_grid(search)
    best = evaluate_quote(request, models, grid[0])
    for edge_bps in grid[1:]:
        candidate = evaluate_quote(request, models, edge_bps)
        if candidate.expected_pnl > best.expected_pnl:
            best = candidate
    return best


def _edge_grid(search: OptimizerParams) -> tuple[float, ...]:
    edges: list[float] = []
    current = search.min_edge_bps
    while current <= search.max_edge_bps + 1e-12:
        edges.append(round(current, 10))
        current += search.step_bps
    if not edges:
        raise ValueError("edge grid is empty")
    return tuple(edges)
