"""Public interface for the edge-consistent RFQ responder demo."""

from rfq_edge.costs import CostParams, trading_cost_bps
from rfq_edge.fill_model import FillModelParams, fill_probability
from rfq_edge.objective import EdgeComponents, ResponderModels, evaluate_quote
from rfq_edge.optimizer import (
    OptimizerParams,
    QuoteSolution,
    maximize_expected_pnl,
    solve_consistent_edge,
)
from rfq_edge.pipeline import (
    PipelineConfig,
    ResponderDecision,
    default_config,
    run_book,
    run_responder,
)
from rfq_edge.selection_model import SelectionModelParams, adverse_selection_bps
from rfq_edge.synthetic import (
    BondInfo,
    IssuerInfo,
    RfqRequest,
    Side,
    SyntheticBookSpec,
    SyntheticRfq,
    demo_book_spec,
    generate_modeling_book,
    generate_rfq_book,
    to_rfq_request,
)
from rfq_edge.value_model import (
    FairValue,
    ValueModelParams,
    estimate_fair_value,
    quoted_price,
)

__all__ = [
    "BondInfo",
    "CostParams",
    "EdgeComponents",
    "FairValue",
    "FillModelParams",
    "IssuerInfo",
    "OptimizerParams",
    "PipelineConfig",
    "QuoteSolution",
    "ResponderDecision",
    "ResponderModels",
    "RfqRequest",
    "SelectionModelParams",
    "Side",
    "SyntheticBookSpec",
    "SyntheticRfq",
    "ValueModelParams",
    "adverse_selection_bps",
    "default_config",
    "demo_book_spec",
    "estimate_fair_value",
    "evaluate_quote",
    "fill_probability",
    "generate_modeling_book",
    "generate_rfq_book",
    "maximize_expected_pnl",
    "quoted_price",
    "run_book",
    "run_responder",
    "solve_consistent_edge",
    "to_rfq_request",
    "trading_cost_bps",
]
