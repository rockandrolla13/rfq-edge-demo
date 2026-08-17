"""Public interface for the edge-consistent RFQ responder demo."""

from rfq_edge.config import ValueModelConfig
from rfq_edge.evaluation import compute_forecast_metrics, format_metrics_table
from rfq_edge.features import (
    VALUE_CATEGORICAL_FEATURES,
    VALUE_FEATURE_COLUMNS,
    VALUE_NUMERIC_FEATURES,
    make_value_features,
    make_value_target,
)
from rfq_edge.responder_value import (
    FairValue,
    ValueModelParams,
    estimate_fair_value,
    quoted_price,
)
from rfq_edge.schema import validate_rfq_schema
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
    SyntheticConfig,
    SyntheticRfq,
    demo_book_spec,
    generate_modeling_book,
    generate_rfq_book,
    make_synthetic_rfqs,
    to_rfq_request,
    validate_synthetic_data,
)
from rfq_edge.value_model import (
    FittedValueModel,
    ValueForecastKind,
    chronological_train_test_split,
    evaluate_value_models,
    fit_value_model,
    make_chronological_oof_v0,
    predict_v0,
    predict_value_residual,
)

__all__ = [
    "BondInfo",
    "CostParams",
    "EdgeComponents",
    "FairValue",
    "FillModelParams",
    "FittedValueModel",
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
    "SyntheticConfig",
    "SyntheticRfq",
    "VALUE_CATEGORICAL_FEATURES",
    "VALUE_FEATURE_COLUMNS",
    "VALUE_NUMERIC_FEATURES",
    "ValueForecastKind",
    "ValueModelConfig",
    "ValueModelParams",
    "adverse_selection_bps",
    "chronological_train_test_split",
    "compute_forecast_metrics",
    "default_config",
    "demo_book_spec",
    "estimate_fair_value",
    "evaluate_quote",
    "evaluate_value_models",
    "fill_probability",
    "fit_value_model",
    "format_metrics_table",
    "generate_modeling_book",
    "generate_rfq_book",
    "make_chronological_oof_v0",
    "make_synthetic_rfqs",
    "make_value_features",
    "make_value_target",
    "maximize_expected_pnl",
    "predict_v0",
    "predict_value_residual",
    "quoted_price",
    "run_book",
    "run_responder",
    "solve_consistent_edge",
    "to_rfq_request",
    "trading_cost_bps",
    "validate_rfq_schema",
    "validate_synthetic_data",
]
