"""Public combined quote optimizer behavior."""

import pandas as pd

from rfq_edge import (
    FillModelConfig,
    OptimizerConfig,
    SelectionModelConfig,
    ValueModelConfig,
    evaluate_quote_grid,
    fit_quote_models,
    format_quote_table,
    make_chronological_oof_v0,
    make_synthetic_rfqs,
    optimize_quote,
    quote_from_aggressiveness,
    value_chronological_split,
)
from rfq_edge.responder_optimizer import (
    OptimizerParams,
    QuoteSolution,
    maximize_expected_pnl,
    solve_consistent_edge,
)
from rfq_edge.pipeline import default_config
from rfq_edge.synthetic import RfqRequest, Side, SyntheticConfig


def _prepared_models():
    frame = make_synthetic_rfqs(
        config=SyntheticConfig(n_rfqs=2_000, n_bonds=80, n_issuers=20),
        random_state=42,
    )
    oof = make_chronological_oof_v0(frame, ValueModelConfig(number_of_oof_splits=3))
    prepared = frame.merge(oof[["rfq_id", "v0_oof"]], on="rfq_id", how="left")
    split = value_chronological_split(prepared, ValueModelConfig(chronological_test_fraction=0.2))
    train_fills = split.train_df.loc[
        split.train_df["won"] & split.train_df["v0_oof"].notna()
    ]
    models = fit_quote_models(
        split.train_df,
        train_fills,
        ValueModelConfig(number_of_oof_splits=3),
        FillModelConfig(number_of_cv_splits=3),
        SelectionModelConfig(number_of_cv_splits=3),
    )
    return prepared.loc[prepared["v0_oof"].notna()], models


def test_evaluate_quote_grid_has_required_columns() -> None:
    prepared, models = _prepared_models()
    rfq = prepared.iloc[[100]]
    table = evaluate_quote_grid(rfq, models, OptimizerConfig(aggressiveness_step=0.5))
    required = {
        "quote",
        "p_win",
        "post_win_value",
        "clean_edge_cents",
        "cost_cents",
        "expected_value_cents",
    }
    assert required.issubset(table.columns)
    assert len(table) >= 3


def test_higher_aggressiveness_raises_win_probability_in_grid() -> None:
    prepared, models = _prepared_models()
    rfq = prepared.iloc[[100]]
    table = evaluate_quote_grid(
        rfq,
        models,
        OptimizerConfig(min_aggressiveness=-1.0, max_aggressiveness=1.0, aggressiveness_step=1.0),
    )
    assert table["p_win"].is_monotonic_increasing


def test_clean_edge_formula_matches_side_sign() -> None:
    prepared, models = _prepared_models()
    rfq = prepared.iloc[[100]]
    table = evaluate_quote_grid(rfq, models, OptimizerConfig(aggressiveness_step=0.5))
    row = rfq.iloc[0]
    for _, candidate in table.iterrows():
        expected_edge = float(row["side_sign"]) * (candidate["post_win_value"] - candidate["quote"])
        assert abs(candidate["clean_edge_cents"] / 100.0 - expected_edge) < 1e-9


def test_expected_value_formula() -> None:
    prepared, models = _prepared_models()
    rfq = prepared.iloc[[100]]
    table = evaluate_quote_grid(rfq, models, OptimizerConfig(aggressiveness_step=0.5))
    for _, candidate in table.iterrows():
        expected = candidate["p_win"] * (
            candidate["clean_edge_cents"]
            - candidate["cost_cents"]
            + candidate["inventory_value_cents"]
        )
        assert abs(candidate["expected_value_cents"] - expected) < 1e-9


def test_optimize_quote_selects_best_expected_value() -> None:
    prepared, models = _prepared_models()
    rfq = prepared.iloc[[100]]
    table = evaluate_quote_grid(rfq, models, OptimizerConfig(aggressiveness_step=0.5))
    decision = optimize_quote(rfq, models, OptimizerConfig(aggressiveness_step=0.5))
    best = table["expected_value_cents"].max()
    if best <= 0.0:
        assert not decision.accepted
        assert decision.quote is None
    else:
        assert decision.accepted
        assert decision.quote is not None
        assert abs(decision.expected_value_cents - best) < 1e-9


def test_format_quote_table_renders_columns() -> None:
    prepared, models = _prepared_models()
    rfq = prepared.iloc[[100]]
    table = evaluate_quote_grid(rfq, models, OptimizerConfig(aggressiveness_step=0.5))
    text = format_quote_table(table)
    assert "Quote" in text
    assert "Expected value" in text
    assert "Win probability" in text


def test_quote_from_aggressiveness_reconstructs_quote() -> None:
    prepared, _models = _prepared_models()
    row = prepared.iloc[100]
    quote = quote_from_aggressiveness(row, 0.5)
    expected = row["cp_plus"] + row["side_sign"] * 0.5 * row["market_width"]
    assert abs(quote - expected) < 1e-9


def test_legacy_consistent_quote_still_works() -> None:
    request = RfqRequest(
        rfq_id="rfq-test",
        side=Side.BUY,
        quantity=1_000.0,
        mid_price=100.0,
        volatility=0.2,
        inventory=0.0,
        time_to_hedge=0.01,
        competition_count=3,
    )
    config = default_config()
    solution = solve_consistent_edge(request, config.models, config.search)
    assert solution.rule == "edge_consistent"


def test_quote_optimizer_demo_for_default_seed() -> None:
    frame = make_synthetic_rfqs(random_state=42)
    oof = make_chronological_oof_v0(frame, ValueModelConfig())
    prepared = frame.merge(oof[["rfq_id", "v0_oof"]], on="rfq_id", how="left")
    split = value_chronological_split(prepared, ValueModelConfig())
    train_fills = split.train_df.loc[
        split.train_df["won"] & split.train_df["v0_oof"].notna()
    ]
    models = fit_quote_models(split.train_df, train_fills)
    rfq = split.test_df.iloc[[0]].copy()
    table = evaluate_quote_grid(rfq, models)
    print("\nquote optimizer table:\n", format_quote_table(table.head(3)))
    decision = optimize_quote(rfq, models)
    assert not table.empty
    assert decision.candidate_table is not None
