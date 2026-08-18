"""Public behavior of the three shared-machinery responders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rfq_edge.config import OptimizerConfig
from rfq_edge.optimizer import resolve_v0
from rfq_edge.responders import (
    ALL_RESPONDERS,
    ResponderKind,
    compare_responders,
    observable_view,
    respond_to_rfq,
    scan_responder_grid,
)


def _test_rfq(framework) -> pd.DataFrame:
    candidates = framework.test_df.loc[framework.test_df["v0_oof"].notna()]
    return candidates.iloc[[0]]


def test_observable_view_drops_latent_columns(demo_frame) -> None:
    view = observable_view(demo_frame)
    assert not any(column.startswith("latent_") for column in view.columns)
    assert len(view) == len(demo_frame)


def test_scan_grid_shares_fill_cost_and_support_across_responders(demo_framework) -> None:
    grid = scan_responder_grid(_test_rfq(demo_framework), demo_framework.models)
    shared_columns = {"quote", "aggressiveness", "p_win", "cost_cents", "inventory_value_cents", "in_support"}
    assert shared_columns.issubset(grid.columns)
    for kind in ALL_RESPONDERS:
        assert f"expected_value_cents_{kind.value}" in grid.columns


def test_plain_cp_plus_assumes_cp_plus_post_win_value(demo_framework) -> None:
    rfq = _test_rfq(demo_framework)
    grid = scan_responder_grid(rfq, demo_framework.models)
    cp_plus = float(rfq.iloc[0]["cp_plus"])
    assert np.allclose(grid["post_win_value_plain_cp_plus"], cp_plus)


def test_plain_v0_assumes_v0_post_win_value(demo_framework) -> None:
    rfq = _test_rfq(demo_framework)
    grid = scan_responder_grid(rfq, demo_framework.models)
    v0 = float(resolve_v0(observable_view(rfq), demo_framework.models).iloc[0])
    assert np.allclose(grid["post_win_value_plain_v0"], v0)


def test_edge_consistent_subtracts_signed_selection(demo_framework) -> None:
    rfq = _test_rfq(demo_framework)
    grid = scan_responder_grid(rfq, demo_framework.models)
    side_sign = float(rfq.iloc[0]["side_sign"])
    expected = grid["post_win_value_plain_v0"] - side_sign * grid["selection_points"]
    np.testing.assert_allclose(
        grid["post_win_value_edge_consistent"].to_numpy(),
        expected.to_numpy(),
        rtol=1e-12,
    )


def test_selection_haircut_scales_the_correction(demo_framework) -> None:
    rfq = _test_rfq(demo_framework)
    raw = scan_responder_grid(rfq, demo_framework.models, OptimizerConfig(selection_haircut=1.0))
    off = scan_responder_grid(rfq, demo_framework.models, OptimizerConfig(selection_haircut=0.0))
    np.testing.assert_allclose(
        off["post_win_value_edge_consistent"].to_numpy(),
        off["post_win_value_plain_v0"].to_numpy(),
        rtol=1e-12,
    )
    assert not np.allclose(
        raw["post_win_value_edge_consistent"].to_numpy(),
        raw["post_win_value_plain_v0"].to_numpy(),
    )


def test_every_responder_can_decline_under_extreme_costs(demo_framework) -> None:
    rfq = _test_rfq(demo_framework)
    punitive = OptimizerConfig(transaction_bps=10_000.0)
    for kind in ALL_RESPONDERS:
        decision = respond_to_rfq(rfq, demo_framework.models, kind, punitive)
        assert not decision.accepted
        assert decision.quote is None


def test_accepted_decision_has_positive_objective_and_support(demo_framework) -> None:
    rfq = _test_rfq(demo_framework)
    for kind in ALL_RESPONDERS:
        decision = respond_to_rfq(rfq, demo_framework.models, kind)
        if decision.accepted:
            assert decision.expected_value_cents > 0.0
            support_low, support_high = demo_framework.models.fill_model.aggressiveness_support
            assert support_low <= decision.aggressiveness <= support_high


def test_compare_responders_returns_one_row_per_responder(demo_framework) -> None:
    comparison = compare_responders(_test_rfq(demo_framework), demo_framework.models)
    assert len(comparison) == len(ALL_RESPONDERS)
    assert set(comparison["kind"]) == {kind.value for kind in ALL_RESPONDERS}
    assert comparison["cost_cents"].nunique() == 1


def test_scan_grid_rejects_multi_row_input(demo_framework) -> None:
    rows = demo_framework.test_df.iloc[[0, 1]]
    with pytest.raises(ValueError, match="exactly one RFQ row"):
        scan_responder_grid(rows, demo_framework.models)


def test_responder_kind_values_are_stable() -> None:
    assert ResponderKind.PLAIN_CP_PLUS.value == "plain_cp_plus"
    assert ResponderKind.PLAIN_V0.value == "plain_v0"
    assert ResponderKind.EDGE_CONSISTENT.value == "edge_consistent"
