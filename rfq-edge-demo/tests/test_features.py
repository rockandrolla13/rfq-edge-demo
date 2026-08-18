"""Public feature-contract behavior."""

import pandas as pd
import pytest

from rfq_edge import (
    make_candidate_quote_features,
    make_fill_features,
    make_selection_features,
    make_synthetic_rfqs,
    make_value_features,
    make_value_target,
)
from rfq_edge.features import (
    COST_FEATURE_COLUMNS,
    FILL_FEATURE_COLUMNS,
    FORBIDDEN_OUTPUT_FEATURES,
    INVENTORY_FEATURE_COLUMNS,
    SELECTION_FEATURE_COLUMNS,
    VALUE_FEATURE_COLUMNS,
)
from rfq_edge.synthetic import SyntheticConfig


def _sample_frame(include_latent: bool = False) -> pd.DataFrame:
    return make_synthetic_rfqs(
        config=SyntheticConfig(n_rfqs=500, n_bonds=40, n_issuers=10),
        random_state=42,
        include_latent=include_latent,
    )


def test_make_value_features_excludes_y5() -> None:
    frame = _sample_frame()
    features = make_value_features(frame)
    assert "y5" not in features.columns


def test_make_value_features_excludes_won() -> None:
    frame = _sample_frame()
    features = make_value_features(frame)
    assert "won" not in features.columns


def test_make_value_features_excludes_quote() -> None:
    frame = _sample_frame()
    features = make_value_features(frame)
    assert "quote" not in features.columns


def test_make_value_features_excludes_latent_columns() -> None:
    frame = _sample_frame(include_latent=True)
    with pytest.raises(ValueError, match="latent columns"):
        make_value_features(frame)


def test_make_value_features_uses_explicit_allowlist() -> None:
    frame = _sample_frame()
    features = make_value_features(frame)
    assert list(features.columns) == list(VALUE_FEATURE_COLUMNS)


def test_make_value_target_equals_y5_minus_cp_plus() -> None:
    frame = _sample_frame()
    target = make_value_target(frame)
    expected = frame["y5"] - frame["cp_plus"]
    pd.testing.assert_series_equal(target, expected, check_names=False)


def test_forbidden_feature_names_are_documented() -> None:
    assert "y5" in FORBIDDEN_OUTPUT_FEATURES
    assert "quote" in FORBIDDEN_OUTPUT_FEATURES
    assert "won" in FORBIDDEN_OUTPUT_FEATURES


def test_value_contract_excludes_quote_dependent_and_competition_features() -> None:
    assert "aggressiveness" not in VALUE_FEATURE_COLUMNS
    assert "number_of_dealers" not in VALUE_FEATURE_COLUMNS
    assert "client_id" not in VALUE_FEATURE_COLUMNS


def test_fill_contract_includes_competition_and_client_features() -> None:
    assert "aggressiveness" in FILL_FEATURE_COLUMNS
    assert "number_of_dealers" in FILL_FEATURE_COLUMNS
    assert "client_id" in FILL_FEATURE_COLUMNS
    assert "venue" in FILL_FEATURE_COLUMNS


def test_selection_contract_includes_winner_curse_features() -> None:
    assert "aggressiveness" in SELECTION_FEATURE_COLUMNS
    assert "number_of_dealers" in SELECTION_FEATURE_COLUMNS
    assert "client_id" in SELECTION_FEATURE_COLUMNS


def test_cost_and_inventory_contracts_are_documented() -> None:
    assert "quote_deadline_ms" in COST_FEATURE_COLUMNS
    assert "is_inventory_axe" in INVENTORY_FEATURE_COLUMNS


def test_make_fill_features_matches_fill_allowlist() -> None:
    frame = _sample_frame()
    features = make_fill_features(frame)
    assert list(features.columns) == list(FILL_FEATURE_COLUMNS)


def test_make_selection_features_matches_selection_allowlist() -> None:
    frame = _sample_frame()
    features = make_selection_features(frame)
    assert list(features.columns) == list(SELECTION_FEATURE_COLUMNS)


def test_candidate_quote_features_recompute_aggressiveness() -> None:
    frame = _sample_frame()
    baseline = make_candidate_quote_features(frame, frame["quote"])
    shifted_quote = frame["quote"] + frame["side_sign"] * frame["market_width"]
    shifted = make_candidate_quote_features(frame, shifted_quote)
    difference = shifted["aggressiveness"] - baseline["aggressiveness"]
    pd.testing.assert_series_equal(
        difference,
        pd.Series(1.0, index=frame.index),
        check_names=False,
    )


def test_candidate_quote_features_require_explicit_quote() -> None:
    frame = _sample_frame()
    with pytest.raises(ValueError, match="explicit quote"):
        make_candidate_quote_features(frame, None)


def test_fill_features_use_candidate_quote_override() -> None:
    frame = _sample_frame()
    historical = make_fill_features(frame)
    counterfactual = make_fill_features(frame, quote=frame["quote"] + 0.5)
    assert not historical["aggressiveness"].equals(counterfactual["aggressiveness"])
