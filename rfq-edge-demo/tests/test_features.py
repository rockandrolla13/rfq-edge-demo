"""Public value-feature construction behavior."""

import pandas as pd
import pytest

from rfq_edge import make_synthetic_rfqs, make_value_features, make_value_target
from rfq_edge.features import FORBIDDEN_OUTPUT_FEATURES, VALUE_FEATURE_COLUMNS
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
