"""Point-in-time feature construction for the unconditional V0 model."""

from __future__ import annotations

import numpy as np
import pandas as pd

VALUE_NUMERIC_FEATURES: tuple[str, ...] = (
    "internal_alpha",
    "log_size",
    "liquidity_score",
    "log_market_width",
    "log_volatility",
    "inventory",
    "market_signal",
    "issuer_signal",
    "day_of_year",
    "month",
    "day_of_week",
)

VALUE_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "side",
    "sector",
    "rating_bucket",
    "client_tier",
    "regime",
    "issuer_id",
    "bond_id",
)

VALUE_FEATURE_COLUMNS: tuple[str, ...] = VALUE_NUMERIC_FEATURES + VALUE_CATEGORICAL_FEATURES

FORBIDDEN_OUTPUT_FEATURES: frozenset[str] = frozenset(
    {
        "y5",
        "won",
        "quote",
        "cp_plus",
        "internal_mid",
        "aggressiveness",
        "value_residual",
        "future_residual",
        "p_win",
        "p_win_true",
    }
)


def make_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the point-in-time feature matrix used to estimate V0.

    Features are restricted to information available before a quote is chosen
    and before the RFQ outcome is observed.

    :param df: Validated RFQ dataframe.
    :return: Feature dataframe indexed like ``df`` with an explicit allowlist.
    :raises ValueError: If latent columns are present or forbidden outputs appear.
    """

    _assert_no_latent_columns(df)
    timestamps = pd.to_datetime(df["timestamp"])
    features = pd.DataFrame(index=df.index)
    features["internal_alpha"] = df["internal_mid"] - df["cp_plus"]
    features["log_size"] = np.log(df["size"])
    features["liquidity_score"] = df["liquidity_score"]
    features["log_market_width"] = np.log(df["market_width"])
    features["log_volatility"] = np.log(df["volatility"])
    features["inventory"] = df["inventory"]
    features["market_signal"] = df["market_signal"]
    features["issuer_signal"] = df["issuer_signal"]
    features["day_of_year"] = timestamps.dt.dayofyear.astype(float)
    features["month"] = timestamps.dt.month.astype(float)
    features["day_of_week"] = timestamps.dt.dayofweek.astype(float)
    features["side"] = df["side"].astype(str)
    features["sector"] = df["sector"].astype(str)
    features["rating_bucket"] = df["rating_bucket"].astype(str)
    features["client_tier"] = df["client_tier"].astype(str)
    features["regime"] = df["regime"].astype(str)
    features["issuer_id"] = df["issuer_id"].astype(str)
    features["bond_id"] = df["bond_id"].astype(str)
    selected = features[list(VALUE_FEATURE_COLUMNS)]
    _assert_feature_matrix_allowed(selected)
    return selected


def make_value_target(df: pd.DataFrame) -> pd.Series:
    """Return the future-value residual target measured from CP+.

    :param df: Validated RFQ dataframe containing ``y5`` and ``cp_plus``.
    :return: ``value_residual = y5 - cp_plus`` aligned to ``df.index``.
    :raises ValueError: If required price columns are missing.
    """

    if "y5" not in df.columns or "cp_plus" not in df.columns:
        raise ValueError("y5 and cp_plus are required to build the value target")
    return df["y5"] - df["cp_plus"]


def _assert_no_latent_columns(df: pd.DataFrame) -> None:
    latent_columns = [column for column in df.columns if column.startswith("latent_")]
    if latent_columns:
        raise ValueError(
            "latent columns are not allowed in value-feature input: "
            f"{latent_columns}"
        )


def _assert_feature_matrix_allowed(features: pd.DataFrame) -> None:
    forbidden_present = [
        column
        for column in features.columns
        if column in FORBIDDEN_OUTPUT_FEATURES or column.startswith("latent_")
    ]
    if forbidden_present:
        raise ValueError(
            "forbidden columns present in value-feature output: "
            f"{forbidden_present}"
        )
