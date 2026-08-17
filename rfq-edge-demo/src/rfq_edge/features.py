"""Point-in-time feature construction for RFQ responder models."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

FILL_NUMERIC_FEATURES: tuple[str, ...] = (
    "aggressiveness",
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

FILL_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "side",
    "sector",
    "rating_bucket",
    "client_tier",
    "regime",
    "issuer_id",
    "bond_id",
)

FILL_FEATURE_COLUMNS: tuple[str, ...] = FILL_NUMERIC_FEATURES + FILL_CATEGORICAL_FEATURES

SELECTION_NUMERIC_FEATURES: tuple[str, ...] = FILL_NUMERIC_FEATURES
SELECTION_CATEGORICAL_FEATURES: tuple[str, ...] = FILL_CATEGORICAL_FEATURES
SELECTION_FEATURE_COLUMNS: tuple[str, ...] = FILL_FEATURE_COLUMNS

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

FORBIDDEN_FILL_FEATURES: frozenset[str] = frozenset(
    {
        "y5",
        "won",
        "value_residual",
        "future_residual",
        "p_win",
        "p_win_true",
        "v0_oof",
    }
)


def quote_aggressiveness(
    side_sign: pd.Series,
    quote: pd.Series,
    cp_plus: pd.Series,
    market_width: pd.Series,
) -> pd.Series:
    """Compute normalized quote aggressiveness z = side_sign * (q - CP+) / width.

    :param side_sign: Dealer side indicator (+1 buy, -1 sell).
    :param quote: Candidate or historical clean quote.
    :param cp_plus: CP+ clean mid at RFQ time.
    :param market_width: Market width in price points.
    :return: Normalized aggressiveness aligned to the input index.
    """

    return side_sign.astype(float) * (quote.astype(float) - cp_plus.astype(float)) / market_width.astype(float)


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


def make_fill_features(
    df: pd.DataFrame,
    quote: pd.Series | float | None = None,
) -> pd.DataFrame:
    """Build features for p(win | q, X).

    The quote enters only through normalized aggressiveness. Outcome columns
    such as ``won`` and ``y5`` are excluded.

    :param df: Validated RFQ dataframe.
    :param quote: Optional counterfactual quote override.
    :return: Feature dataframe with an explicit allowlist.
    :raises ValueError: If latent columns are present or forbidden outputs appear.
    """

    frame = _frame_with_quote(df, quote)
    _assert_no_latent_columns(frame)
    timestamps = pd.to_datetime(frame["timestamp"])
    features = pd.DataFrame(index=frame.index)
    features["aggressiveness"] = quote_aggressiveness(
        frame["side_sign"],
        frame["quote"],
        frame["cp_plus"],
        frame["market_width"],
    )
    features["log_size"] = np.log(frame["size"])
    features["liquidity_score"] = frame["liquidity_score"]
    features["log_market_width"] = np.log(frame["market_width"])
    features["log_volatility"] = np.log(frame["volatility"])
    features["inventory"] = frame["inventory"]
    features["market_signal"] = frame["market_signal"]
    features["issuer_signal"] = frame["issuer_signal"]
    features["day_of_year"] = timestamps.dt.dayofyear.astype(float)
    features["month"] = timestamps.dt.month.astype(float)
    features["day_of_week"] = timestamps.dt.dayofweek.astype(float)
    features["side"] = frame["side"].astype(str)
    features["sector"] = frame["sector"].astype(str)
    features["rating_bucket"] = frame["rating_bucket"].astype(str)
    features["client_tier"] = frame["client_tier"].astype(str)
    features["regime"] = frame["regime"].astype(str)
    features["issuer_id"] = frame["issuer_id"].astype(str)
    features["bond_id"] = frame["bond_id"].astype(str)
    selected = features[list(FILL_FEATURE_COLUMNS)]
    _assert_fill_feature_matrix_allowed(selected)
    return selected


def make_selection_features(
    df: pd.DataFrame,
    quote: pd.Series | float | None = None,
) -> pd.DataFrame:
    """Build features for A(q, X) on won RFQs.

    Sparse bonds borrow strength from issuer and population levels through
    pooled categorical encoding with minimum-frequency pooling.

    :param df: RFQ dataframe, typically restricted to fills.
    :param quote: Optional counterfactual quote override.
    :return: Feature dataframe with an explicit allowlist.
    """

    return make_fill_features(df, quote=quote)


def build_feature_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    minimum_category_frequency: int,
) -> ColumnTransformer:
    """Create the shared numeric and categorical preprocessing block.

    :param numeric_features: Numeric feature names.
    :param categorical_features: Categorical feature names.
    :param minimum_category_frequency: Minimum count for a dedicated one-hot level.
    :return: Column transformer for model pipelines.
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=minimum_category_frequency,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_features)),
            ("categorical", categorical_pipeline, list(categorical_features)),
        ]
    )


def _frame_with_quote(
    df: pd.DataFrame,
    quote: pd.Series | float | None,
) -> pd.DataFrame:
    if quote is None:
        return df
    frame = df.copy()
    if isinstance(quote, pd.Series):
        frame["quote"] = quote.astype(float)
    else:
        frame["quote"] = float(quote)
    return frame


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


def _assert_fill_feature_matrix_allowed(features: pd.DataFrame) -> None:
    forbidden_present = [
        column
        for column in features.columns
        if column in FORBIDDEN_FILL_FEATURES or column.startswith("latent_")
    ]
    if forbidden_present:
        raise ValueError(
            "forbidden columns present in fill-feature output: "
            f"{forbidden_present}"
        )
