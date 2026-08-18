"""Point-in-time feature contracts for RFQ responder models.

Each model consumes an explicit allowlist. The value model V0 must never see
the quote, quote-derived features, or outcomes. Fill and selection models see
the quote only through :func:`make_candidate_quote_features`, which recomputes
quote-dependent features for every candidate price. No model may see latent
simulator columns.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

QUOTE_DEPENDENT_FEATURES: tuple[str, ...] = ("aggressiveness",)

VALUE_NUMERIC_FEATURES: tuple[str, ...] = (
    "internal_alpha",
    "log_size",
    "liquidity_score",
    "log_market_width",
    "log_volatility",
    "inventory",
    "market_signal",
    "issuer_signal",
    "log_bond_age",
    "log_staleness",
    "recent_trade_count",
    "recent_market_move",
    "recent_issuer_move",
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
    "number_of_dealers",
    "log_quote_deadline",
    "log_size",
    "liquidity_score",
    "log_market_width",
    "log_volatility",
    "inventory",
    "is_inventory_axe",
    "market_signal",
    "issuer_signal",
    "log_staleness",
    "recent_trade_count",
    "recent_market_move",
    "recent_issuer_move",
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
    "client_id",
    "venue",
)

FILL_FEATURE_COLUMNS: tuple[str, ...] = FILL_NUMERIC_FEATURES + FILL_CATEGORICAL_FEATURES

SELECTION_NUMERIC_FEATURES: tuple[str, ...] = (
    "aggressiveness",
    "number_of_dealers",
    "log_size",
    "liquidity_score",
    "log_market_width",
    "log_volatility",
    "is_inventory_axe",
    "log_staleness",
    "recent_market_move",
    "recent_issuer_move",
    "day_of_year",
    "month",
    "day_of_week",
)

SELECTION_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "side",
    "sector",
    "rating_bucket",
    "client_tier",
    "regime",
    "issuer_id",
    "bond_id",
    "client_id",
    "venue",
)

SELECTION_FEATURE_COLUMNS: tuple[str, ...] = (
    SELECTION_NUMERIC_FEATURES + SELECTION_CATEGORICAL_FEATURES
)

COST_FEATURE_COLUMNS: tuple[str, ...] = (
    "cp_plus",
    "market_width",
    "volatility",
    "size",
    "liquidity_score",
    "quote_deadline_ms",
)

INVENTORY_FEATURE_COLUMNS: tuple[str, ...] = (
    "side",
    "size",
    "inventory",
    "is_inventory_axe",
)

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


def make_candidate_quote_features(
    df: pd.DataFrame,
    quote: pd.Series | float,
) -> pd.DataFrame:
    """Recompute quote-dependent features for one candidate quote.

    Every candidate price q must pass through this function so that
    quote-dependent features are never stale when the optimizer scans a grid.

    :param df: Validated RFQ dataframe.
    :param quote: Candidate clean quote as a scalar or aligned series.
    :return: Quote-dependent feature block aligned to ``df.index``.
    :raises ValueError: If ``quote`` is None.
    """

    if quote is None:
        raise ValueError("candidate quote features require an explicit quote")
    quote_series = _as_quote_series(df, quote)
    features = pd.DataFrame(index=df.index)
    features["aggressiveness"] = quote_aggressiveness(
        df["side_sign"],
        quote_series,
        df["cp_plus"],
        df["market_width"],
    )
    return features


def make_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the point-in-time feature matrix used to estimate V0.

    V0 predicts unconditional future value, so the contract excludes the
    quote, quote-derived features, competition, and every outcome column.

    :param df: Validated RFQ dataframe.
    :return: Feature dataframe indexed like ``df`` with an explicit allowlist.
    :raises ValueError: If latent columns are present or forbidden outputs appear.
    """

    _assert_no_latent_columns(df)
    state = _point_in_time_state_features(df)
    selected = state[list(VALUE_FEATURE_COLUMNS)]
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

    The quote enters only through the recomputed candidate-quote block.
    Outcome columns such as ``won`` and ``y5`` are excluded.

    :param df: Validated RFQ dataframe.
    :param quote: Optional counterfactual quote; defaults to the historical quote.
    :return: Feature dataframe with an explicit allowlist.
    :raises ValueError: If latent columns are present or forbidden outputs appear.
    """

    _assert_no_latent_columns(df)
    effective_quote = df["quote"] if quote is None else quote
    state = _point_in_time_state_features(df)
    quote_block = make_candidate_quote_features(df, effective_quote)
    combined = pd.concat([state, quote_block], axis=1)
    selected = combined[list(FILL_FEATURE_COLUMNS)]
    _assert_fill_feature_matrix_allowed(selected)
    return selected


def make_selection_features(
    df: pd.DataFrame,
    quote: pd.Series | float | None = None,
) -> pd.DataFrame:
    """Build features for A(q, X) on won RFQs.

    Sparse bonds and clients borrow strength from issuer, tier, and population
    levels through pooled categorical encoding with minimum-frequency pooling.

    :param df: RFQ dataframe, typically restricted to fills.
    :param quote: Optional counterfactual quote; defaults to the historical quote.
    :return: Feature dataframe with an explicit allowlist.
    :raises ValueError: If latent columns are present or forbidden outputs appear.
    """

    _assert_no_latent_columns(df)
    effective_quote = df["quote"] if quote is None else quote
    state = _point_in_time_state_features(df)
    quote_block = make_candidate_quote_features(df, effective_quote)
    combined = pd.concat([state, quote_block], axis=1)
    selected = combined[list(SELECTION_FEATURE_COLUMNS)]
    _assert_fill_feature_matrix_allowed(selected)
    return selected


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


SECONDS_PER_HOUR = 3_600.0


def _point_in_time_state_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive all quote-independent observable features shared by the models.

    :param frame: Validated RFQ dataframe.
    :return: Superset feature dataframe from which allowlists select columns.
    """

    timestamps = pd.to_datetime(frame["timestamp"])
    features = pd.DataFrame(index=frame.index)
    features["internal_alpha"] = frame["internal_mid"] - frame["cp_plus"]
    features["log_size"] = np.log(frame["size"])
    features["liquidity_score"] = frame["liquidity_score"]
    features["log_market_width"] = np.log(frame["market_width"])
    features["log_volatility"] = np.log(frame["volatility"])
    features["inventory"] = frame["inventory"]
    features["market_signal"] = frame["market_signal"]
    features["issuer_signal"] = frame["issuer_signal"]
    features["number_of_dealers"] = frame["number_of_dealers"].astype(float)
    features["log_quote_deadline"] = np.log(frame["quote_deadline_ms"].astype(float))
    features["is_inventory_axe"] = frame["is_inventory_axe"].astype(float)
    features["log_bond_age"] = np.log1p(frame["bond_age_days"].astype(float))
    features["log_staleness"] = np.log1p(
        frame["time_since_last_trade_seconds"].astype(float) / SECONDS_PER_HOUR
    )
    features["recent_trade_count"] = frame["recent_trade_count"].astype(float)
    features["recent_market_move"] = frame["recent_market_move"]
    features["recent_issuer_move"] = frame["recent_issuer_move"]
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
    features["client_id"] = frame["client_id"].astype(str)
    features["venue"] = frame["venue"].astype(str)
    return features


def _as_quote_series(df: pd.DataFrame, quote: pd.Series | float) -> pd.Series:
    if isinstance(quote, pd.Series):
        return quote.astype(float)
    return pd.Series(float(quote), index=df.index)


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
