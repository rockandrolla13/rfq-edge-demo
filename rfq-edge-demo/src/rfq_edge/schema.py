"""Input schema validation for RFQ modelling datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rfq_edge.synthetic import OBSERVABLE_COLUMNS

REQUIRED_RFQ_COLUMNS: tuple[str, ...] = OBSERVABLE_COLUMNS

PRICE_COLUMNS: tuple[str, ...] = (
    "cp_plus",
    "internal_mid",
    "quote",
    "y5",
)

FORBIDDEN_MODEL_INPUT_PREFIXES: tuple[str, ...] = ("latent_",)


def validate_rfq_schema(df: pd.DataFrame) -> None:
    """Validate that a dataframe satisfies the RFQ modelling schema.

    :param df: Candidate RFQ dataset.
    :raises ValueError: If required columns, types, or constraints fail.
    :raises TypeError: If ``df`` is not a dataframe.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    missing = [column for column in REQUIRED_RFQ_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    latent_columns = [column for column in df.columns if column.startswith("latent_")]
    if latent_columns:
        raise ValueError(
            "latent columns are not allowed in ordinary model input: "
            f"{latent_columns}"
        )

    if df["rfq_id"].duplicated().any():
        duplicated = df.loc[df["rfq_id"].duplicated(), "rfq_id"].tolist()
        raise ValueError(f"rfq_id must be unique; duplicates found: {duplicated[:5]}")

    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("timestamp contains values that are not parseable")

    side_sign_values = set(df["side_sign"].unique())
    allowed_signs = {-1.0, 1.0, -1, 1}
    if not side_sign_values.issubset(allowed_signs):
        raise ValueError(f"side_sign must be +1 or -1; found {sorted(side_sign_values)}")

    if (df["size"] <= 0.0).any():
        raise ValueError("size must be strictly positive")
    if (df["market_width"] <= 0.0).any():
        raise ValueError("market_width must be strictly positive")

    won_values = set(df["won"].unique())
    if not won_values.issubset({True, False, 0, 1}):
        raise ValueError("won must be binary")

    for column in PRICE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"{column} must be numeric")
        if not np.isfinite(df[column].to_numpy()).all():
            raise ValueError(f"{column} must contain only finite values")
