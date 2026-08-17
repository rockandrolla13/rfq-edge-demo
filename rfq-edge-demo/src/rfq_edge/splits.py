"""Chronological dataset splits shared across models."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ChronologicalSplit:
    """Chronological train and test partitions."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    split_index: int


def chronological_train_test_split(
    df: pd.DataFrame,
    test_fraction: float,
) -> ChronologicalSplit:
    """Split RFQs by timestamp without shuffling.

    :param df: RFQ dataframe containing ``timestamp`` and ``rfq_id``.
    :param test_fraction: Fraction of rows reserved for testing.
    :return: Chronological train and test partitions.
    :raises ValueError: If the split produces an empty partition.
    """

    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    ordered = df.sort_values(["timestamp", "rfq_id"]).reset_index(drop=True)
    split_index = int(len(ordered) * (1.0 - test_fraction))
    if split_index <= 0 or split_index >= len(ordered):
        raise ValueError("chronological split produced an empty train or test set")
    return ChronologicalSplit(
        train_df=ordered.iloc[:split_index].copy(),
        test_df=ordered.iloc[split_index:].copy(),
        split_index=split_index,
    )
