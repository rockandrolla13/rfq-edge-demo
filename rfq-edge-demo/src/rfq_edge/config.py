"""Modelling configuration for RFQ responder components."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValueModelConfig:
    """Configuration for chronological V0 estimation and evaluation.

    :param chronological_test_fraction: Fraction of rows reserved for testing.
    :param number_of_oof_splits: Expanding-window folds for out-of-fold V0.
    :param ridge_alpha_grid: Candidate Ridge penalties searched chronologically.
    :param minimum_category_frequency: Minimum count for a dedicated one-hot level.
    :param size_weight_cap: Cap applied before normalizing RFQ size weights.
    :param random_state: Seed used by sklearn components that accept one.
    """

    chronological_test_fraction: float = 0.20
    number_of_oof_splits: int = 5
    ridge_alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0)
    minimum_category_frequency: int = 20
    size_weight_cap: float = 5.0
    random_state: int = 42

    def __post_init__(self) -> None:
        _validate_test_fraction(self.chronological_test_fraction)
        if self.number_of_oof_splits < 2:
            raise ValueError("number_of_oof_splits must be at least 2")
        if not self.ridge_alpha_grid:
            raise ValueError("ridge_alpha_grid must not be empty")
        if self.minimum_category_frequency < 1:
            raise ValueError("minimum_category_frequency must be at least 1")
        if self.size_weight_cap <= 0.0:
            raise ValueError("size_weight_cap must be positive")


@dataclass(frozen=True)
class FillModelConfig:
    """Configuration for the win-probability model p(win | q, X).

    :param chronological_test_fraction: Fraction of rows reserved for testing.
    :param number_of_cv_splits: Chronological CV folds for logistic penalty search.
    :param logistic_c_grid: Inverse-regularization grid for LogisticRegressionCV.
    :param minimum_category_frequency: Minimum count for a dedicated one-hot level.
    :param calibration_bins: Number of bins for reliability summaries.
    :param random_state: Seed used by sklearn components that accept one.
    """

    chronological_test_fraction: float = 0.20
    number_of_cv_splits: int = 5
    logistic_c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    minimum_category_frequency: int = 20
    calibration_bins: int = 10
    random_state: int = 42

    def __post_init__(self) -> None:
        _validate_test_fraction(self.chronological_test_fraction)
        if self.number_of_cv_splits < 2:
            raise ValueError("number_of_cv_splits must be at least 2")
        if not self.logistic_c_grid:
            raise ValueError("logistic_c_grid must not be empty")
        if self.minimum_category_frequency < 1:
            raise ValueError("minimum_category_frequency must be at least 1")
        if self.calibration_bins < 2:
            raise ValueError("calibration_bins must be at least 2")


@dataclass(frozen=True)
class SelectionModelConfig:
    """Configuration for the adverse-selection model A(q, X) on fills.

    :param chronological_test_fraction: Fraction of rows reserved for testing.
    :param number_of_cv_splits: Chronological CV folds for Ridge penalty search.
    :param ridge_alpha_grid: Candidate Ridge penalties searched chronologically.
    :param minimum_category_frequency: Minimum count for a dedicated one-hot level.
    :param random_state: Seed used by sklearn components that accept one.
    """

    chronological_test_fraction: float = 0.20
    number_of_cv_splits: int = 5
    ridge_alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0)
    minimum_category_frequency: int = 20
    random_state: int = 42

    def __post_init__(self) -> None:
        _validate_test_fraction(self.chronological_test_fraction)
        if self.number_of_cv_splits < 2:
            raise ValueError("number_of_cv_splits must be at least 2")
        if not self.ridge_alpha_grid:
            raise ValueError("ridge_alpha_grid must not be empty")
        if self.minimum_category_frequency < 1:
            raise ValueError("minimum_category_frequency must be at least 1")


def _validate_test_fraction(test_fraction: float) -> None:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("chronological_test_fraction must be between 0 and 1")
