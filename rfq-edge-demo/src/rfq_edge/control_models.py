"""Fitted quote models for the control environment.

Fitted from an observable RFQ history only (regime, aggressiveness, size,
win outcome, realized selection on fills). Hidden simulator variables never
enter the design matrices. The interface matches
:class:`rfq_edge.oracle_control.OracleControlModels`, so planners and
controllers are agnostic about which bundle they hold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rfq_edge.control_config import REGIME_ORDER
from rfq_edge.control_state import RFQEvent

MINIMUM_FILLS_PER_REGIME = 30


@dataclass(frozen=True)
class FittedControlModels:
    """Fitted fill and adverse-selection models on the control market.

    :param fill_coefficients: Logistic coefficients per regime, each row
        (intercept, slope_z, size_penalty).
    :param selection_coefficients: Linear coefficients per regime, each row
        (intercept, slope_z) for A(z, r) in points.
    """

    fill_coefficients: np.ndarray
    selection_coefficients: np.ndarray

    def fill_probability(
        self,
        regime_index: int,
        aggressiveness: np.ndarray,
        size: int,
    ) -> np.ndarray:
        """Predicted fill probability per candidate aggressiveness.

        :param regime_index: Regime index.
        :param aggressiveness: Candidate normalized aggressiveness values.
        :param size: RFQ size in units.
        :return: Probabilities per candidate.
        """

        intercept, slope, size_penalty = self.fill_coefficients[regime_index]
        z = np.asarray(aggressiveness, dtype=float)
        logits = intercept + slope * z - size_penalty * float(size - 1)
        return 1.0 / (1.0 + np.exp(-logits))

    def selection_points(
        self,
        regime_index: int,
        aggressiveness: np.ndarray,
    ) -> np.ndarray:
        """Predicted adverse selection A(z, r) in points.

        :param regime_index: Regime index.
        :param aggressiveness: Candidate normalized aggressiveness values.
        :return: Adverse selection per candidate, in points.
        """

        intercept, slope = self.selection_coefficients[regime_index]
        z = np.asarray(aggressiveness, dtype=float)
        return intercept + slope * z

    def event_fill_probability(
        self,
        event: RFQEvent,
        aggressiveness: np.ndarray,
    ) -> np.ndarray:
        """Per-event prediction using observable event fields only.

        :param event: RFQ event; hidden fields are never read.
        :param aggressiveness: Candidate normalized aggressiveness values.
        :return: Probabilities per candidate.
        """

        return self.fill_probability(event.regime.value, aggressiveness, event.size)

    def event_post_win_value(
        self,
        event: RFQEvent,
        aggressiveness: np.ndarray,
    ) -> np.ndarray:
        """Per-event post-win value m(q, X) = V0 - side_sign * A(z, r).

        The control-market V0 equals CP+ because observables carry no alpha
        by construction; all conditional information sits in A.

        :param event: RFQ event; hidden fields are never read.
        :param aggressiveness: Candidate normalized aggressiveness values.
        :return: Post-win clean values per candidate, in points.
        """

        selection = self.selection_points(event.regime.value, aggressiveness)
        return event.cp_plus - float(event.side_sign) * selection


def fit_control_models(history: pd.DataFrame) -> FittedControlModels:
    """Fit per-regime fill and selection models from observable history.

    The fill model is a logistic regression on (z, size); the selection model
    is a per-regime linear fit of the realized selection target on z over
    fills. Both are estimated independently per regime.

    :param history: Output of ``generate_training_history``.
    :return: Fitted models.
    :raises ValueError: If any regime lacks observations or fills.
    """

    required = {"regime_index", "aggressiveness", "size", "won", "realized_selection_points"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history is missing columns: {sorted(missing)}")

    n_regimes = len(REGIME_ORDER)
    fill_coefficients = np.zeros((n_regimes, 3))
    selection_coefficients = np.zeros((n_regimes, 2))
    for regime_index in range(n_regimes):
        regime_rows = history.loc[history["regime_index"] == regime_index]
        if regime_rows.empty:
            raise ValueError(f"no history for regime index {regime_index}")
        fill_coefficients[regime_index] = _fit_logistic(
            aggressiveness=regime_rows["aggressiveness"].to_numpy(dtype=float),
            size=regime_rows["size"].to_numpy(dtype=float),
            won=regime_rows["won"].to_numpy(dtype=bool),
        )
        fills = regime_rows.loc[regime_rows["won"]]
        if len(fills) < MINIMUM_FILLS_PER_REGIME:
            raise ValueError(
                f"regime index {regime_index} has too few fills "
                f"({len(fills)} < {MINIMUM_FILLS_PER_REGIME})"
            )
        selection_coefficients[regime_index] = _fit_linear(
            aggressiveness=fills["aggressiveness"].to_numpy(dtype=float),
            target=fills["realized_selection_points"].to_numpy(dtype=float),
        )
    return FittedControlModels(
        fill_coefficients=fill_coefficients,
        selection_coefficients=selection_coefficients,
    )


def _fit_logistic(
    aggressiveness: np.ndarray,
    size: np.ndarray,
    won: np.ndarray,
    n_iterations: int = 60,
) -> np.ndarray:
    """Newton-Raphson logistic fit of win on (1, z, -(size - 1))."""

    design = np.column_stack(
        [np.ones_like(aggressiveness), aggressiveness, -(size - 1.0)]
    )
    outcome = won.astype(float)
    beta = np.zeros(design.shape[1])
    for _ in range(n_iterations):
        logits = design @ beta
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (outcome - probabilities)
        weights = probabilities * (1.0 - probabilities)
        hessian = design.T @ (design * weights[:, None]) + 1e-8 * np.eye(design.shape[1])
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return beta


def _fit_linear(aggressiveness: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Ordinary least squares of the selection target on (1, z)."""

    design = np.column_stack([np.ones_like(aggressiveness), aggressiveness])
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return coefficients
