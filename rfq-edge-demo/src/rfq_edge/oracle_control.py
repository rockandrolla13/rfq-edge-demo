"""Synthetic-truth oracle for the control environment.

Completely separate from fitted models. The oracle knows the data-generating
process, so it can compute, for an arbitrary candidate aggressiveness:

* the marginal (population) true fill probability and adverse selection by
  Gauss-Hermite quadrature over the hidden client signal — deterministic,
  so no Monte Carlo seed is needed;
* per-event truths using the event's hidden client signal, used only by the
  oracle benchmark controller and realized-outcome diagnostics.

Fitted policies never receive hidden fields; the audit tests verify that
perturbing hidden fields does not change fitted-policy actions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rfq_edge.control_config import ControlMarketConfig, MarketRegime, REGIME_ORDER
from rfq_edge.control_state import RFQEvent

QUADRATURE_NODES = 41


@dataclass(frozen=True)
class OracleControlModels:
    """Truth-based quote model with the same interface as the fitted one.

    :param market_config: Data-generating configuration.
    """

    market_config: ControlMarketConfig
    _nodes: np.ndarray = field(init=False, repr=False)
    _weights: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Gauss-Hermite nodes rescaled for a standard normal integrand.
        nodes, weights = np.polynomial.hermite_e.hermegauss(QUADRATURE_NODES)
        object.__setattr__(self, "_nodes", nodes)
        object.__setattr__(self, "_weights", weights / weights.sum())

    def fill_probability(
        self,
        regime_index: int,
        aggressiveness: np.ndarray,
        size: int,
    ) -> np.ndarray:
        """Marginal true fill probability, hidden signal integrated out.

        :param regime_index: Regime index.
        :param aggressiveness: Candidate normalized aggressiveness values.
        :param size: RFQ size in units.
        :return: Fill probabilities per candidate.
        """

        params = self.market_config.parameters_for(REGIME_ORDER[regime_index])
        z = np.asarray(aggressiveness, dtype=float)
        base = (
            params.fill_intercept
            + self.market_config.fill_aggressiveness_coef * z
            - self.market_config.size_win_penalty * float(size - 1)
        )
        # side_sign drops out marginally because h is symmetric around zero.
        logits = base[:, None] - params.information_coefficient * self._nodes[None, :]
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        return probabilities @ self._weights

    def selection_points(
        self,
        regime_index: int,
        aggressiveness: np.ndarray,
    ) -> np.ndarray:
        """Marginal true adverse selection A(z, r) in points.

        A = E[side_sign * (V0 - y5) | win at z]; by symmetry of the hidden
        signal it is identical for both dealer sides and positive.

        :param regime_index: Regime index.
        :param aggressiveness: Candidate normalized aggressiveness values.
        :return: Adverse selection per candidate, in points.
        """

        params = self.market_config.parameters_for(REGIME_ORDER[regime_index])
        z = np.asarray(aggressiveness, dtype=float)
        base = params.fill_intercept + self.market_config.fill_aggressiveness_coef * z
        # For side_sign = +1: win weight sigmoid(base - kappa * h); the
        # selection D = -epsilon = -scale * g with E[g | h] = rho * h.
        logits = base[:, None] - params.information_coefficient * self._nodes[None, :]
        win_weight = 1.0 / (1.0 + np.exp(-logits))
        numerator = (
            -params.residual_scale
            * params.information_strength
            * (win_weight * self._nodes[None, :]) @ self._weights
        )
        denominator = win_weight @ self._weights
        return numerator / np.maximum(denominator, 1e-12)

    def event_fill_probability(
        self,
        event: RFQEvent,
        aggressiveness: np.ndarray,
    ) -> np.ndarray:
        """True per-event fill probability using the hidden client signal.

        :param event: RFQ event.
        :param aggressiveness: Candidate normalized aggressiveness values.
        :return: Fill probabilities per candidate.
        """

        params = self.market_config.parameters_for(event.regime)
        z = np.asarray(aggressiveness, dtype=float)
        logits = (
            params.fill_intercept
            + self.market_config.fill_aggressiveness_coef * z
            - params.information_coefficient
            * float(event.side_sign)
            * event.hidden_client_signal
            - self.market_config.size_win_penalty * float(event.size - 1)
        )
        return 1.0 / (1.0 + np.exp(-logits))

    def event_post_win_value(
        self,
        event: RFQEvent,
        aggressiveness: np.ndarray,
    ) -> np.ndarray:
        """True per-event post-win clean value in points.

        Conditional on the hidden signal h, winning carries no further
        information, so m_true = cp + residual_scale * rho * h for every
        candidate quote.

        :param event: RFQ event.
        :param aggressiveness: Candidate values (fixes the output length).
        :return: Post-win clean values per candidate, in points.
        """

        params = self.market_config.parameters_for(event.regime)
        conditional_residual = (
            params.residual_scale
            * params.information_strength
            * event.hidden_client_signal
        )
        z = np.asarray(aggressiveness, dtype=float)
        return np.full(z.shape, event.cp_plus + conditional_residual)


def oracle_regime_selection_table(
    oracle: OracleControlModels,
    aggressiveness: np.ndarray,
) -> dict[MarketRegime, np.ndarray]:
    """Tabulate marginal true selection per regime for diagnostics.

    :param oracle: Oracle models.
    :param aggressiveness: Candidate normalized aggressiveness values.
    :return: Mapping from regime to selection in points.
    """

    return {
        regime: oracle.selection_points(regime.value, aggressiveness)
        for regime in REGIME_ORDER
    }
