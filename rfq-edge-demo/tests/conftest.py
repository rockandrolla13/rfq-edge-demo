"""Shared fixtures for responder, policy-evaluation, and plot tests."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest

from rfq_edge.config import (
    FillModelConfig,
    OptimizerConfig,
    SelectionModelConfig,
    ValueModelConfig,
)
from rfq_edge.pipeline import FittedFramework, fit_framework
from rfq_edge.policy_evaluation import PolicyEvaluationResult, evaluate_policies
from rfq_edge.simulation_diagnostics import OracleContext, build_oracle_context
from rfq_edge.synthetic import SyntheticConfig, make_synthetic_rfqs

DEMO_SYNTHETIC_CONFIG = SyntheticConfig(n_rfqs=2_000, n_bonds=80, n_issuers=20)
DEMO_SEED = 42


@pytest.fixture(scope="session")
def demo_frame():
    """Small latent-inclusive dataset shared across the new test modules."""

    return make_synthetic_rfqs(
        config=DEMO_SYNTHETIC_CONFIG,
        random_state=DEMO_SEED,
        include_latent=True,
    )


@pytest.fixture(scope="session")
def demo_framework(demo_frame) -> FittedFramework:
    """Framework fitted once per session on the shared dataset."""

    return fit_framework(
        demo_frame,
        value_config=ValueModelConfig(number_of_oof_splits=3),
        fill_config=FillModelConfig(number_of_cv_splits=3),
        selection_config=SelectionModelConfig(number_of_cv_splits=3),
    )


@pytest.fixture(scope="session")
def demo_oracle_context(demo_frame) -> OracleContext:
    """Oracle context matching the shared dataset."""

    return build_oracle_context(demo_frame, DEMO_SYNTHETIC_CONFIG)


@pytest.fixture(scope="session")
def demo_policy_result(demo_framework, demo_oracle_context) -> PolicyEvaluationResult:
    """One held-out policy comparison reused by summary and plot tests."""

    return evaluate_policies(
        test_df=demo_framework.test_df,
        models=demo_framework.models,
        oracle_context=demo_oracle_context,
        optimizer_config=OptimizerConfig(aggressiveness_step=0.5),
        random_state=7,
        bootstrap_samples=50,
    )
