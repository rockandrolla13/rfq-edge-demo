"""End-to-end assembly of the control stack.

Builds, in order: the training history, the fitted control models, the
oracle models, and per-episode Bellman solutions; then constructs the five
comparison policies for any episode configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from rfq_edge.bellman import BellmanSolution, solve_bellman
from rfq_edge.control_config import (
    ControlMarketConfig,
    EpisodeConfig,
    default_control_market,
    market_making_episode,
)
from rfq_edge.control_models import FittedControlModels, fit_control_models
from rfq_edge.controllers import (
    DynamicExecutionController,
    DynamicMarketMaker,
    EdgeConsistentMyopicResponder,
    OracleDynamicController,
    PlainResponder,
)
from rfq_edge.event_simulator import ControlPolicy
from rfq_edge.market_dynamics import generate_training_history
from rfq_edge.oracle_control import OracleControlModels

DEFAULT_TRAINING_EVENTS = 20_000

POLICY_ORDER: tuple[str, ...] = (
    "PlainResponder",
    "EdgeConsistentMyopic",
    "DynamicMarketMaker",
    "DynamicExecution",
    "OracleDynamic",
)


@dataclass(frozen=True)
class ControlArtifacts:
    """Everything needed to run and compare control policies.

    :param market_config: Market configuration.
    :param training_history: Observable history the models were fitted on.
    :param fitted_models: Fitted fill and selection models.
    :param oracle_models: Truth-based models for benchmarks and diagnostics.
    """

    market_config: ControlMarketConfig
    training_history: pd.DataFrame
    fitted_models: FittedControlModels
    oracle_models: OracleControlModels


def build_control_artifacts(
    market_config: ControlMarketConfig | None = None,
    n_training_events: int = DEFAULT_TRAINING_EVENTS,
    random_state: int = 0,
) -> ControlArtifacts:
    """Generate history, fit models, and build the oracle.

    :param market_config: Market configuration; defaults to the calibrated one.
    :param n_training_events: Historical RFQs used to fit the models.
    :param random_state: Seed for the training history.
    :return: Control artifacts.
    """

    config = market_config if market_config is not None else default_control_market()
    history = generate_training_history(
        market_config=config,
        n_events=n_training_events,
        random_state=random_state,
    )
    fitted = fit_control_models(history)
    oracle = OracleControlModels(config)
    return ControlArtifacts(
        market_config=config,
        training_history=history,
        fitted_models=fitted,
        oracle_models=oracle,
    )


def market_making_variant(episode_config: EpisodeConfig) -> EpisodeConfig:
    """Zero-target variant of an episode used by the DynamicMarketMaker.

    The market maker ignores any position target: it plans for target zero
    with the standard market-making penalties, whatever the episode asks for.

    :param episode_config: Episode being simulated.
    :return: Planning configuration with a zero-inventory objective.
    """

    reference = market_making_episode()
    return replace(
        episode_config,
        target_inventory=0,
        running_penalty_cents=reference.running_penalty_cents,
        terminal_penalty_cents=reference.terminal_penalty_cents,
    )


def solve_episode_policies(
    artifacts: ControlArtifacts,
    episode_config: EpisodeConfig,
) -> dict[str, BellmanSolution]:
    """Solve the Bellman problems needed by the dynamic policies.

    :param artifacts: Control artifacts.
    :param episode_config: Episode to plan for.
    :return: Solutions keyed by policy name.
    """

    return {
        "DynamicMarketMaker": solve_bellman(
            market_making_variant(episode_config),
            artifacts.market_config,
            artifacts.fitted_models,
        ),
        "DynamicExecution": solve_bellman(
            episode_config,
            artifacts.market_config,
            artifacts.fitted_models,
        ),
        "OracleDynamic": solve_bellman(
            episode_config,
            artifacts.market_config,
            artifacts.oracle_models,
        ),
    }


def make_policies(
    artifacts: ControlArtifacts,
    episode_config: EpisodeConfig,
    solutions: dict[str, BellmanSolution] | None = None,
) -> dict[str, ControlPolicy]:
    """Construct the five comparison policies for one episode configuration.

    :param artifacts: Control artifacts.
    :param episode_config: Episode to build policies for.
    :param solutions: Pre-solved Bellman solutions; solved here when omitted.
    :return: Policies keyed by name, in POLICY_ORDER.
    """

    solved = solutions if solutions is not None else solve_episode_policies(
        artifacts, episode_config
    )
    policies: dict[str, ControlPolicy] = {
        "PlainResponder": PlainResponder(
            artifacts.market_config, artifacts.fitted_models
        ),
        "EdgeConsistentMyopic": EdgeConsistentMyopicResponder(
            artifacts.market_config, episode_config, artifacts.fitted_models
        ),
        "DynamicMarketMaker": DynamicMarketMaker(
            artifacts.market_config,
            artifacts.fitted_models,
            solved["DynamicMarketMaker"],
        ),
        "DynamicExecution": DynamicExecutionController(
            artifacts.market_config,
            artifacts.fitted_models,
            solved["DynamicExecution"],
        ),
        "OracleDynamic": OracleDynamicController(
            artifacts.market_config,
            artifacts.oracle_models,
            solved["OracleDynamic"],
        ),
    }
    return policies
