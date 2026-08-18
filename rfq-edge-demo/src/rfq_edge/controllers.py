"""Control policies: plain, edge-consistent myopic, dynamic, and oracle.

All policies share the same candidate quote grid, costs, inventory limits,
and decline option. They differ only in the post-win value they assume and
in whether they use the Bellman continuation value:

* PlainResponder — m = CP+, myopic, never executes actively.
* EdgeConsistentMyopicResponder — fitted m(q, X), static per-trade inventory
  penalty, no continuation value, never executes actively.
* DynamicMarketMaker — fitted models plus Bellman continuation value with a
  zero-inventory objective; may execute actively.
* DynamicExecutionController — fitted models plus Bellman continuation value
  for the episode's target and deadline; may execute actively.
* OracleDynamicController — true synthetic p and m (hidden signal known)
  with the same action set, costs, and constraints; upper-bound benchmark.

Quotes map to prices via q = cp_plus + side_sign * z * market_width, so
z > 0 is always more aggressive (more likely to fill) on either side.
"""

from __future__ import annotations

import numpy as np

from rfq_edge.bellman import BellmanSolution
from rfq_edge.control_config import ControlMarketConfig, EpisodeConfig
from rfq_edge.control_models import FittedControlModels
from rfq_edge.control_state import ActionType, ControlAction, ControlState, RFQEvent
from rfq_edge.oracle_control import OracleControlModels

QuoteModels = FittedControlModels | OracleControlModels


def _wait(policy_name: str) -> ControlAction:
    return ControlAction(
        action_type=ActionType.WAIT,
        quote=None,
        normalized_aggressiveness=None,
        active_execution_amount=0,
        respond_or_decline=False,
        policy_name=policy_name,
    )


def _decline(policy_name: str) -> ControlAction:
    return ControlAction(
        action_type=ActionType.DECLINE_RFQ,
        quote=None,
        normalized_aggressiveness=None,
        active_execution_amount=0,
        respond_or_decline=False,
        policy_name=policy_name,
    )


def _quote_action(
    policy_name: str,
    event: RFQEvent,
    aggressiveness: float,
    p_win: float,
    selection_points: float,
    post_win_value: float,
) -> ControlAction:
    quote = event.cp_plus + event.side_sign * aggressiveness * event.market_width
    return ControlAction(
        action_type=ActionType.QUOTE_RFQ,
        quote=quote,
        normalized_aggressiveness=aggressiveness,
        active_execution_amount=0,
        respond_or_decline=True,
        policy_name=policy_name,
        predicted_p_win=p_win,
        predicted_selection=selection_points,
        predicted_post_win_value=post_win_value,
    )


def _fill_breaches_limit(state: ControlState, event: RFQEvent) -> bool:
    filled = state.inventory + event.side_sign * event.size
    return abs(filled) > state.inventory_limit


class PlainResponder:
    """Myopic responder that assumes CP+ is the post-win value."""

    name = "PlainResponder"

    def __init__(
        self,
        market_config: ControlMarketConfig,
        models: FittedControlModels,
    ) -> None:
        """:param market_config: Market configuration.
        :param models: Fitted models (used for fill probability only).
        """

        self._market_config = market_config
        self._models = models

    def respond_to_rfq(self, state: ControlState, event: RFQEvent) -> ControlAction:
        """Quote for standalone myopic edge with m = CP+, or decline.

        :param state: Control state.
        :param event: Incoming RFQ.
        :return: Quote or decline action.
        """

        if _fill_breaches_limit(state, event):
            return _decline(self.name)
        z_grid = np.asarray(self._market_config.aggressiveness_grid)
        p_win = self._models.event_fill_probability(event, z_grid)
        # m = CP+ means side_sign * (m - q) = -z * width in points.
        edge_cents = -z_grid * event.market_width * 100.0
        reward = float(event.size) * (
            edge_cents - self._market_config.rfq_transaction_cost_cents
        )
        objective = p_win * reward
        best = int(np.argmax(objective))
        if objective[best] <= 0.0:
            return _decline(self.name)
        return _quote_action(
            policy_name=self.name,
            event=event,
            aggressiveness=float(z_grid[best]),
            p_win=float(p_win[best]),
            selection_points=0.0,
            post_win_value=event.cp_plus,
        )

    def choose_active_execution(self, state: ControlState) -> ControlAction:
        """Never executes actively.

        :param state: Control state.
        :return: Wait action.
        """

        return _wait(self.name)


class EdgeConsistentMyopicResponder:
    """Myopic responder using fitted m(q, X) and a static inventory penalty."""

    name = "EdgeConsistentMyopic"

    def __init__(
        self,
        market_config: ControlMarketConfig,
        episode_config: EpisodeConfig,
        models: FittedControlModels,
    ) -> None:
        """:param market_config: Market configuration.
        :param episode_config: Provides the static inventory penalty and target.
        :param models: Fitted fill and selection models.
        """

        self._market_config = market_config
        self._episode_config = episode_config
        self._models = models

    def respond_to_rfq(self, state: ControlState, event: RFQEvent) -> ControlAction:
        """Quote for selection-adjusted edge with a static inventory penalty.

        The penalty charges kappa * [(I_fill - I*)^2 - (I - I*)^2] on a fill;
        it is a fixed configuration, not a continuation value.

        :param state: Control state.
        :param event: Incoming RFQ.
        :return: Quote or decline action.
        """

        if _fill_breaches_limit(state, event):
            return _decline(self.name)
        z_grid = np.asarray(self._market_config.aggressiveness_grid)
        p_win = self._models.event_fill_probability(event, z_grid)
        post_win_value = self._models.event_post_win_value(event, z_grid)
        quotes = event.cp_plus + event.side_sign * z_grid * event.market_width
        edge_cents = float(event.side_sign) * (post_win_value - quotes) * 100.0
        reward = float(event.size) * (
            edge_cents - self._market_config.rfq_transaction_cost_cents
        )
        target = self._episode_config.target_inventory
        filled_inventory = state.inventory + event.side_sign * event.size
        penalty = self._episode_config.myopic_inventory_penalty_cents * (
            float((filled_inventory - target) ** 2) - float((state.inventory - target) ** 2)
        )
        objective = p_win * (reward - penalty)
        best = int(np.argmax(objective))
        if objective[best] <= 0.0:
            return _decline(self.name)
        selection = self._models.selection_points(event.regime.value, z_grid)
        return _quote_action(
            policy_name=self.name,
            event=event,
            aggressiveness=float(z_grid[best]),
            p_win=float(p_win[best]),
            selection_points=float(selection[best]),
            post_win_value=float(post_win_value[best]),
        )

    def choose_active_execution(self, state: ControlState) -> ControlAction:
        """Never executes actively.

        :param state: Control state.
        :return: Wait action.
        """

        return _wait(self.name)


class _BellmanPolicy:
    """Shared machinery for controllers driven by a Bellman solution."""

    name = "BellmanPolicy"

    def __init__(
        self,
        market_config: ControlMarketConfig,
        models: QuoteModels,
        solution: BellmanSolution,
    ) -> None:
        """:param market_config: Market configuration.
        :param models: Quote models (fitted or oracle).
        :param solution: Bellman solution providing continuation values.
        """

        self._market_config = market_config
        self._models = models
        self._solution = solution

    def respond_to_rfq(self, state: ControlState, event: RFQEvent) -> ControlAction:
        """Maximize the RFQ jump increment; respond only if it is positive.

        RFQIncrement(q) = p(q, X) * [r_rfq + U_k(I_fill, r) - U_k(I, r)].

        :param state: Control state.
        :param event: Incoming RFQ.
        :return: Quote or decline action.
        """

        if _fill_breaches_limit(state, event):
            return _decline(self.name)
        solution = self._solution
        step = min(state.time_index, solution.episode_config.n_steps - 1)
        regime_index = state.market_regime.value
        z_grid = solution.aggressiveness_grid
        p_win = self._models.event_fill_probability(event, z_grid)
        post_win_value = self._models.event_post_win_value(event, z_grid)
        quotes = event.cp_plus + event.side_sign * z_grid * event.market_width
        edge_cents = float(event.side_sign) * (post_win_value - quotes) * 100.0
        reward = float(event.size) * (
            edge_cents - self._market_config.rfq_transaction_cost_cents
        )
        continuation = solution.post_rfq_value[step, :, regime_index]
        index_before = solution.inventory_index(state.inventory)
        index_after = solution.inventory_index(
            state.inventory + event.side_sign * event.size
        )
        continuation_delta = float(continuation[index_after] - continuation[index_before])
        increments = p_win * (reward + continuation_delta)
        best = int(np.argmax(increments))
        if increments[best] <= 0.0:
            return _decline(self.name)
        selection = float(
            float(event.side_sign) * (event.cp_plus - post_win_value[best])
        )
        return _quote_action(
            policy_name=self.name,
            event=event,
            aggressiveness=float(z_grid[best]),
            p_win=float(p_win[best]),
            selection_points=selection,
            post_win_value=float(post_win_value[best]),
        )

    def choose_active_execution(self, state: ControlState) -> ControlAction:
        """Look up the precomputed optimal active execution.

        :param state: Control state at the active stage.
        :return: Active execution or wait action.
        """

        if not state.active_execution_available:
            return _wait(self.name)
        solution = self._solution
        step = min(state.time_index, solution.episode_config.n_steps - 1)
        amount = int(
            solution.active_policy[
                step,
                solution.inventory_index(state.inventory),
                state.market_regime.value,
            ]
        )
        if amount == 0:
            return _wait(self.name)
        return ControlAction(
            action_type=ActionType.ACTIVE_EXECUTION,
            quote=None,
            normalized_aggressiveness=None,
            active_execution_amount=amount,
            respond_or_decline=False,
            policy_name=self.name,
        )


class DeclineAllRFQs(_BellmanPolicy):
    """Ablation policy: declines every RFQ, keeps dynamic active execution.

    Used only to attribute how much of the dynamic controller's value comes
    from internalizing RFQ flow versus trading actively. The Bellman
    solution passed in should be solved on a zero-arrival market so the
    active policy does not count on RFQ flow that will never be accepted.
    """

    name = "DynamicNoRFQ"

    def respond_to_rfq(self, state: ControlState, event: RFQEvent) -> ControlAction:
        """Always decline.

        :param state: Control state.
        :param event: Incoming RFQ.
        :return: Decline action.
        """

        return _decline(self.name)


class DynamicMarketMaker(_BellmanPolicy):
    """Dynamic controller with a zero-inventory market-making objective."""

    name = "DynamicMarketMaker"


class DynamicExecutionController(_BellmanPolicy):
    """Dynamic controller for the episode's target inventory and deadline."""

    name = "DynamicExecution"


class OracleDynamicController(_BellmanPolicy):
    """Dynamic controller using true synthetic p and m (benchmark only)."""

    name = "OracleDynamic"

    def __init__(
        self,
        market_config: ControlMarketConfig,
        models: OracleControlModels,
        solution: BellmanSolution,
    ) -> None:
        """:param market_config: Market configuration.
        :param models: Oracle models; fitted models are never accepted.
        :param solution: Bellman solution built from oracle marginals.
        :raises TypeError: If fitted models are passed in.
        """

        if not isinstance(models, OracleControlModels):
            raise TypeError("OracleDynamicController requires oracle models")
        super().__init__(market_config, models, solution)
