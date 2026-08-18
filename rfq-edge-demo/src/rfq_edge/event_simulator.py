"""Forward event simulator for the control environment.

Event ordering within each step — identical in the Bellman recursion, this
simulator, the tests, and the notebook:

1. Observe inventory, regime, and time remaining.
2. Observe whether an RFQ arrived and its state.
3. If an RFQ arrived, choose quote or decline (RFQ jump operator).
4. Apply the fill or no-fill inventory transition.
5. Choose active execution, if permitted.
6. Apply the running inventory penalty on end-of-step inventory.
7. Transition to the next regime and time step.

Fill realizations compare the pre-drawn uniform of the step against the true
fill probability at the chosen quote, so different policies face identical
exogenous paths (common random numbers) and differ only through decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from rfq_edge.control_config import (
    ControlMarketConfig,
    EpisodeConfig,
    REGIME_ORDER,
)
from rfq_edge.control_state import (
    ActionType,
    ControlAction,
    ControlState,
    RFQEvent,
    apply_rfq_fill,
    classify_economic_mode,
    target_shortfall,
)
from rfq_edge.dynamic_objective import realized_rfq_reward_cents
from rfq_edge.execution_costs import (
    active_execution_cost_cents,
    running_inventory_penalty_cents,
    terminal_penalty_cents,
)
from rfq_edge.market_dynamics import ExogenousPath, true_fill_probability


class ControlPolicy(Protocol):
    """Interface every control policy implements."""

    name: str

    def respond_to_rfq(self, state: ControlState, event: RFQEvent) -> ControlAction:
        """Quote or decline an incoming RFQ."""

    def choose_active_execution(self, state: ControlState) -> ControlAction:
        """Choose the signed active execution amount (0 means wait)."""


@dataclass(frozen=True)
class EpisodeResult:
    """Outcome of one simulated episode.

    :param log: One row per time step with the full event record.
    :param total_reward_cents: Sum of all reward components, in cents.
    :param terminal_inventory: Inventory at the horizon.
    :param terminal_penalty_cents: Terminal shortfall penalty, in cents.
    """

    log: pd.DataFrame
    total_reward_cents: float
    terminal_inventory: int
    terminal_penalty_cents: float


def simulate_episode(
    policy: ControlPolicy,
    episode_config: EpisodeConfig,
    market_config: ControlMarketConfig,
    path: ExogenousPath,
) -> EpisodeResult:
    """Run one policy over one pre-drawn exogenous path.

    :param policy: Control policy.
    :param episode_config: Episode configuration.
    :param market_config: Market configuration.
    :param path: Exogenous path shared across policies for this episode.
    :return: Episode result with the detailed event log.
    :raises ValueError: If the policy violates the inventory limit.
    """

    inventory = episode_config.initial_inventory
    cumulative_reward = 0.0
    records: list[dict[str, object]] = []

    for step in range(episode_config.n_steps):
        regime = REGIME_ORDER[int(path.regimes[step])]
        regime_params = market_config.parameters_for(regime)
        event = path.events[step]
        inventory_before = inventory

        state = ControlState(
            time_index=step,
            time_remaining=episode_config.n_steps - step,
            market_regime=regime,
            inventory=inventory,
            target_inventory=episode_config.target_inventory,
            initial_inventory=episode_config.initial_inventory,
            inventory_limit=episode_config.inventory_limit,
            current_cp_plus=float(path.cp_plus[step]),
            volatility=regime_params.volatility,
            liquidity_score=regime_params.liquidity_score,
            market_width=regime_params.market_width,
            active_execution_available=episode_config.active_execution_allowed,
            current_rfq=event,
        )

        rfq_action: ControlAction | None = None
        filled = False
        oracle_p_win = float("nan")
        rfq_reward_realized = 0.0
        rfq_reward_expected = float("nan")
        if event is not None:
            rfq_action = policy.respond_to_rfq(state, event)
            if rfq_action.action_type is ActionType.QUOTE_RFQ:
                _assert_fill_within_limit(inventory, event, episode_config)
                assert rfq_action.normalized_aggressiveness is not None
                assert rfq_action.quote is not None
                oracle_p_win = true_fill_probability(
                    market_config, event, rfq_action.normalized_aggressiveness
                )
                filled = bool(path.fill_uniforms[step] < oracle_p_win)
                if filled:
                    rfq_reward_realized = realized_rfq_reward_cents(
                        quote=rfq_action.quote,
                        cp_plus=event.cp_plus,
                        future_residual=event.hidden_future_residual,
                        side_sign=event.side_sign,
                        size=event.size,
                        transaction_cost_cents=market_config.rfq_transaction_cost_cents,
                    )
                rfq_reward_expected = float(
                    rfq_action.predicted_p_win
                    * float(event.size)
                    * (
                        float(event.side_sign)
                        * (rfq_action.predicted_post_win_value - rfq_action.quote)
                        * 100.0
                        - market_config.rfq_transaction_cost_cents
                    )
                )

        inventory_after_rfq = (
            apply_rfq_fill(inventory, event.side_sign, event.size)
            if (event is not None and filled)
            else inventory
        )
        _assert_within_limit(inventory_after_rfq, episode_config)

        active_amount = 0
        active_cost = 0.0
        if episode_config.active_execution_allowed:
            active_state = ControlState(
                time_index=step,
                time_remaining=episode_config.n_steps - step,
                market_regime=regime,
                inventory=inventory_after_rfq,
                target_inventory=episode_config.target_inventory,
                initial_inventory=episode_config.initial_inventory,
                inventory_limit=episode_config.inventory_limit,
                current_cp_plus=float(path.cp_plus[step]),
                volatility=regime_params.volatility,
                liquidity_score=regime_params.liquidity_score,
                market_width=regime_params.market_width,
                active_execution_available=True,
                current_rfq=None,
            )
            active_action = policy.choose_active_execution(active_state)
            active_amount = int(active_action.active_execution_amount)
            if active_amount != 0:
                active_cost = active_execution_cost_cents(active_amount, regime_params)

        inventory_after = inventory_after_rfq + active_amount
        _assert_within_limit(inventory_after, episode_config)

        running_penalty = running_inventory_penalty_cents(inventory_after, episode_config)
        cumulative_reward += rfq_reward_realized - active_cost - running_penalty

        mode = classify_economic_mode(
            inventory_before=inventory_before,
            target_inventory=episode_config.target_inventory,
            rfq_action=rfq_action,
            rfq_side_sign=event.side_sign if event is not None else None,
            rfq_size=event.size if event is not None else None,
            active_amount=active_amount,
        )
        records.append(
            {
                "time_index": step,
                "regime": regime.name,
                "inventory_before": inventory_before,
                "target_inventory": episode_config.target_inventory,
                "time_remaining": episode_config.n_steps - step,
                "rfq_arrived": event is not None,
                "rfq_side": event.side if event is not None else "",
                "rfq_side_sign": event.side_sign if event is not None else 0,
                "rfq_size": event.size if event is not None else 0,
                "action": _primary_action_label(rfq_action, active_amount, event),
                "quote": rfq_action.quote if rfq_action is not None else float("nan"),
                "aggressiveness": (
                    rfq_action.normalized_aggressiveness
                    if rfq_action is not None
                    else float("nan")
                ),
                "predicted_p_win": (
                    rfq_action.predicted_p_win if rfq_action is not None else float("nan")
                ),
                "predicted_selection_points": (
                    rfq_action.predicted_selection if rfq_action is not None else float("nan")
                ),
                "predicted_post_win_value": (
                    rfq_action.predicted_post_win_value
                    if rfq_action is not None
                    else float("nan")
                ),
                "oracle_p_win": oracle_p_win,
                "filled": filled,
                "inventory_after_rfq": inventory_after_rfq,
                "active_execution_amount": active_amount,
                "inventory_after": inventory_after,
                "rfq_reward_cents": rfq_reward_realized,
                "rfq_reward_expected_cents": rfq_reward_expected,
                "active_execution_cost_cents": active_cost,
                "running_inventory_penalty_cents": running_penalty,
                "cumulative_reward_cents": cumulative_reward,
                "remaining_target_shortfall": target_shortfall(
                    inventory_after, episode_config.target_inventory
                ),
                "economic_mode": mode.value,
                "policy": policy.name,
            }
        )
        inventory = inventory_after

    final_penalty = terminal_penalty_cents(inventory, episode_config)
    cumulative_reward -= final_penalty
    log = pd.DataFrame(records)
    log["terminal_penalty_cents"] = 0.0
    log.loc[log.index[-1], "terminal_penalty_cents"] = final_penalty
    log.loc[log.index[-1], "cumulative_reward_cents"] = cumulative_reward
    return EpisodeResult(
        log=log,
        total_reward_cents=cumulative_reward,
        terminal_inventory=inventory,
        terminal_penalty_cents=final_penalty,
    )


def _primary_action_label(
    rfq_action: ControlAction | None,
    active_amount: int,
    event: RFQEvent | None,
) -> str:
    if rfq_action is not None and rfq_action.action_type is ActionType.QUOTE_RFQ:
        rfq_label = ActionType.QUOTE_RFQ.value
    elif event is not None:
        rfq_label = ActionType.DECLINE_RFQ.value
    else:
        rfq_label = ActionType.WAIT.value
    if active_amount != 0:
        return f"{rfq_label}+{ActionType.ACTIVE_EXECUTION.value}"
    return rfq_label


def _assert_within_limit(inventory: int, config: EpisodeConfig) -> None:
    if abs(inventory) > config.inventory_limit:
        raise ValueError(
            f"inventory {inventory} violates the limit {config.inventory_limit}"
        )


def _assert_fill_within_limit(
    inventory: int,
    event: RFQEvent,
    config: EpisodeConfig,
) -> None:
    filled_inventory = apply_rfq_fill(inventory, event.side_sign, event.size)
    if abs(filled_inventory) > config.inventory_limit:
        raise ValueError(
            "policy quoted an RFQ whose fill would violate the inventory limit"
        )
