"""Synthetic-oracle diagnostics computed from the known data-generating process.

Everything in this module reads ``latent_*`` columns and the simulator
configuration. It exists only for simulation diagnostics and policy
evaluation. Fitted models must never receive these quantities; the feature
layer independently rejects any ``latent_*`` column.

The oracle inverts the simulated win logit per RFQ, then shifts it with the
known aggressiveness coefficient to obtain the exact counterfactual fill
probability at any candidate quote. Conditional post-win values have no
closed form under Student-t noise, so they use seeded Monte Carlo
integration over the joint draw of future residual and client information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rfq_edge.config import OptimizerConfig
from rfq_edge.costs import (
    points_to_cents,
    rfq_inventory_value_points,
    rfq_trading_cost_points,
)
from rfq_edge.synthetic import (
    CLIENT_INFO_NOISE_STD,
    FUTURE_NOISE_SCALE,
    LATENT_COLUMNS,
    MARKOUT_SCALE_BY_REGIME,
    SyntheticConfig,
)

PROBABILITY_FLOOR = 1e-9
DEFAULT_ORACLE_DRAWS = 2_000
DEFAULT_ORACLE_SEED = 20_260_818


@dataclass(frozen=True)
class OracleContext:
    """Frozen view of the data-generating process used by oracle functions.

    :param config: Simulator configuration used to generate the dataset.
    :param client_information_mean: Population mean of latent client information.
    :param client_information_std: Population std of latent client information.
    """

    config: SyntheticConfig
    client_information_mean: float
    client_information_std: float


def build_oracle_context(
    latent_df: pd.DataFrame,
    config: SyntheticConfig,
) -> OracleContext:
    """Build the oracle context from the full latent dataset.

    The simulator standardizes client information across the whole generated
    population, so the same constants must be reused for counterfactuals.

    :param latent_df: Full dataset generated with ``include_latent=True``.
    :param config: Simulator configuration used to generate the dataset.
    :return: Context for all oracle computations.
    :raises ValueError: If latent columns are missing.
    """

    _require_latent_columns(latent_df)
    information = latent_df["latent_client_information"].astype(float)
    std = float(information.std(ddof=0))
    if std <= 0.0:
        raise ValueError("latent client information has zero variance")
    return OracleContext(
        config=config,
        client_information_mean=float(information.mean()),
        client_information_std=std,
    )


def oracle_fill_probability(
    df: pd.DataFrame,
    quote: pd.Series | float,
    context: OracleContext,
) -> pd.Series:
    """Return the exact true fill probability at a candidate quote.

    The simulated win logit is linear in aggressiveness, so the counterfactual
    probability is the historical logit shifted by the known coefficient.

    :param df: Latent RFQ rows.
    :param quote: Candidate clean quote as a scalar or aligned series.
    :param context: Oracle context.
    :return: True win probabilities aligned to ``df.index``.
    """

    _require_latent_columns(df)
    historical_logit = _logit(df["latent_p_win"].astype(float))
    new_aggressiveness = _aggressiveness_of(df, quote)
    shift = context.config.win_aggressiveness_coef * (
        new_aggressiveness - df["latent_aggressiveness"].astype(float)
    )
    probabilities = _sigmoid(historical_logit + shift)
    return pd.Series(probabilities, index=df.index, name="oracle_p_win")


def oracle_post_win_value(
    df: pd.DataFrame,
    quote: pd.Series | float,
    context: OracleContext,
    n_draws: int = DEFAULT_ORACLE_DRAWS,
    random_state: int = DEFAULT_ORACLE_SEED,
) -> pd.Series:
    """Approximate the true post-win clean value E[y5 | win at q, X].

    Monte Carlo integrates over the joint draw of future residual and client
    information. The same seed yields common random numbers across candidate
    quotes and responders, which reduces comparison noise.

    :param df: Latent RFQ rows.
    :param quote: Candidate clean quote as a scalar or aligned series.
    :param context: Oracle context.
    :param n_draws: Monte Carlo draws per RFQ.
    :param random_state: Seed for reproducible integration.
    :return: True conditional clean values aligned to ``df.index``.
    """

    conditional_residual = _conditional_residual_given_win(
        df=df,
        quote=quote,
        context=context,
        n_draws=n_draws,
        random_state=random_state,
    )
    values = df["cp_plus"].astype(float) + conditional_residual
    return values.rename("oracle_post_win_value")


def oracle_selection(
    df: pd.DataFrame,
    quote: pd.Series | float,
    context: OracleContext,
    n_draws: int = DEFAULT_ORACLE_DRAWS,
    random_state: int = DEFAULT_ORACLE_SEED,
) -> pd.Series:
    """Approximate true adverse selection at a candidate quote.

    Selection compares the unconditional true value E[y5 | X] with the
    win-conditional value, signed so that positive means the dealer loses.

    :param df: Latent RFQ rows.
    :param quote: Candidate clean quote as a scalar or aligned series.
    :param context: Oracle context.
    :param n_draws: Monte Carlo draws per RFQ.
    :param random_state: Seed for reproducible integration.
    :return: True adverse selection aligned to ``df.index``.
    """

    conditional_residual = _conditional_residual_given_win(
        df=df,
        quote=quote,
        context=context,
        n_draws=n_draws,
        random_state=random_state,
    )
    unconditional_residual = df["latent_mu_value"].astype(float)
    selection = df["side_sign"].astype(float) * (
        unconditional_residual - conditional_residual
    )
    return selection.rename("oracle_selection")


def oracle_conditional_edge(
    df: pd.DataFrame,
    quote: pd.Series | float,
    context: OracleContext,
    n_draws: int = DEFAULT_ORACLE_DRAWS,
    random_state: int = DEFAULT_ORACLE_SEED,
) -> pd.Series:
    """Approximate the true conditional clean edge at a candidate quote.

    :param df: Latent RFQ rows.
    :param quote: Candidate clean quote as a scalar or aligned series.
    :param context: Oracle context.
    :param n_draws: Monte Carlo draws per RFQ.
    :param random_state: Seed for reproducible integration.
    :return: True conditional edge in price points, aligned to ``df.index``.
    """

    post_win_value = oracle_post_win_value(
        df=df,
        quote=quote,
        context=context,
        n_draws=n_draws,
        random_state=random_state,
    )
    quote_series = _as_series(df, quote)
    edge = df["side_sign"].astype(float) * (post_win_value - quote_series)
    return edge.rename("oracle_conditional_edge")


def oracle_expected_objective(
    df: pd.DataFrame,
    quote: pd.Series | float,
    context: OracleContext,
    optimizer_config: OptimizerConfig | None = None,
    n_draws: int = DEFAULT_ORACLE_DRAWS,
    random_state: int = DEFAULT_ORACLE_SEED,
) -> pd.Series:
    """Approximate the true expected objective J(q, X) in cents.

    Uses the same cost and inventory-value functions as the responders so the
    only difference from a fitted policy is the truth of p and m.

    :param df: Latent RFQ rows.
    :param quote: Candidate clean quote as a scalar or aligned series.
    :param context: Oracle context.
    :param optimizer_config: Cost and inventory calibration.
    :param n_draws: Monte Carlo draws per RFQ.
    :param random_state: Seed for reproducible integration.
    :return: True expected objective in cents, aligned to ``df.index``.
    """

    config = optimizer_config or OptimizerConfig()
    p_win = oracle_fill_probability(df, quote, context)
    edge_points = oracle_conditional_edge(
        df=df,
        quote=quote,
        context=context,
        n_draws=n_draws,
        random_state=random_state,
    )
    objective = np.empty(len(df), dtype=float)
    for position, (_, row) in enumerate(df.iterrows()):
        cost_cents = points_to_cents(rfq_trading_cost_points(row, config))
        inventory_cents = points_to_cents(rfq_inventory_value_points(row, config))
        edge_cents = points_to_cents(float(edge_points.iloc[position]))
        objective[position] = float(p_win.iloc[position]) * (
            edge_cents - cost_cents + inventory_cents
        )
    return pd.Series(objective, index=df.index, name="oracle_expected_objective")


def oracle_optimal_quote(
    rfq: pd.DataFrame,
    context: OracleContext,
    optimizer_config: OptimizerConfig | None = None,
    n_draws: int = DEFAULT_ORACLE_DRAWS,
    random_state: int = DEFAULT_ORACLE_SEED,
) -> pd.DataFrame:
    """Scan the candidate grid for one RFQ with oracle quantities.

    :param rfq: Single latent RFQ row as a one-row dataframe.
    :param context: Oracle context.
    :param optimizer_config: Grid, cost, and inventory calibration.
    :param n_draws: Monte Carlo draws per candidate quote.
    :param random_state: Seed for reproducible integration.
    :return: Grid table with oracle p, m, edge, and objective per candidate.
    :raises ValueError: If ``rfq`` does not contain exactly one row.
    """

    if len(rfq) != 1:
        raise ValueError("oracle_optimal_quote expects exactly one RFQ row")
    config = optimizer_config or OptimizerConfig()
    row = rfq.iloc[0]
    grid = np.arange(
        config.min_aggressiveness,
        config.max_aggressiveness + config.aggressiveness_step / 2.0,
        config.aggressiveness_step,
    )
    records: list[dict[str, float]] = []
    for aggressiveness in grid:
        quote = float(
            row["cp_plus"]
            + float(row["side_sign"]) * float(aggressiveness) * float(row["market_width"])
        )
        p_win = float(oracle_fill_probability(rfq, quote, context).iloc[0])
        post_win = float(
            oracle_post_win_value(
                rfq, quote, context, n_draws=n_draws, random_state=random_state
            ).iloc[0]
        )
        edge_points = float(row["side_sign"]) * (post_win - quote)
        cost_cents = points_to_cents(rfq_trading_cost_points(row, config))
        inventory_cents = points_to_cents(rfq_inventory_value_points(row, config))
        objective = p_win * (points_to_cents(edge_points) - cost_cents + inventory_cents)
        records.append(
            {
                "quote": quote,
                "aggressiveness": float(aggressiveness),
                "oracle_p_win": p_win,
                "oracle_post_win_value": post_win,
                "oracle_edge_cents": points_to_cents(edge_points),
                "oracle_expected_objective_cents": objective,
            }
        )
    return pd.DataFrame(records)


def append_oracle_objective(
    comparison: pd.DataFrame,
    rfq: pd.DataFrame,
    context: OracleContext,
    optimizer_config: OptimizerConfig | None = None,
    n_draws: int = DEFAULT_ORACLE_DRAWS,
    random_state: int = DEFAULT_ORACLE_SEED,
) -> pd.DataFrame:
    """Attach the oracle expected objective to each responder's decision.

    Declined responders receive an oracle objective of zero, matching the
    value of the decline option.

    :param comparison: Output of ``compare_responders`` for one RFQ.
    :param rfq: The same single latent RFQ row.
    :param context: Oracle context.
    :param optimizer_config: Cost and inventory calibration.
    :param n_draws: Monte Carlo draws per selected quote.
    :param random_state: Seed for reproducible integration.
    :return: Copy of ``comparison`` with ``oracle_expected_objective_cents``.
    :raises ValueError: If ``rfq`` does not contain exactly one row.
    """

    if len(rfq) != 1:
        raise ValueError("append_oracle_objective expects exactly one RFQ row")
    augmented = comparison.copy()
    oracle_values: list[float] = []
    for _, decision in augmented.iterrows():
        if not bool(decision["accepted"]) or pd.isna(decision["quote"]):
            oracle_values.append(0.0)
            continue
        objective = oracle_expected_objective(
            rfq,
            float(decision["quote"]),
            context,
            optimizer_config=optimizer_config,
            n_draws=n_draws,
            random_state=random_state,
        )
        oracle_values.append(float(objective.iloc[0]))
    augmented["oracle_expected_objective_cents"] = oracle_values
    return augmented


def oracle_best_decision(oracle_grid: pd.DataFrame) -> dict[str, float | bool]:
    """Summarize the oracle-optimal action from an oracle grid table.

    :param oracle_grid: Output of :func:`oracle_optimal_quote`.
    :return: Best quote, aggressiveness, objective, and respond/decline flag.
    :raises ValueError: If the grid is empty.
    """

    if oracle_grid.empty:
        raise ValueError("oracle grid must not be empty")
    best = oracle_grid.loc[oracle_grid["oracle_expected_objective_cents"].idxmax()]
    accepted = bool(best["oracle_expected_objective_cents"] > 0.0)
    return {
        "accepted": accepted,
        "quote": float(best["quote"]) if accepted else float("nan"),
        "aggressiveness": float(best["aggressiveness"]) if accepted else float("nan"),
        "oracle_p_win": float(best["oracle_p_win"]) if accepted else 0.0,
        "oracle_expected_objective_cents": float(best["oracle_expected_objective_cents"]),
    }


def win_rate_by_aggressiveness_bucket(df: pd.DataFrame, n_buckets: int = 8) -> pd.DataFrame:
    """Summarize empirical win rate by historical aggressiveness bucket.

    :param df: RFQ dataframe with observed quotes and outcomes.
    :param n_buckets: Number of quantile buckets.
    :return: Bucket table with mean aggressiveness and win rate.
    """

    aggressiveness = (
        df["side_sign"].astype(float)
        * (df["quote"].astype(float) - df["cp_plus"].astype(float))
        / df["market_width"].astype(float)
    )
    frame = pd.DataFrame({"aggressiveness": aggressiveness, "won": df["won"].astype(float)})
    frame["bucket"] = pd.qcut(frame["aggressiveness"], q=n_buckets, duplicates="drop")
    grouped = frame.groupby("bucket", observed=False).agg(
        mean_aggressiveness=("aggressiveness", "mean"),
        win_rate=("won", "mean"),
        count=("won", "size"),
    )
    return grouped.reset_index(drop=True)


def post_win_tilt_by_side(df: pd.DataFrame) -> pd.DataFrame:
    """Show that winning selects different future outcomes per dealer side.

    Uses the realized future residual (y5 - CP+), so this is an empirical
    diagnostic rather than an oracle counterfactual.

    :param df: RFQ dataframe with realized outcomes.
    :return: Mean future residual for wins and for all RFQs, per side.
    """

    residual = df["y5"].astype(float) - df["cp_plus"].astype(float)
    frame = pd.DataFrame(
        {
            "side": df["side"].astype(str),
            "residual": residual,
            "won": df["won"].astype(bool),
        }
    )
    records: list[dict[str, float | str]] = []
    for side, group in frame.groupby("side"):
        records.append(
            {
                "side": side,
                "mean_residual_all": float(group["residual"].mean()),
                "mean_residual_wins": float(group.loc[group["won"], "residual"].mean()),
                "n_rfqs": int(len(group)),
                "n_wins": int(group["won"].sum()),
            }
        )
    return pd.DataFrame(records)


def realized_selection_summary(df: pd.DataFrame) -> dict[str, float]:
    """Compare realized selection among wins with the full population.

    Realized selection is measured against CP+ as the pre-trade anchor:
    ``side_sign * (cp_plus - y5)``, so positive means the dealer lost value.

    :param df: RFQ dataframe with realized outcomes.
    :return: Mean selection for all RFQs and for wins, plus the gap.
    """

    selection = df["side_sign"].astype(float) * (
        df["cp_plus"].astype(float) - df["y5"].astype(float)
    )
    mean_all = float(selection.mean())
    mean_wins = float(selection.loc[df["won"].astype(bool)].mean())
    return {
        "mean_selection_all": mean_all,
        "mean_selection_wins": mean_wins,
        "selection_gap": mean_wins - mean_all,
    }


def _conditional_residual_given_win(
    df: pd.DataFrame,
    quote: pd.Series | float,
    context: OracleContext,
    n_draws: int,
    random_state: int,
) -> pd.Series:
    """Monte Carlo estimate of E[future residual | win at quote, X]."""

    _require_latent_columns(df)
    config = context.config
    rng = np.random.default_rng(random_state)
    n_rows = len(df)

    mu = df["latent_mu_value"].astype(float).to_numpy()[:, None]
    strength = df["latent_information_strength"].astype(float).to_numpy()[:, None]
    side = df["side_sign"].astype(float).to_numpy()[:, None]
    regime = df["regime"].astype(str).to_numpy()
    markout_scale = np.array(
        [MARKOUT_SCALE_BY_REGIME[label] for label in regime], dtype=float
    )[:, None]

    t_scale = np.sqrt((config.student_t_df - 2.0) / config.student_t_df)
    noise = (
        rng.standard_t(config.student_t_df, size=(n_rows, n_draws))
        * t_scale
        * FUTURE_NOISE_SCALE
        * markout_scale
    )
    residual_draws = mu + noise
    information_draws = strength * residual_draws + rng.normal(
        0.0, CLIENT_INFO_NOISE_STD, size=(n_rows, n_draws)
    )
    standardized_information = (
        information_draws - context.client_information_mean
    ) / context.client_information_std

    offset = _logit_offset(df, context)[:, None]
    new_aggressiveness = _aggressiveness_of(df, quote).to_numpy()[:, None]
    logits = (
        offset
        + config.win_aggressiveness_coef * new_aggressiveness
        - config.win_information_coef * side * standardized_information
    )
    win_probabilities = _sigmoid(logits)

    weight_sums = win_probabilities.sum(axis=1)
    weight_sums = np.maximum(weight_sums, PROBABILITY_FLOOR)
    conditional = (residual_draws * win_probabilities).sum(axis=1) / weight_sums
    return pd.Series(conditional, index=df.index, name="oracle_conditional_residual")


def _logit_offset(df: pd.DataFrame, context: OracleContext) -> np.ndarray:
    """Recover the quote- and information-free part of each row's win logit."""

    config = context.config
    historical_logit = _logit(df["latent_p_win"].astype(float)).to_numpy()
    historical_aggressiveness = df["latent_aggressiveness"].astype(float).to_numpy()
    standardized_information = (
        df["latent_client_information"].astype(float).to_numpy()
        - context.client_information_mean
    ) / context.client_information_std
    side = df["side_sign"].astype(float).to_numpy()
    return (
        historical_logit
        - config.win_aggressiveness_coef * historical_aggressiveness
        + config.win_information_coef * side * standardized_information
    )


def _aggressiveness_of(df: pd.DataFrame, quote: pd.Series | float) -> pd.Series:
    quote_series = _as_series(df, quote)
    return (
        df["side_sign"].astype(float)
        * (quote_series - df["cp_plus"].astype(float))
        / df["market_width"].astype(float)
    )


def _as_series(df: pd.DataFrame, quote: pd.Series | float) -> pd.Series:
    if isinstance(quote, pd.Series):
        return quote.astype(float)
    return pd.Series(float(quote), index=df.index)


def _require_latent_columns(df: pd.DataFrame) -> None:
    missing = [column for column in LATENT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "oracle diagnostics require latent columns; generate data with "
            f"include_latent=True (missing: {missing})"
        )


def _logit(probabilities: pd.Series) -> pd.Series:
    clipped = probabilities.clip(PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray | pd.Series) -> np.ndarray | pd.Series:
    return 1.0 / (1.0 + np.exp(-values))
