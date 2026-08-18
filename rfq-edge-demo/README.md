# rfq-edge: an edge-consistent RFQ responder demo

A simulation study of corporate-bond RFQ quoting. It shows why a responder should
price the *conditional* value of winning — "if I win because I quoted `q`, what is
the bond worth?" — instead of the unconditional pre-trade value, and when that
distinction changes quote decisions.

## Layout

- `src/rfq_edge/synthetic.py` — seeded synthetic RFQ market with realistic
  structure (clients, venues, dealer competition, regimes, staleness, axes) and
  a hidden adverse-selection mechanism stored in `latent_*` columns.
- `src/rfq_edge/features.py` — point-in-time feature contracts per model; any
  attempt to feed latent or outcome columns to a model raises.
- `src/rfq_edge/value_model.py`, `fill_model.py`, `selection_model.py` — pooled
  V0 forecast, counterfactual fill probability `p(q, X)`, and adverse selection
  `A(q, X)` fitted on out-of-fold targets.
- `src/rfq_edge/costs.py`, `optimizer.py`, `responders.py` — costs, inventory
  values, quote-support restriction, and the three responders (plain CP+,
  plain V0, edge-consistent), all sharing one grid and decline option.
- `src/rfq_edge/simulation_diagnostics.py` — synthetic oracle (exact
  counterfactual fill probability, Monte Carlo conditional values), used only
  for diagnostics and evaluation, never for fitting.
- `src/rfq_edge/policy_evaluation.py` — held-out policy comparison with common
  random numbers, date-block bootstrap intervals, and sensitivity scenarios.
- `src/rfq_edge/plots.py` — all figures used by the first notebook.
- `notebooks/01_edge_consistent_responder_demo.ipynb` — the static end-to-end
  narrative; it only imports and calls package functions.

Dynamic control layer (second notebook):

- `src/rfq_edge/control_config.py`, `control_state.py` — episode and market
  configuration, frozen state/event/action dataclasses, economic mode labels.
- `src/rfq_edge/market_dynamics.py`, `execution_costs.py` — regime-switching
  event market with hidden client information, active execution costs, and
  inventory penalties.
- `src/rfq_edge/event_simulator.py` — forward simulator with a documented
  event ordering and common-random-number exogenous paths.
- `src/rfq_edge/control_models.py`, `oracle_control.py` — fitted control-market
  fill/selection models and the quadrature-based synthetic-truth oracle.
- `src/rfq_edge/bellman.py`, `dynamic_objective.py` — exact backward induction
  on (time, inventory, regime): a discrete Bellman approximation to a
  jump-HJB, with residual reporting and inventory shadow values.
- `src/rfq_edge/controllers.py`, `control_pipeline.py` — the five policies
  (plain, edge-consistent myopic, dynamic market maker, dynamic execution,
  oracle benchmark) and artifact assembly.
- `src/rfq_edge/control_evaluation.py`, `control_plots.py`,
  `control_reporting.py` — CRN policy evaluation with paired episode-block
  bootstrap, ablations, sensitivity scenarios, and all control figures.
- `notebooks/02_dynamic_rfq_market_making_and_execution.ipynb` — the dynamic
  market-making and position-execution narrative.

## Setup

```bash
pip install -e ".[demo,dev]"
```

## Run the tests

```bash
pytest -q
```

## Execute the notebooks

```bash
jupyter nbconvert --to notebook --execute \
  notebooks/01_edge_consistent_responder_demo.ipynb \
  --output /tmp/executed_rfq_demo.ipynb --ExecutePreprocessor.timeout=600

jupyter nbconvert --to notebook --execute \
  notebooks/02_dynamic_rfq_market_making_and_execution.ipynb \
  --output /tmp/executed_dynamic_rfq_demo.ipynb --ExecutePreprocessor.timeout=900
```

Everything is seeded; the notebooks reproduce exactly. Simulated clean value
and simulated control reward are diagnostic quantities on synthetic data, not
real trading PnL. The dynamic solver is a discrete Bellman approximation to a
jump-HJB, not an exact continuous-time PDE solution.
