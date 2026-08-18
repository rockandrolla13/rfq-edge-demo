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
- `src/rfq_edge/plots.py` — all figures used by the notebook.
- `notebooks/01_edge_consistent_responder_demo.ipynb` — the end-to-end
  narrative; it only imports and calls package functions.

## Setup

```bash
pip install -e ".[demo,dev]"
```

## Run the tests

```bash
pytest -q
```

## Execute the notebook

```bash
jupyter nbconvert --to notebook --execute \
  notebooks/01_edge_consistent_responder_demo.ipynb \
  --output /tmp/executed_rfq_demo.ipynb --ExecutePreprocessor.timeout=600
```

Everything is seeded; the notebook reproduces exactly. Simulated clean value is
a diagnostic quantity on synthetic data, not real trading PnL.
