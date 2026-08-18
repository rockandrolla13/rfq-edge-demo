# RFQ Edge-Consistent Responder POC

## Implementation guide with factor attribution and order-flow imbalance

**Status:** Build specification  
**Purpose:** A reproducible proof of concept showing how an edge-consistent RFQ responder learns future value, fill probability, adverse selection, factor exposure, order-flow imbalance, and inventory-aware quote control.  
**Primary comparison:** Plain responder versus edge-consistent responder versus factor-and-flow-aware dynamic controller.

---

## 1. What this POC must prove

The POC answers one central question:

> Does an edge-consistent, factor-aware responder make better quote and inventory decisions than a plain responder when winning is informative, order flow is imbalanced, and fills change the factor risk of the book?

The POC must demonstrate the complete chain:

```text
Observable RFQ and market state
             |
             v
Predict unconditional future value V0
             |
             +--------------------+
             |                    |
             v                    v
Choose candidate quote q     Measure order-flow state
             |                    |
             v                    v
Predict fill probability p(q,X) and post-win value m(q,X)
             |
             v
Calculate conditional clean edge
             |
             v
Calculate factor exposure if filled
             |
             v
Add inventory continuation value
             |
             v
Quote, decline, warehouse, recycle, hedge, or execute
             |
             v
Attribute the result to quote edge, factors, selection, costs, and residual toxicity
```

The POC is successful only if every arrow above is implemented, visible in the notebooks, and tested.

---

## 2. What the POC is not

This is not initially:

- a production RFQ service;
- a live trading backtest;
- a causal estimate of real counterfactual RFQ outcomes;
- a high-dimensional nonlinear HJB;
- a reinforcement-learning policy;
- a complete client-service and relationship optimizer;
- a multi-venue smart-order router;
- evidence of real cash profitability.

The POC uses a known synthetic data-generating process so that estimated quantities can be compared with oracle truth. Its output is **simulated clean-value reward**, not live P&L.

---

## 3. Fixed conventions

These conventions must be used in every module and notebook.

### 3.1 Dealer-side sign

\[
\sigma_i=
\begin{cases}
+1, & \text{dealer buys from the client},\\
-1, & \text{dealer sells to the client}.
\end{cases}
\]

A filled RFQ of size \(n_i\) changes dealer inventory by:

\[
Q_{t+}=Q_t+\sigma_i n_i.
\]

### 3.2 Client-flow sign

Define:

\[
c_i=-\sigma_i.
\]

Therefore:

- \(c_i=+1\): client buys;
- \(c_i=-1\): client sells.

Use `client_sign` for order-flow imbalance and `side_sign` for dealer inventory and edge.

### 3.3 Price units

Use bond price points internally.

```text
0.01 price points = 1 cent per 100 notional
0.075 price points = 7.5 cents
```

Convert to cents only for tables and plots.

### 3.4 Core values

- \(M_i\): CP+ or public market mid at RFQ time.
- \(V_{0,i}\): unconditional pre-action future clean value.
- \(q_i\): actual dealer quote.
- \(W_i(q)\): win indicator under quote \(q\).
- \(Y_{H,i}\): independent future clean value at horizon \(H\).
- \(X_i\): point-in-time observable state.

### 3.5 Core edge equations

Apparent edge:

\[
h_i(q)=\sigma_i(V_{0,i}-q).
\]

Post-win clean value:

\[
m_i(q,X)=E[Y_{H,i}\mid W_i(q)=1,q,X_i].
\]

Adverse-selection adjustment:

\[
A_i(q,X)=\sigma_i(V_{0,i}-m_i(q,X)).
\]

Conditional clean edge:

\[
e_i(q,X)=\sigma_i(m_i(q,X)-q)=h_i(q)-A_i(q,X).
\]

The identity \(e=h-A\) must hold row by row within numerical tolerance.

---

## 4. Minimum POC scope

Use a synthetic market containing approximately:

| Item | POC size |
|---|---:|
| Bonds | 30-50 |
| Issuers | 8-12 |
| RFQs | 15,000-25,000 |
| Historical period | Approximately one year |
| Overall win rate | 20%-50% |
| Market regimes | 3 |
| Market factors | 4 |
| Detailed control bonds | 1-5 |

Suggested market factors:

1. Broad credit.
2. Rates/duration.
3. Sector.
4. Quality or curve.

Add an issuer factor and an idiosyncratic residual for every bond.

---

## 5. Repository structure

The notebooks are explanations and orchestrators. All implementation belongs in modules.

```text
rfq-edge-poc/
|-- pyproject.toml
|-- README.md
|-- src/rfq_edge/
|   |-- __init__.py
|   |-- config.py
|   |-- schema.py
|   |-- synthetic.py
|   |-- features.py
|   |-- order_flow.py
|   |-- factor_model.py
|   |-- factor_attribution.py
|   |-- value_model.py
|   |-- fill_model.py
|   |-- selection_model.py
|   |-- costs.py
|   |-- responders.py
|   |-- quote_grid.py
|   |-- static_objective.py
|   |-- control_state.py
|   |-- inventory_value.py
|   |-- bellman.py
|   |-- execution.py
|   |-- policy_evaluation.py
|   |-- plots.py
|   |-- notebook_api.py
|   `-- pipeline.py
|-- notebooks/
|   |-- 00_framework_map.ipynb
|   |-- 01_learning_and_static_responder.ipynb
|   |-- 02_dynamic_market_maker.ipynb
|   |-- 03_execution_controller.ipynb
|   |-- 04_factor_flow_and_macro.ipynb
|   `-- README.md
`-- tests/
    |-- test_synthetic.py
    |-- test_order_flow.py
    |-- test_factors.py
    |-- test_value_model.py
    |-- test_fill_model.py
    |-- test_selection_model.py
    |-- test_responders.py
    |-- test_attribution.py
    |-- test_bellman.py
    `-- test_pipeline.py
```

---

# Part I - Build the synthetic market

## Step 1. Define configuration and data contracts

### Purpose

Make units, signs, horizons, features, model settings, and simulation assumptions explicit.

### Implementation

Create frozen configuration dataclasses for:

- simulation;
- value model;
- fill model;
- selection model;
- factor model;
- order-flow windows;
- costs;
- static responder;
- dynamic controller.

Create schemas for:

- `RFQEvent`;
- `MarketState`;
- `PrivateValueState`;
- `OrderFlowState`;
- `FactorState`;
- `InventoryState`;
- `QuoteDecision`;
- `FillOutcome`;
- `CleanMarkout`.

### Minimum RFQ table

| Field | Meaning |
|---|---|
| `rfq_id` | Unique decision identifier |
| `timestamp` | Point-in-time RFQ timestamp |
| `bond_id`, `issuer_id` | Instrument hierarchy |
| `side`, `side_sign`, `client_sign` | Direction conventions |
| `size`, `cr01` | Notional and risk-unit conversion |
| `cp_plus` | Public market anchor |
| `internal_mid` | Raw private-value estimate |
| `quote` | Historical dealer action |
| `won` | Observed fill outcome |
| `y_h` | Independent clean future value |
| `market_width` | Quote normalization scale |
| `client_tier` | Pooled client context |
| `liquidity_score` | Point-in-time liquidity |
| `regime` | Observable market regime |
| `inventory` | Dealer inventory before quote |
| `target_inventory` | Desired inventory |
| factor columns | Bond loadings and factor state |
| flow columns | Lagged order-flow state |

### Output

- Validated configuration.
- Explicit schema checks.
- One source of truth for sign and unit conversions.

### Acceptance gate

- Invalid signs, duplicate RFQ identifiers, nonpositive size, and nonfinite prices fail loudly.
- No `latent_` field is admitted into a fitted-model feature matrix.

---

## Step 2. Simulate the bond universe and market factors

### Purpose

Create correlated bonds whose RFQ fills change economically meaningful factor risk.

### Implementation

Assign each bond:

- issuer;
- sector;
- rating bucket;
- maturity bucket;
- base price;
- CR01;
- liquidity level;
- activity weight;
- factor-loading vector \(\beta_i\).

Simulate factor changes:

\[
\Delta f_t\sim t_\nu(0,\Sigma_F(Z_t)),
\]

where \(Z_t\) is one of:

- `CALM_LIQUID`;
- `NORMAL`;
- `STRESSED_ILLIQUID`.

Generate bond price changes using:

\[
\Delta M_{i,t}
=
\beta_i^\top\Delta f_t
+\beta_i^{issuer}\Delta f_{issuer(i),t}
+\epsilon_{i,t}.
\]

Use heavy-tailed innovations rather than independent Gaussian prices.

### What should be visible

- Example CP+ price paths.
- Factor correlation heatmap.
- Bond-loading heatmap.
- Different volatility and liquidity under each regime.

### Acceptance gate

- Simulated covariance is positive semidefinite.
- Bonds from the same issuer are more correlated than unrelated bonds.
- Stress increases volatility, width, and active execution cost.

---

## Step 3. Simulate RFQ arrivals and historical quotes

### Purpose

Create sparse, uneven RFQ activity and a historically plausible quoting policy.

### Implementation

Use heavy-tailed bond activity weights so most bonds are sparse and a few are active.

Simulate RFQ arrival intensity:

\[
\lambda_{i,s,t}
=
\lambda(bond,side,client,liquidity,Z_t,flow_t).
\]

Generate historical normalized aggressiveness:

\[
z_i=\sigma_i\frac{q_i-M_i}{w_i},
\]

and reconstruct:

\[
q_i=M_i+\sigma_i z_iw_i.
\]

Larger \(z_i\) is more aggressive for both dealer buys and dealer sells.

Historical quotes may depend on observable state, internal alpha, inventory, and client tier. They must not depend directly on hidden client information.

### Output

- Chronologically ordered RFQ table.
- Sparse observations per bond.
- Historical quote-support ranges.

### Acceptance gate

- Median RFQs per bond is low relative to the full sample.
- Quote identity above holds numerically.
- Candidate quote support can be estimated by side and liquidity bucket.

---

## Step 4. Simulate future clean value and adverse selection

### Purpose

Ensure the synthetic market contains a genuine winner's curse.

### Implementation

Construct the observable conditional mean:

\[
\mu_i=E[Y_{H,i}-M_i\mid X_i].
\]

Generate:

\[
Y_{H,i}=M_i+\mu_i+\varepsilon^H_i.
\]

Create an unobserved client signal correlated with the future residual:

\[
U_i=\rho_U(Y_{H,i}-M_i)+\sqrt{1-\rho_U^2}\eta_i.
\]

Generate fill probability with:

\[
\operatorname{logit}p_i
=
\alpha
+\beta_z z_i
-\beta_U\sigma_iU_i
+g(X_i).
\]

The sign on hidden information is critical:

- dealer buys win more often when the client's future-value signal is poor;
- dealer sells win more often when the client's future-value signal is strong.

### Output

- `y_h` for every RFQ, including losses.
- `won` for every quote.
- oracle `p_true` and `m_true` available only when diagnostics are explicitly requested.

### Acceptance gate

- Fill probability rises with aggressiveness.
- Winning produces positive average adverse selection.
- Oracle fields never enter fitted features.

---

# Part II - Build order-flow imbalance

## Step 5. Construct point-in-time order-flow features

### Purpose

Represent client pressure without leaking the current or future RFQ outcome.

### Requested-flow imbalance

For group \(g\) and a lagged event window:

\[
OFI^{request}_{g,t}
=
\frac{\sum_{i<t,\,i\in g}w_i n_i c_i}
{\sum_{i<t,\,i\in g}w_i n_i}.
\]

Calculate at:

- market level;
- sector level;
- issuer level;
- bond level where supported;
- client-tier level.

Use at least two event clocks:

- fast: last 20 RFQs;
- slow: last 100 RFQs.

### Captured-flow imbalance

For policy evaluation:

\[
OFI^{captured}_{g,t}
=
\frac{\sum_{i<t}w_i n_i c_iW_i}
{\sum_{i<t}w_i n_iW_i}.
\]

Do not use contemporaneous `won` to predict the current RFQ.

### Flow surprise

\[
OFI^{surprise}_t
=OFI^{request}_t-E_{t-1}[OFI^{request}_t].
\]

### Three distinct uses

| Flow role | Where it belongs |
|---|---|
| Predicts future value | \(V_0\) model |
| Changes who accepts a quote | Fill and selection models |
| Determines factor inventory captured | Control transition |

Never add the same flow signal as an independent quote bonus after it has already been used in these components.

### What should be visible

- Fast and slow OFI through time.
- OFI by issuer and sector.
- OFI versus subsequent clean residual.
- Requested versus captured flow.

### Acceptance gate

- Every feature uses rows strictly earlier than the current RFQ.
- OFI is bounded and defined when the denominator is small.
- A fallback hierarchy is used: bond to issuer to sector to market.

---

# Part III - Learn the static responder

## Step 6. Fit unconditional future value \(V_0\)

### Purpose

Test whether private information and lagged flow improve future-value prediction beyond CP+.

### Target

\[
r_i=Y_{H,i}-M_i.
\]

### Models

1. CP+ baseline: \(\hat r_i=0\).
2. Raw internal mid: \(\hat r_i=internal\_mid_i-M_i\).
3. Regularized pooled residual model without OFI.
4. Regularized pooled residual model with OFI.

Use all eligible RFQ timestamps, not only fills.

Fit chronologically with a regularized pooled model. Do not fit one model per bond.

### Evaluation

- MAE and weighted MAE.
- Bias.
- RMSE.
- Directional accuracy for material moves.
- Calibration by prediction bucket.
- Stability by issuer, liquidity, regime, and bond-history bucket.
- Incremental improvement from OFI.

### Required output

Chronological out-of-fold predictions:

```text
rfq_id, timestamp, v0_oof, residual_prediction_oof, fold_id
```

### Acceptance gate

- `quote`, `won`, `y_h`, and `latent_*` are excluded from the feature matrix.
- OFI value is measured only from preceding RFQs.
- Comparison uses identical held-out rows.

---

## Step 7. Fit the fill model

### Purpose

Estimate:

\[
p(q,X,OFI)=P(W(q)=1\mid q,X,OFI).
\]

### Required quote feature

\[
z=\sigma\frac{q-M}{w}.
\]

Use all historical RFQs, including wins and losses.

Compare:

1. Aggressiveness-only logistic model.
2. Contextual regularized logistic model.
3. Contextual model with OFI.

### Evaluation

- Log loss.
- Brier score.
- Reliability plot.
- Fill rate by aggressiveness.
- Calibration by side, client tier, liquidity, regime, and OFI bucket.

### Acceptance gate

- Candidate quote changes the predicted probability.
- Larger aggressiveness generally raises predicted fill probability.
- The model is calibrated chronologically.

---

## Step 8. Fit conditional post-win value

### Purpose

Estimate what the bond is worth conditional on winning at the actual quote.

### Selection target

On filled RFQs:

\[
D_i=\sigma_i(V_{0,i}^{OOF}-Y_{H,i}).
\]

Fit:

\[
A(q,X,OFI)=E[D\mid W=1,q,X,OFI].
\]

Reconstruct:

\[
m(q,X,OFI)=V_0-\sigma A(q,X,OFI).
\]

### POC models

1. Constant selection haircut.
2. Regularized pooled selection model.
3. Regularized selection model with OFI.
4. Synthetic oracle benchmark.

The first POC may use the won-only model as an explicitly labelled baseline. It must state that production requires a selection-consistent joint model or control-function correction.

### Evaluation

- Selection MAE and bias.
- Selection calibration.
- Selection by quote aggressiveness.
- Selection by client tier and regime.
- Selection with and without OFI.
- Estimated versus oracle selection.

### Acceptance gate

- Only fills train the baseline selection model.
- The target uses out-of-fold \(V_0\).
- Candidate quote changes \(A\) and \(m\).
- \(e=h-A\) holds exactly.

---

# Part IV - Factor attribution

## Step 9. Compute factor exposure for every RFQ

### Purpose

Translate a bond fill into the risk that the dealer actually acquires.

For RFQ \(i\):

\[
\Delta F_i
=
\sigma_i n_i CR01_i\beta_i.
\]

Portfolio factor inventory is:

\[
F_t=\sum_j Q_{j,t}CR01_j\beta_j.
\]

Maintain:

- current exposure \(F_t\);
- desired factor target \(F_t^*\);
- exposure after a candidate fill;
- distance from target before and after.

### Expected captured factor flow

For one RFQ:

\[
E[\Delta F_i\mid q_i]=p_i(q_i,X_i,OFI_i)\Delta F_i.
\]

Across expected arrivals:

\[
\mu_F^\pi
=
\sum_i\lambda_i(X,Z)p_i(q_i,X,OFI)\Delta F_i.
\]

### What should be visible

- Factor exposure of a candidate fill.
- Current versus target exposure.
- Expected captured factor-flow vector.
- Difference between requested and captured factor flow.

### Acceptance gate

- Units reconcile from notional through CR01 to factor exposure.
- Buy and sell exposures are exact opposites for otherwise identical RFQs.

---

## Step 10. Build ex-post clean-markout attribution

### Purpose

Explain realized clean edge rather than treating every adverse markout as client toxicity.

Decompose:

\[
Y_{H,i}-M_i
=
\beta_i^\top\Delta f_H
+\beta_i^{issuer}\Delta f^{issuer}_H
+carry_i
+RV_i
+\epsilon_{H,i}.
\]

Then:

\[
\begin{aligned}
\sigma_i(Y_{H,i}-q_i)
={}&\sigma_i(M_i-q_i)\\
&+\sigma_i\beta_i^\top\Delta f_H\\
&+\sigma_i\beta_i^{issuer}\Delta f^{issuer}_H\\
&+\sigma_i carry_i\\
&+\sigma_i RV_i\\
&+\sigma_i\epsilon_{H,i}.
\end{aligned}
\]

Subtract transaction, hedge, liquidity, and funding costs separately.

### Required attribution table

| Component | Meaning |
|---|---|
| Quote-to-market | Spread captured relative to CP+ |
| Broad credit | Market credit factor contribution |
| Rates | Duration/rates contribution |
| Sector/quality | Other common-factor contribution |
| Issuer | Issuer-specific move |
| Carry/RV | Deliberately held carry or convergence |
| Residual | Unexplained favorable/adverse markout |
| Costs | RFQ, hedge, liquidity, funding |
| Net clean result | Reconciled total |

### Acceptance gate

- Components sum exactly to realized clean edge before costs.
- Net components sum exactly after costs.
- Residual toxicity is calculated only after factor and intended-alpha removal.

---

# Part V - Compare static responders

## Step 11. Implement four quote policies

All policies use the same RFQ state, quote grid, costs, support, and decline option.

### Policy A: Plain CP+ responder

\[
J_{CP+}(q)
=p(q,X)[\sigma(M-q)-K].
\]

### Policy B: Plain private-value responder

\[
J_{V_0}(q)
=p(q,X)[\sigma(V_0-q)-K].
\]

### Policy C: Edge-consistent responder

\[
J_{edge}(q)
=p(q,X,OFI)[\sigma(m(q,X,OFI)-q)-K].
\]

### Policy D: Factor-and-flow-aware responder

\[
J_{full}(q)
=p(q,X,OFI)
\left[n\,e(q,X,OFI)-K+\Delta\mathcal V^{factor}\right].
\]

Decline when the maximum objective is nonpositive.

### Required single-RFQ table

For every candidate quote, display:

- quote;
- normalized aggressiveness;
- fill probability;
- \(V_0\);
- adverse selection;
- post-win value;
- apparent edge;
- conditional clean edge;
- costs;
- factor exposure if filled;
- factor continuation value;
- expected objective.

### What should be visible

- Four objective curves on the same quote grid.
- The selected quote from each policy.
- A waterfall explaining why the full responder differs from the plain responder.

### Acceptance gate

- Plain and edge-consistent policies differ when selection is material.
- Factor-aware policy changes when inventory or factor targets change.
- Re-labelling \(V_0\) does not change the edge-consistent optimum when \(p\) and \(m\) are fixed.

---

# Part VI - Dynamic market making and execution

## Step 12. Demonstrate market making

### Purpose

Show how the same RFQ engine manages a near-zero inventory target through time.

Set:

\[
F^*=0.
\]

Use a running factor-risk penalty:

\[
\frac{\phi}{2}(F_t-F_t^*)^\top\Sigma_F(Z_t)(F_t-F_t^*).
\]

For pedagogical verification, solve a one-dimensional inventory problem by exact backward induction.

For scalable multi-factor control, use:

\[
\mathcal V(t,F,Z)
\approx
-\frac12(F-F^*)^\top A_Z(t)(F-F^*)-C_Z(t).
\]

Approximate candidate-fill continuation value by:

\[
\Delta\mathcal V_i
\approx
-\Delta F_i^\top A_Z(t)(F-F^*)
-\frac12\Delta F_i^\top A_Z(t)\Delta F_i.
\]

### Episode

Start with a flat book and simulate two-sided RFQs.

The controller should:

- quote for positive conditional edge near target;
- become defensive on trades that worsen accumulated factor exposure;
- become more aggressive on trades that reduce exposure;
- decline negative continuation-value RFQs;
- occasionally execute or hedge when risk becomes excessive.

### Required plots

- Inventory and factor exposure through time.
- Quote aggressiveness by inventory.
- Exact scalar value function.
- Inventory shadow value.
- Market-making action timeline.
- Reward attribution.

### Acceptance gate

- With zero risk penalty, the dynamic decision reduces to the myopic edge-consistent decision.
- Helpful RFQs become more aggressive; harmful RFQs become less aggressive.
- Bellman residual is below numerical tolerance in the scalar demonstration.

---

## Step 13. Demonstrate position execution

### Purpose

Show when the same controller stops behaving like a neutral market maker and uses RFQ flow to execute a position.

### Liquidation episode

Set:

```text
Initial inventory: long
Target inventory: zero
Deadline: finite
```

The controller can:

- wait;
- quote or decline an incoming RFQ;
- passively execute through helpful client flow;
- trade actively;
- hedge factor risk.

Use terminal penalty:

\[
\mathcal V(T,F)=-\frac{\eta}{2}(F_T-F_T^*)^\top\Sigma_F(F_T-F_T^*).
\]

### Expected behavior

- Client buys help a long dealer position and receive more aggressive offers.
- Client sells worsen a long position and receive defensive bids or declines.
- Early in the episode, the controller waits for economical RFQs.
- Near the deadline, active execution increases.
- In stress, reduced future RFQ arrivals reduce recycling value and can trigger hedging.

### Required plots

- Inventory versus target.
- Target shortfall through time.
- Passive versus active executed volume.
- RFQ event and action timeline.
- Completion-cost frontier.
- Mode timeline: market making, passive execution, active execution, wait, decline.

### Acceptance gate

- Urgency increases as the deadline approaches.
- Terminal shortfall falls when the terminal penalty rises.
- Inventory limits are never violated.
- Reward accounting does not count t+H alpha twice.

---

# Part VII - Evaluation

## Step 14. Run chronological and policy evaluations

### Predictive evaluation

Evaluate:

- \(V_0\) forecast quality;
- fill calibration;
- selection calibration;
- factor attribution accuracy;
- OFI incremental information.

### Policy evaluation

Run every policy on identical held-out states and common simulated random numbers.

Measure:

- response rate;
- fill rate;
- quote aggressiveness;
- conditional clean edge;
- realized simulated clean edge;
- adverse-selection loss;
- transaction costs;
- factor inventory volatility;
- distance from factor target;
- inventory-limit breaches;
- passive internalization;
- active execution cost;
- terminal shortfall;
- total simulated control reward.

Use paired block bootstrap by episode or simulated date.

### Required ablations

1. Remove OFI from \(V_0\).
2. Remove OFI from the fill model.
3. Remove OFI from the selection model.
4. Remove factor continuation value.
5. Remove expected recycling.
6. Replace estimated factors with oracle factors.
7. Replace estimated fill probability with oracle probability.
8. Replace estimated post-win value with oracle post-win value.

This identifies whether performance is constrained by value, fill, selection, factor risk, or flow modelling.

### Acceptance gate

- No superiority claim when paired confidence intervals include zero.
- OFI is retained only if it adds chronological out-of-sample information or improves control outcomes.
- Attribution reconciles exactly.
- All policy comparisons use identical costs and candidate quote support.

---

# Part VIII - Notebook specification

## Notebook 00 - Framework map

### Purpose

Provide an executive overview and direct the reader to the detailed notebooks.

### Sections

1. Economic question.
2. State, action, outcome, and clean markout.
3. Diagram of the complete workflow.
4. Plain versus edge-consistent responder.
5. Factor and flow extension.
6. Market making versus execution.
7. Notebook map and completion status.

### Output

A reader should understand the complete architecture without seeing implementation details.

---

## Notebook 01 - Learning and static responder

### Purpose

Show the simulation and learn the three central statistical quantities:

\[
V_0,\qquad p(q,X),\qquad m(q,X).
\]

### Sections

1. Generate the synthetic RFQs.
2. Show bond sparsity.
3. Show the hidden adverse-selection mechanism for simulation diagnostics.
4. Construct lagged OFI.
5. Compare CP+, raw internal mid, and fitted \(V_0\).
6. Fit and calibrate the fill model.
7. Fit and calibrate selection/post-win value.
8. Follow one buy and one sell RFQ across a quote grid.
9. Compare plain CP+, plain \(V_0\), and edge-consistent responders.
10. Run OFI ablations.

### Essential visualizations

- RFQs per bond.
- OFI through time.
- OFI versus future residual.
- Value-model comparison.
- Fill calibration.
- Selection calibration.
- Quote surface.
- Responder objective comparison.

---

## Notebook 02 - Dynamic market maker

### Purpose

Show how inventory continuation value changes otherwise identical RFQ decisions.

### Sections

1. Define inventory and factor target near zero.
2. Present the scalar Bellman recursion.
3. Validate the Bellman residual.
4. Present the multi-factor quadratic approximation.
5. Run one complete market-making episode.
6. Compare myopic and dynamic quotes.
7. Show defensive behavior near risk limits.
8. Evaluate across repeated episodes.

### Essential visualizations

- Value function.
- Shadow value.
- Quote policy heatmap.
- Factor inventory path.
- Action timeline.
- Reward waterfall.

---

## Notebook 03 - Execution controller

### Purpose

Show how nonzero targets and deadlines turn RFQ market making into passive and active execution.

### Sections

1. Define initial position, target, and deadline.
2. Run a long-position liquidation.
3. Identify helpful and harmful RFQs.
4. Show passive internalization.
5. Show active execution near deadline.
6. Compare wait, RFQ, active execution, and hedge.
7. Repeat for accumulation of a long position.
8. Compare policies over repeated episodes.

### Essential visualizations

- Inventory versus target.
- Passive versus active volume.
- Urgency surface.
- Completion-cost frontier.
- Mode timeline.

---

## Notebook 04 - Factor flow and macro state

### Purpose

Make factor attribution and order-flow skewing explicit.

### Sections

1. Factor model and loadings.
2. Requested order flow.
3. Captured order flow.
4. RFQ factor exposure.
5. Expected captured factor flow.
6. Current and target factor inventory.
7. Ex-ante factor shadow value.
8. Ex-post clean-markout attribution.
9. Risk-on versus risk-off primitives.
10. Optional exogenous macro retreat.

### Essential visualizations

- Loading heatmap.
- Factor covariance heatmap.
- Requested versus captured flow.
- Expected factor-flow vector.
- Factor inventory stacked area.
- Quote attribution waterfall.
- Realized markout decomposition.
- Risk-on versus risk-off quote changes by side.

---

# Part IX - Example end-to-end RFQ

Consider a dealer-buy RFQ:

```text
CP+                         100.00
Predicted V0                100.15
RFQ size                    1 inventory unit
Market width                0.10
Current credit exposure     above target
Fast client OFI             strongly client selling
Regime                      stressed
Cost                        0.075 price points
```

Candidate quotes might produce:

| Quantity | 99.94 | 99.98 | 100.02 |
|---|---:|---:|---:|
| Aggressiveness | -0.60 | -0.20 | 0.20 |
| Fill probability | 15% | 30% | 60% |
| Predicted post-win value | 100.08 | 100.05 | 100.01 |
| Conditional clean edge | 14c | 7c | -1c |
| RFQ cost | -7.5c | -7.5c | -7.5c |
| Factor continuation value | -3c | -3c | -3c |
| Net value if filled | 3.5c | -3.5c | -11.5c |
| Expected value | 0.5c | -1.1c | -6.9c |

The plain private-value responder sees apparent edge and may quote too aggressively. The edge-consistent responder recognizes that post-win value falls with aggressiveness. The factor-aware responder also recognizes that another dealer-buy fill worsens an already long credit-factor position.

This one example must be followed by aggregate policy evaluation; it is explanatory, not evidence by itself.

---

# Part X - POC acceptance criteria

## Statistical gates

- Chronological train/test separation.
- No target or future leakage.
- Calibrated fill probability.
- Calibrated conditional clean edge.
- OFI evaluated incrementally, not assumed useful.
- Sparse bonds use pooled models and fallbacks.

## Economic gates

- Candidate quote affects both \(p\) and \(m\).
- \(e=h-A\) holds exactly.
- Helpful inventory trades receive more aggressive quotes.
- Harmful trades receive more defensive quotes.
- Decline is possible.
- Market making is the near-zero-target case.
- Execution emerges from nonzero target and deadline.

## Factor and flow gates

- Every filled RFQ has a factor-risk vector.
- Requested and captured flow are distinguished.
- Factor attribution reconciles exactly.
- Residual toxicity excludes explained factor moves.
- Flow is not double-counted across value, selection, and control.

## Numerical gates

- All probabilities lie in \([0,1]\).
- No unsupported candidate quotes are optimized.
- Inventory limits are respected.
- Scalar Bellman residual is below tolerance.
- Reward components sum exactly to total reward.

## Demonstration gates

- Every notebook executes independently.
- Every plot has units and interpretation.
- Plain and edge-consistent decisions are compared on identical states.
- Synthetic oracle results are clearly separated from fitted-model results.
- Results are labelled simulated clean reward, not real P&L.

---

# Part XI - Recommended implementation order

Build in this order and stop at each gate:

1. Repository and notebook refactor.
2. Schemas, configuration, and sign/unit tests.
3. Synthetic factor market.
4. RFQ arrivals, quotes, fills, and independent markouts.
5. Point-in-time OFI.
6. \(V_0\) model and out-of-fold predictions.
7. Fill model.
8. Selection/post-win model.
9. Static responder comparison.
10. RFQ factor exposure and ex-post attribution.
11. Scalar Bellman market-maker demonstration.
12. Multi-factor quadratic inventory value.
13. Execution-controller episode.
14. Aggregate policy evaluation and ablations.
15. Notebook execution and hostile audit.

Do not build the next layer until the previous layer's tests and notebook cells pass.

---

# Part XII - Deferred production extensions

After the POC succeeds, extend in this order:

1. Selection-consistent joint fill/markout estimation.
2. Event/volume-clock clean marks with calendar caps.
3. Online fill correction and delayed markout queue.
4. Quality-adjusted hit ratio and client service shadow price.
5. Information-leakage penalty and alpha pass-through.
6. Background-flow recycling model.
7. Explicit hedge instruments and hedge-basis modelling.
8. Exogenous macro-retreat wiring and netting.
9. Real RFQ replay with historical-support restrictions.
10. Production monitoring, latency, audit, and governance.

---

## Definition of done

The POC is done when a reader can open the notebooks and observe, without reading package internals:

1. How the synthetic market creates adverse selection and order-flow imbalance.
2. Whether internal value and OFI predict the clean future mark.
3. How quote aggressiveness affects fill probability.
4. How quote aggressiveness affects post-win value.
5. Why a plain responder and edge-consistent responder choose different quotes.
6. What factor exposure each possible fill creates.
7. How current factor inventory changes the quote.
8. When the system is market making, passively executing, actively executing, or declining.
9. How realized clean results decompose into quote edge, factors, issuer, carry/RV, residual toxicity, and costs.
10. Which component - value, fill, selection, factor risk, or order flow - is responsible for any measured improvement.

