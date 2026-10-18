# First Synthetic Context-Reduction Experiment

## Objective and research questions

This experiment asks whether removing conditioning-context variables can
preserve an optimal policy in simple cases, and whether a variable with modest
outcome predictiveness can still be needed for causal estimation. It is a
prototype, not a proposed validated causal-selection algorithm.

The published C4/mGISS baseline reduces candidate intervention nodes. These
experiments instead reduce variables conditioning the policy and keep that
distinction explicit.

## Experimental cases

### Test A: Irrelevant context

The context is `C in {0, 1}` and the actions are `0` and `1`. The exact
expected reward is:

* `E[Y | C, X=0] = 0.70`
* `E[Y | C, X=1] = 0.30`

The reward does not depend on `C`, so the full-context policy is `[0, 0]`.
The prototype removes `C`, and the same policy remains available.

### Test B: Policy-relevant context

The exact reward is `0.80` when the action equals the context and `0.20`
otherwise. Thus the full-context policy is `[0, 1]`. Under the explicitly
chosen distribution `P(C=0)=P(C=1)=0.5`, a context-free policy selects action
0 by the documented low-index tie break and receives an expected reward of
`0.50`, versus `0.80` with the full context.

The predictive-only prototype gives `C` a score of `0.9`, so it retains `C`
and computes policy `[0, 1]`, matching the full-context policy. This score is
intentionally not treated as causal evidence. A separate context-free
comparison actually removes `C`, computes policy `[0, 0]`, and measures the
cost of that removal.

### Test C: Weakly predictive but causally important variable

The DAG is:

```text
Z ───> X ───> Y
└───────────> Y
```

`Z` is a binary observed variable with probability `0.5`. Treatment/action
assignment is confounded:

* `P(X=1 | Z=0)=0.2`
* `P(X=1 | Z=1)=0.8`

The outcome is Bernoulli with
`P(Y=1 | X,Z)=0.20 + 0.10 Z + 0.40 X`. Therefore the true interventional
effect is `E[Y|do(X=1)] - E[Y|do(X=0)] = 0.40`. The direct effect of `Z` on
the outcome is deliberately smaller than the direct effect of `X`, but `Z`
also changes treatment assignment, making it a potential confounder.

The script generates 20,000 observations with NumPy seed `20261018`.
The unadjusted estimate is the difference in observed means between `X=1`
and `X=0`. The adjusted estimate standardizes the within-`Z` outcome means
to the observed marginal distribution of `Z`.

## Methods and metrics

Tests A and B calculate rewards exactly from explicit tables. They compare a
full-context policy and reduced policies under the same context distribution.
For Test B, the predictive-only selection retains `C`, while the separate
context-removed comparison removes `C`. Reported metrics are policy agreement,
expected reward, reward loss, original and retained context counts, and two
distinct fractions:

* `context fraction removed = (original count - retained count) / original count`
* `context fraction retained = retained count / original count`

Test C compares the known true causal effect with unadjusted and
`Z`-adjusted observational estimates and reports absolute estimation error.
Randomized interventions, observational estimation, and policy optimization
are different tasks: a randomized treatment can identify an effect without
adjustment, while this observational data requires adjustment for `Z` to
recover the effect under the stated model and positivity assumptions.
This does not imply that `Z` must be retained by every bandit policy; it is
required here for this observational adjustment strategy, whereas a policy
that directly observes the relevant reward structure may have different
requirements.

## Actual results

Running
`C:/Program Files/Python313/python.exe experiments\synthetic_context_tests.py`
produced the following actual output:

```text
Test A — irrelevant context
  prototype-selected context: []
  policy: [0, 0], agreement: 1.000
  expected reward: 0.700 -> 0.700
  context fraction removed: 1.000; retained: 0.000
Test B — policy-relevant context
predictive-only selected context: ['C']
  full policy: [0, 1], predictive-only policy: [0, 1]
  context-removed policy: [0, 0]
  predictive-only agreement: 1.000; reward: 0.800
  agreement: 0.500, reward loss: 0.300
  context fraction removed: 1.000; retained: 0.000
Test C — weakly predictive but confounder
  seed: 20261018, samples: 20000
  true effect: 0.400000
  unadjusted: 0.473412 (error 0.073412)
  adjusted for Z: 0.414545 (error 0.014545)
```

The same output is summarized below:

| Case | Full policy | Compared policy | Original contexts | Retained contexts | Agreement | Full reward | Compared reward | Fraction removed | Fraction retained |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | `[0, 0]` | `[0, 0]` | 1 | 0 | 1.000 | 0.700 | 0.700 | 1.000 | 0.000 |
| B, `C` removed | `[0, 1]` | `[0, 0]` | 1 | 0 | 0.500 | 0.800 | 0.500 | 1.000 | 0.000 |
| B, `C` retained | `[0, 1]` | `[0, 1]` | 1 | 1 | 1.000 | 0.800 | 0.800 | 0.000 | 1.000 |

For Test B the reward loss is `0.300`.

For Test C (`n=20,000`, seed `20261018`):

| Quantity | Value |
|---|---:|
| True causal effect | 0.400000 |
| Unadjusted estimate | 0.473412 |
| Unadjusted absolute error | 0.073412 |
| Adjusted estimate | 0.414545 |
| Adjusted absolute error | 0.014545 |

These are finite-sample simulation results, not guarantees.

## Limitations and assumptions

* The reduction rule uses an explicit score threshold and is not a causal
  selection or adjustment-set discovery method.
* Tests A and B have only two contexts and two actions and do not model
  learning, regret, or intervention uncertainty.
* Test C assumes the DAG is correctly specified, `Z` is observed, both
  actions occur in each `Z` stratum (positivity), and the outcome model is
  correctly represented by the data-generating process.
* The adjusted estimate is still subject to sampling error.
* No result here establishes novelty, general policy preservation, or causal
  validity outside the stated synthetic settings.

## Validation and next research steps

The complete available pytest suite passed: **8 passed**. The next task is to extend the
experiments to multiple DAGs and context distributions, define a justified
causal selection criterion, and evaluate learned bandit policies and regret
without conflating intervention-node reduction with context reduction.
