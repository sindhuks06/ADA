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

Tests A and B calculate rewards exactly from explicit tables and compare three
methods using the same reward table, candidate actions, context values, and
context distribution within each test:

1. `full_context_reference`: retains `C` and uses the exact full-context
   optimal policy.
2. `predictive_only`: applies a predictive score threshold. It scores `C` at
   `0.0` in Test A and `0.9` in Test B.
3. `prototype_context_reduction`: applies the prototype threshold interface to
   a separate illustrative score. It scores `C` at `0.0` in Test A and `0.4`
   in Test B, so it removes `C` in Test B.

The Test B prototype score is an intentionally explicit prototype input, not
an estimated or causally justified score. Reported metrics are policy
agreement, expected reward, original and retained context counts, and two
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
Comparison CSV: C:\Users\Nagaraja\Desktop\context-reduction-causal-bandits\results\synthetic_context_comparison.csv
```

The six comparison rows were written to
[results/synthetic_context_comparison.csv](../results/synthetic_context_comparison.csv).
The file contains the method name, retained variables, both context
fractions, policy agreement, expected reward, and full-context reward.

The exact CSV-backed policy results are:

| Case | Method | Retained variables | Original contexts | Retained contexts | Agreement | Expected reward | Fraction removed | Fraction retained |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | full_context_reference | `C` | 1 | 1 | 1.000 | 0.700 | 0.000 | 1.000 |
| A | predictive_only | none | 1 | 0 | 1.000 | 0.700 | 1.000 | 0.000 |
| A | prototype_context_reduction | none | 1 | 0 | 1.000 | 0.700 | 1.000 | 0.000 |
| B | full_context_reference | `C` | 1 | 1 | 1.000 | 0.800 | 0.000 | 1.000 |
| B | predictive_only | `C` | 1 | 1 | 1.000 | 0.800 | 0.000 | 1.000 |
| B | prototype_context_reduction | none | 1 | 0 | 0.500 | 0.500 | 1.000 | 0.000 |

For Test B, the prototype's context removal causes an expected-reward loss of
`0.300` relative to the full-context reference. The predictive-only method
retains `C` and matches the reference in this controlled example.

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

### Review limitations

The Test A predictive-only result is produced by the defined threshold rule:
the experiment passes `C` a hand-specified score of `0.0`, and the default
threshold is `0.5`, so `reduce_context` selects no variables. It is therefore
not hard-coded as an output, but the score itself is not learned from data.
Likewise, Test B uses illustrative scores `0.9` and `0.4` to make the
predictive and prototype comparisons differ. These scores demonstrate the
mechanics of selection; they do not validate predictive feature selection or
the prototype as a causal method.

The comparison CSV is generated by `run_policy_comparisons()` and
`write_comparison_csv()`. Each row uses `evaluate_policy_method()`, which calls
the shared `compare_policies()` function for policy agreement and exact
expected reward. The automated comparison tests consume the same
`run_policy_comparisons()` rows, while the CSV is a persisted copy of those
rows. The fraction checks are mathematically consistent: for every row,
`context fraction removed + context fraction retained = 1.0`, and the
reported expected-reward difference from the full reference is the policy
reward loss.

* The prototype uses an explicit, hand-specified score threshold and is not a
  causal selection or adjustment-set discovery method.
* The predictive-only scores and prototype scores are illustrative inputs;
  they are not learned estimates, and predictive scores are not proof of
  causal importance or irrelevance.
* Tests A and B have only two contexts and two actions and do not model
  learning, regret, or intervention uncertainty.
* The full-context, predictive-only, and prototype comparisons are exact
  policy evaluations on shared synthetic reward tables; they do not constitute
  independent empirical datasets or a bandit-learning evaluation.
* Test C assumes the DAG is correctly specified, `Z` is observed, both
  actions occur in each `Z` stratum (positivity), and the outcome model is
  correctly represented by the data-generating process.
* The adjusted estimate is still subject to sampling error.
* No result here establishes novelty, general policy preservation, or causal
  validity outside the stated synthetic settings.
* Test C's adjustment result depends on the stated DAG, consistency,
  positivity, observed confounding, and correct outcome-model assumptions.
  It shows finite-sample behavior for this data-generating process, not a
  guarantee that `Z` must be retained by every policy or estimator.

## Validation and next research steps

The complete available pytest suite passed: **10 passed**. The next task is to extend the
experiments to multiple DAGs and context distributions, define a justified
causal selection criterion, and evaluate learned bandit policies and regret
without conflating intervention-node reduction with context reduction.
