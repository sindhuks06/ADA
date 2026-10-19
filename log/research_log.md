# Research Log — Context Reduction in Conditional Causal Bandits

## 17 October 2026 — Baseline Setup and First Experiment

**Objective:** Set up the existing conditional causal-bandit implementation and run the first experiment to understand how causal-graph structure can reduce the intervention search space.

### Work Completed

- Cloned the authors' official repository: [Minimal Search Space for Conditional Causal Bandits](https://github.com/francisco-simoes/minimal-set-conditional-intervention-bandits).
- Set up a Python 3.12 virtual environment and installed the required dependencies.
- Resolved Windows installation issues involving PyGraphviz, Graphviz and Microsoft C++ Build Tools.
- Installed Matplotlib and disabled external LaTeX rendering to allow the experiment's plots to be generated.
- Ran `experiments/mgiss_for_random_graphs.py` to evaluate the C4 algorithm on randomly generated directed acyclic graphs (DAGs).
- Generated histograms showing the fraction of the target's ancestors retained in the minimal search space (mGISS).

### Experiment Configuration

- **Number of graphs:** 1,000 per configuration.
- **Graph sizes:** 20, 100, 300 and 500 nodes.
- **Expected degrees:** 2, 5, 8 and 11.
- **Output:** A grid of 16 histograms showing the distribution of mGISS fractions.

### Results and Observations

The generated plots showed two general trends:

- For a fixed graph size, the average mGISS fraction increased as the expected degree increased.
- For a fixed expected degree, the average fraction generally decreased as the graph size increased.

For example, at expected degree 2, the observed average fraction decreased from 0.50 for 20-node graphs to 0.17 for 500-node graphs. At expected degree 11, it decreased from 0.94 to 0.76.

These observations suggest that the reduction in search-space size depends on graph structure and density. The results have not yet been systematically compared against the values reported in the original paper.

### Issues Resolved

- Used Python 3.12 because the original dependency versions were incompatible with Python 3.13.
- Configured Graphviz and the C++ build environment to install PyGraphviz.
- Disabled Matplotlib's external LaTeX dependency to render the plots successfully.
- Adjusted the figure layout to prevent overlapping titles and axis labels.

### Next Steps

1. Compare the experiment configuration and generated results with the original paper and repository outputs.
2. Run `experiments/mgiss_for_real_world_graphs.py` to evaluate the reduction on real-world graphs.
3. Record the baseline results before implementing or comparing context-selection methods such as mRMR and Model-X Knockoffs.

## 19 October 2026 — Reward-Preserving Context Reduction

**Objective:** Implement and evaluate a greedy context-reduction method that
removes unnecessary conditioning variables while keeping expected reward
within a configurable tolerance.

### Work Completed

- Implemented `reward_preserving_backward_elimination` with greedy
  single-variable removal, epsilon-based feasibility, deterministic
  tie-breaking, policy agreement, reward-gap, reduction-ratio, and runtime
  reporting.
- Added validation for zero-probability context groups and separate selection
  and evaluation feasibility metrics.
- Added tests covering irrelevant and policy-relevant context, epsilon
  tolerance, reward gaps, deterministic ties, interactions, separate
  evaluation tables, and invalid probability groups.
- Added a reproducible comparison experiment for the full-context reference,
  the existing prototype, and the new greedy method.
- Evaluated irrelevant, policy-relevant, interacting, and epsilon-tolerant
  synthetic cases using seed `20261018` and epsilon `0.1`.
- Saved results to `results/reward_preserving_reduction.csv` and documented
  the measured findings in `log/reward_preserving_reduction.md`.

### Results

- Irrelevant context was removed with no reward loss.
- The policy-relevant variable was retained by the new method; the existing
  prototype reduced reward from `0.8` to `0.5`.
- Interacting variables were retained.
- The epsilon case removed the variable with a measured reward gap of
  approximately `0.05`, within epsilon `0.1`.
- The complete test suite passed: `21 passed`.

No exhaustive-search comparison was included because no existing exhaustive
subset-search implementation was available.
