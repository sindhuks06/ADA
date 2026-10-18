# Context Reduction in Conditional Causal Bandits

## 1. Overview

This project investigates **context reduction in conditional causal bandits**, with the goal of reducing the number of context variables required for decision-making while preserving the optimal intervention policy.

In conditional causal bandits, an agent selects interventions based on observed context variables and learns which interventions produce the best outcomes. Using unnecessary context variables can increase complexity, while removing important variables may lead to suboptimal decisions or invalid causal conclusions.

The project aims to investigate context-reduction methods that balance efficiency, policy preservation, and causal validity.

## 2. Research Objectives

- Investigate methods for reducing conditioning-context variables in conditional causal bandits.
- Study existing causal-bandit methods and establish reproducible baselines.
- Develop a context-reduction approach that preserves optimal policy decisions under clearly stated assumptions.
- Evaluate the trade-off between context reduction, decision quality, and computational cost.
- Examine the risk of removing variables that are important for causal reasoning even when they have weak predictive power.

## 3. Research Question

**Can the conditioning context in a conditional causal bandit be reduced while preserving the optimal intervention policy and maintaining valid causal reasoning under appropriate assumptions?**

The project will investigate the assumptions and conditions under which such a reduction is possible.

## 4. Baseline Research

The initial baseline is based on the following paper:

**Simoes et al. — The Minimal Search Space for Conditional Causal Bandits (UAI 2026)**

- Paper: https://proceedings.mlr.press/v337/simoes26a.html
- Official implementation: https://github.com/francisco-simoes/minimal-set-conditional-intervention-bandits

The paper introduces C4, an algorithm for identifying a minimal globally interventionally superior set (mGISS) to reduce the candidate intervention search space.

This baseline provides a reference for studying search-space reduction and its effect on bandit performance.

**Important distinction:** Reducing candidate intervention nodes is not necessarily the same as reducing the conditioning variables used by an intervention policy. This project focuses on the latter as its intended research direction.

## 5. Current Progress

### Completed

- Set up the research repository and initial project structure.
- Set up the published baseline implementation locally.
- Executed the baseline random-graph experiment.
- Executed the baseline real-world Bayesian-network experiment.
- Saved baseline visualizations for documenting the observed search-space reductions.
- Started testing the conditional-intervention UCB experiment.
- Added small, reproducible synthetic experiments for conditioning-context reduction.
- Added exact policy metrics and a seeded observational confounding experiment.
- Added automated tests covering the synthetic experiment utilities.

### In Progress

- Completing and validating the baseline bandit experiment.
- Reviewing related work on context selection, causal inference, and conditional causal bandits.
- Defining and validating a context-reduction method and its assumptions.

### Planned

- Evaluate the synthetic prototype under more causal graph structures.
- Implement and justify a causal context-selection method rather than relying on predictive scores.
- Compare a validated method against full-context and relevant existing baselines.
- Evaluate policy agreement, cumulative regret, context size, runtime, and causal validity where measurable.
- Document experimental results and limitations.

## 6. Proposed Evaluation

The experiments will investigate the following metrics:

| Metric | Purpose |
|---|---|
| Context size | Measures the number of context variables retained. |
| Reduction ratio | Measures how much the context is reduced. |
| Policy agreement | Measures whether the reduced context selects the same optimal policy as the full context. |
| Cumulative regret | Measures decision-making performance over time. |
| Runtime | Measures computational cost. |
| Causal validity | Examines whether the reduction preserves the causal assumptions required for valid conclusions. |

The evaluation will distinguish empirical observations from theoretical guarantees. Any policy-preservation or causal-validity claim will be stated only under assumptions supported by the method and experiments.

## 7. Repository Structure

```text
ADA/
├── README.md
├── requirements.txt
├── src/
│   ├── context_reduction.py
│   ├── evaluation.py
│   └── synthetic_data.py
├── experiments/
│   └── synthetic_context_tests.py
├── tests/
│   └── test_synthetic_context_tests.py
├── results/
│   ├── random_graphs_baseline.png
│   └── real_world_graphs_baseline.png
└── log/
    └── research_log.md
```

The structure represents the intended organization of the project. Some files and experiments may still be under development.

- `src/`: Implementation of the proposed method and evaluation utilities.
- `experiments/`: Reproducible experiment scripts.
- `results/`: Plots, metrics, and experiment outputs.
- `log/research_log.md`: Chronological record of research progress, experiment settings, observations, and issues.

## 8. Research Principles

- Preserve a clear distinction between the published baseline and the proposed method.
- Record experiment configurations and limitations.
- Avoid equating predictive feature importance with causal importance.
- State assumptions explicitly, especially when relying on a known causal graph.
- Compare methods using consistent evaluation settings wherever possible.
- Do not claim novelty, policy preservation, or causal guarantees without appropriate evidence.

## 9. Current Status

**Stage:** Baseline reproduction plus initial synthetic prototype experiments.

The graph-based baseline experiments have been executed, while the conditional-intervention UCB evaluation is still being validated. The proposed context-reduction algorithm and comparative evaluation remain future work.

The synthetic code is deliberately a controlled prototype: Test A and Test B
illustrate policy behavior, while Test C illustrates confounding and adjustment.
These experiments do not establish a causal-selection algorithm, novelty, or a
general policy-preservation guarantee.

## 10. Acknowledgements

The initial baseline implementation is based on the work of Simoes et al. Please refer to the original paper and repository for the authors' implementation, methodology, and licensing terms.
