"""Data-generating processes used by the small synthetic context experiments."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def expected_rewards_from_function(
    contexts: Sequence[object],
    actions: Sequence[object],
    reward_function,
) -> np.ndarray:
    """Evaluate an explicit reward function into a context-by-action table."""
    if not contexts or not actions:
        raise ValueError("contexts and actions must be non-empty")
    return np.asarray(
        [[reward_function(context, action) for action in actions] for context in contexts],
        dtype=float,
    )


def compute_true_expected_rewards(
    contexts: Sequence[object],
    actions: Sequence[object],
    reward_function,
) -> np.ndarray:
    """Compute exact expected rewards from a supplied structural reward function."""
    return expected_rewards_from_function(contexts, actions, reward_function)


def compute_optimal_policy(expected_rewards: np.ndarray) -> np.ndarray:
    """Compute the exact full-context policy using lowest-index tie breaking."""
    from .evaluation import optimal_actions

    return optimal_actions(expected_rewards)


def generate_confounding_data(
    n_samples: int = 20_000,
    seed: int = 20261018,
) -> pd.DataFrame:
    """Generate binary observational data for ``Z -> X`` and ``Z -> Y``.

    DAG: ``Z -> X`` and ``Z -> Y``; ``X -> Y``.  Z is a confounder because it
    causes both treatment/action assignment X and outcome Y. The outcome model
    is E[Y | X, Z] = 0.20 + 0.10 Z + 0.40 X, so the true causal effect of X
    is 0.40 for either value of Z.
    """
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    rng = np.random.default_rng(seed)
    z = rng.binomial(1, 0.5, n_samples)
    treatment_probability = np.where(z == 1, 0.8, 0.2)
    x = rng.binomial(1, treatment_probability)
    outcome_probability = 0.20 + 0.10 * z + 0.40 * x
    y = rng.binomial(1, outcome_probability)
    return pd.DataFrame({"Z": z, "X": x, "Y": y})


def true_causal_effect() -> float:
    """Return the known interventional effect E[Y|do(X=1)] - E[Y|do(X=0)]."""
    return 0.40


def estimate_unadjusted_effect(data: pd.DataFrame) -> float:
    """Estimate the treatment effect as the unadjusted observational difference."""
    _validate_observational_data(data)
    return float(data.loc[data["X"] == 1, "Y"].mean() - data.loc[data["X"] == 0, "Y"].mean())


def estimate_effect_adjusted_for_z(data: pd.DataFrame) -> float:
    """Estimate the effect by standardizing within Z to its marginal distribution."""
    _validate_observational_data(data)
    marginal_z = data["Z"].value_counts(normalize=True).sort_index()
    conditional_means = data.groupby(["Z", "X"])["Y"].mean().unstack()
    if conditional_means.isnull().any().any():
        raise ValueError("both actions must be observed within each Z stratum")
    standardized = [
        float(sum(marginal_z[z] * conditional_means.loc[z, action] for z in marginal_z.index))
        for action in (0, 1)
    ]
    return standardized[1] - standardized[0]


def _validate_observational_data(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame) or not {"Z", "X", "Y"}.issubset(data.columns):
        raise ValueError("data must be a DataFrame with Z, X, and Y columns")
    if data.empty or not data[["Z", "X", "Y"]].isin([0, 1]).all().all():
        raise ValueError("Z, X, and Y must be non-empty binary columns")
