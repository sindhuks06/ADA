"""Reusable exact-policy and context-reduction evaluation utilities."""

from collections.abc import Sequence
import math

import numpy as np


def _validate_rewards(expected_rewards: np.ndarray) -> np.ndarray:
    rewards = np.asarray(expected_rewards, dtype=float)
    if rewards.ndim != 2 or rewards.shape[0] == 0 or rewards.shape[1] == 0:
        raise ValueError("expected_rewards must be a non-empty 2D array")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("expected_rewards must contain only finite values")
    return rewards


def optimal_actions(expected_rewards: np.ndarray) -> np.ndarray:
    """Return the best action in each context, breaking ties by low index."""
    return np.argmax(_validate_rewards(expected_rewards), axis=1)


def _validate_distribution(context_distribution: Sequence[float], n_contexts: int) -> np.ndarray:
    distribution = np.asarray(context_distribution, dtype=float)
    if distribution.shape != (n_contexts,) or np.any(distribution < 0):
        raise ValueError("context_distribution must be non-negative and match contexts")
    total = distribution.sum()
    if not math.isclose(float(total), 1.0):
        raise ValueError("context_distribution must sum to 1")
    return distribution


def expected_policy_reward(
    expected_rewards: np.ndarray,
    policy: Sequence[int],
    context_distribution: Sequence[float],
) -> float:
    """Compute the exact expected reward of a policy over contexts."""
    rewards = _validate_rewards(expected_rewards)
    if len(policy) != rewards.shape[0]:
        raise ValueError("policy must contain one action per context")
    actions = np.asarray(policy, dtype=int)
    if np.any(actions < 0) or np.any(actions >= rewards.shape[1]):
        raise ValueError("policy contains an action outside the reward table")
    distribution = _validate_distribution(context_distribution, rewards.shape[0])
    return float(np.dot(distribution, rewards[np.arange(rewards.shape[0]), actions]))


def compare_policies(
    full_policy: Sequence[int],
    reduced_policy: Sequence[int],
    expected_rewards: np.ndarray,
    context_distribution: Sequence[float],
) -> dict[str, float]:
    """Compare agreement and exact expected rewards for two policies."""
    if len(full_policy) != len(reduced_policy):
        raise ValueError("policies must have equal length")
    distribution = _validate_distribution(context_distribution, len(full_policy))
    full_reward = expected_policy_reward(expected_rewards, full_policy, distribution)
    reduced_reward = expected_policy_reward(expected_rewards, reduced_policy, distribution)
    agreement = float(
        np.dot(distribution, np.asarray(full_policy) == np.asarray(reduced_policy))
    )
    return {
        "policy_agreement": agreement,
        "full_context_reward": full_reward,
        "reduced_context_reward": reduced_reward,
        "reward_loss": full_reward - reduced_reward,
    }


def evaluate_policy_method(
    method_name: str,
    retained_variables: Sequence[str],
    full_policy: Sequence[int],
    expected_rewards: np.ndarray,
    context_distribution: Sequence[float],
    original_variables: Sequence[str],
) -> dict[str, object]:
    """Return consistently named policy and context metrics for one method."""
    original = list(original_variables)
    retained = list(retained_variables)
    if len(set(original)) != len(original) or len(set(retained)) != len(retained):
        raise ValueError("context variable lists must not contain duplicates")
    if not set(retained).issubset(original):
        raise ValueError("retained_variables must be contained in original_variables")
    if not method_name:
        raise ValueError("method_name must not be empty")

    metrics = compare_policies(
        full_policy,
        full_policy if set(retained) == set(original) else _context_free_policy(
            expected_rewards, context_distribution
        ),
        expected_rewards,
        context_distribution,
    )
    return {
        "method": method_name,
        "retained_variables": ",".join(retained),
        "original_context_count": len(original),
        "retained_context_count": len(retained),
        "context_fraction_removed": context_reduction_fraction_removed(
            len(original), len(retained)
        ),
        "context_fraction_retained": context_retention_fraction(
            len(original), len(retained)
        ),
        "policy_agreement": metrics["policy_agreement"],
        "expected_reward": metrics["reduced_context_reward"],
        "full_context_reward": metrics["full_context_reward"],
    }


def _context_free_policy(
    expected_rewards: np.ndarray, context_distribution: Sequence[float]
) -> np.ndarray:
    """Return the best single action repeated for every context."""
    rewards = _validate_rewards(expected_rewards)
    distribution = _validate_distribution(context_distribution, rewards.shape[0])
    return np.repeat(np.argmax(np.dot(distribution, rewards)), rewards.shape[0])


def policy_agreement(
    full_policy: Sequence[int],
    reduced_policy: Sequence[int],
    context_distribution: Sequence[float] | None = None,
) -> float:
    """Return weighted (or uniform) agreement between two policies."""
    if len(full_policy) != len(reduced_policy) or len(full_policy) == 0:
        raise ValueError("policies must be non-empty and have equal length")
    if context_distribution is None:
        distribution = np.full(len(full_policy), 1.0 / len(full_policy))
    else:
        distribution = _validate_distribution(context_distribution, len(full_policy))
    return float(np.dot(distribution, np.asarray(full_policy) == np.asarray(reduced_policy)))


def context_reduction_fraction_removed(original_count: int, retained_count: int) -> float:
    """Return the fraction of the original context variables removed."""
    if not isinstance(original_count, int) or not isinstance(retained_count, int):
        raise TypeError("context counts must be integers")
    if original_count <= 0 or not 0 <= retained_count <= original_count:
        raise ValueError("counts must satisfy 0 <= retained <= original and original > 0")
    return (original_count - retained_count) / original_count


def context_retention_fraction(original_count: int, retained_count: int) -> float:
    """Return the fraction of the original context variables retained."""
    if not isinstance(original_count, int) or not isinstance(retained_count, int):
        raise TypeError("context counts must be integers")
    if original_count <= 0 or not 0 <= retained_count <= original_count:
        raise ValueError("counts must satisfy 0 <= retained <= original and original > 0")
    return retained_count / original_count


def reporting_estimation_error(estimate: float, true_effect: float) -> float:
    """Return absolute error relative to a known causal effect."""
    if not np.isfinite(estimate) or not np.isfinite(true_effect):
        raise ValueError("estimate and true_effect must be finite")
    return float(abs(estimate - true_effect))
