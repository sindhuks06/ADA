"""Small, explicitly non-causal context-selection prototype."""

from collections.abc import Mapping, Sequence
import math
import time

import numpy as np

from .evaluation import (
    expected_policy_reward,
    optimal_actions,
    policy_agreement,
)


def reduce_context(
    candidate_context: Sequence[str],
    protected_variables: Sequence[str],
    scores: Mapping[str, float],
    threshold: float = 0.5,
) -> list[str]:
    """Select variables by score while always retaining protected variables.

    This is a mechanics-only prototype. A predictive score is not evidence of
    causal irrelevance, and this function does not identify a valid adjustment
    set or establish policy preservation.
    """
    if isinstance(candidate_context, (str, bytes)):
        raise TypeError("candidate_context must be a sequence of variable names")
    if isinstance(protected_variables, (str, bytes)):
        raise TypeError("protected_variables must be a sequence of variable names")
    if not isinstance(scores, Mapping):
        raise TypeError("scores must be a mapping from variable names to scores")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
        raise ValueError("threshold must be a finite number")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    candidates = list(candidate_context)
    protected = list(protected_variables)
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate_context must not contain duplicate variables")
    if len(set(protected)) != len(protected):
        raise ValueError("protected_variables must not contain duplicate variables")
    if not set(protected).issubset(candidates):
        raise ValueError("protected_variables must be contained in candidate_context")
    for variable, score in scores.items():
        if variable not in candidates:
            raise ValueError(f"score provided for unknown variable: {variable!r}")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"score for {variable!r} must be a finite number")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score for {variable!r} must be between 0 and 1")

    return [
        variable
        for variable in candidates
        if variable in protected or scores.get(variable, 0.0) >= threshold
    ]


def _context_key(
    context: object, variables: Sequence[str], all_variables: Sequence[str]
) -> tuple[object, ...]:
    if isinstance(context, Mapping):
        try:
            return tuple(context[variable] for variable in variables)
        except KeyError as error:
            raise ValueError(f"context is missing variable {error.args[0]!r}") from error
    if isinstance(context, (str, bytes)):
        raise TypeError("contexts must be mappings or sequences of values")
    values = tuple(context)
    if len(values) != len(all_variables):
        raise ValueError("each context must contain one value per context variable")
    positions = {variable: index for index, variable in enumerate(all_variables)}
    return tuple(values[positions[variable]] for variable in variables)


def _best_policy_for_context(
    contexts: Sequence[object],
    variables: Sequence[str],
    all_variables: Sequence[str],
    expected_rewards: np.ndarray,
    context_distribution: Sequence[float],
) -> np.ndarray:
    """Choose one best action for each positive-probability context group.

    A zero-probability group is rejected because its policy action cannot be
    identified from the supplied distribution without inventing a reward.
    """
    rewards = np.asarray(expected_rewards, dtype=float)
    distribution = np.asarray(context_distribution, dtype=float)
    groups: dict[tuple[object, ...], list[int]] = {}
    for index, context in enumerate(contexts):
        groups.setdefault(_context_key(context, variables, all_variables), []).append(index)

    policy = np.empty(len(contexts), dtype=int)
    for indices in groups.values():
        weights = distribution[indices]
        total = weights.sum()
        if total <= 0:
            raise ValueError(
                "cannot evaluate a zero-probability candidate context group"
            )
        action = int(np.argmax(np.average(rewards[indices], axis=0, weights=weights)))
        policy[indices] = action
    return policy


def reward_preserving_backward_elimination(
    context_variables: Sequence[str],
    contexts: Sequence[object],
    expected_rewards: np.ndarray,
    context_distribution: Sequence[float],
    epsilon: float = 0.0,
    evaluation_expected_rewards: np.ndarray | None = None,
) -> dict[str, object]:
    """Greedily remove context variables while preserving expected reward.

    ``expected_rewards`` is used for selecting removals.  Supplying
    ``evaluation_expected_rewards`` evaluates the selected policy on a
    separate table, which is recommended when the reward table is estimated
    from data.  If it is omitted, the table is treated as an exact structural
    reward specification rather than an independently held-out estimate.

    The procedure is greedy: it can stop at a feasible subset even when a
    smaller feasible subset exists through a different sequence of removals.
    The returned ``selection_feasible`` reports whether the final selected
    policy meets the epsilon threshold on ``expected_rewards``. The legacy
    ``feasible`` key is exactly this same selection-table status. The
    ``evaluation_reward_gap`` and ``evaluation_within_epsilon`` fields report
    the selected policy's performance on the evaluation table and are
    intentionally independent of ``selection_feasible``. The returned
    ``policy_agreement`` is probability-weighted agreement with the
    full-context policy on the evaluation table.
    """
    if isinstance(context_variables, (str, bytes)):
        raise TypeError("context_variables must be a sequence of variable names")
    variables = list(context_variables)
    if not variables or len(set(variables)) != len(variables):
        raise ValueError("context_variables must contain unique variable names")
    if isinstance(contexts, (str, bytes)):
        raise TypeError("contexts must be a sequence of context assignments")
    context_list = list(contexts)
    if not context_list:
        raise ValueError("contexts must not be empty")
    if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be a finite non-negative number")

    rewards = np.asarray(expected_rewards, dtype=float)
    if rewards.ndim != 2 or rewards.shape[0] != len(context_list):
        raise ValueError("expected_rewards must have one row per context")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("expected_rewards must contain only finite values")
    distribution = np.asarray(context_distribution, dtype=float)
    if distribution.shape != (len(context_list),) or np.any(distribution < 0):
        raise ValueError("context_distribution must match contexts and be non-negative")
    if not math.isclose(float(distribution.sum()), 1.0):
        raise ValueError("context_distribution must sum to 1")
    evaluation_rewards = (
        rewards if evaluation_expected_rewards is None
        else np.asarray(evaluation_expected_rewards, dtype=float)
    )
    if evaluation_rewards.shape != rewards.shape or not np.all(np.isfinite(evaluation_rewards)):
        raise ValueError("evaluation_expected_rewards must match expected_rewards and be finite")

    started = time.perf_counter()
    reference_policy = optimal_actions(rewards)
    selected = list(variables)
    reference_reward = expected_policy_reward(
        evaluation_rewards, reference_policy, distribution
    )
    selection_reference_reward = expected_policy_reward(
        rewards, reference_policy, distribution
    )
    selected_policy = _best_policy_for_context(
        context_list, selected, variables, rewards, distribution
    )
    while selected:
        feasible: list[tuple[float, tuple[str, ...], np.ndarray]] = []
        for variable in selected:
            candidate = [item for item in selected if item != variable]
            candidate_policy = _best_policy_for_context(
                context_list, candidate, variables, rewards, distribution
            )
            candidate_reward = expected_policy_reward(
                rewards, candidate_policy, distribution
            )
            if candidate_reward >= selection_reference_reward - epsilon:
                feasible.append((candidate_reward, tuple(candidate), candidate_policy))
        if not feasible:
            break
        # Explicit lexicographic tie-breaking is independent of set iteration order.
        best_reward = max(item[0] for item in feasible)
        tied = [item for item in feasible if item[0] == best_reward]
        _, candidate_variables, selected_policy = min(tied, key=lambda item: item[1])
        selected = list(candidate_variables)

    selected_reward = expected_policy_reward(
        evaluation_rewards, selected_policy, distribution
    )
    selection_reward = expected_policy_reward(rewards, selected_policy, distribution)
    selection_gap = selection_reference_reward - selection_reward
    evaluation_gap = reference_reward - selected_reward
    selection_feasible = selection_gap <= epsilon
    return {
        "selected_context": selected,
        "reference_reward": reference_reward,
        "selected_reward": selected_reward,
        "reward_gap": evaluation_gap,
        "evaluation_reward_gap": evaluation_gap,
        "evaluation_within_epsilon": evaluation_gap <= epsilon,
        "policy_agreement": policy_agreement(
            reference_policy, selected_policy, distribution
        ),
        "retained_variables": len(selected),
        "reduction_ratio": (len(variables) - len(selected)) / len(variables),
        "runtime": time.perf_counter() - started,
        "selection_feasible": selection_feasible,
        "feasible": selection_feasible,
        "selection_reference_reward": selection_reference_reward,
        "selection_reward": selection_reward,
        "reference_policy": reference_policy,
        "selected_policy": selected_policy,
    }