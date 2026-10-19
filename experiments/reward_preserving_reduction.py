"""Reproducible comparisons for reward-preserving context reduction."""

import argparse
import csv
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.context_reduction import (  # noqa: E402
    _best_policy_for_context,
    reduce_context,
    reward_preserving_backward_elimination,
)
from src.evaluation import (  # noqa: E402
    context_reduction_fraction_removed,
    expected_policy_reward,
    optimal_actions,
    policy_agreement,
)
from src.synthetic_data import compute_true_expected_rewards  # noqa: E402


RESULTS_PATH = ROOT / "results" / "reward_preserving_reduction.csv"
DEFAULT_SEED = 20261018
DEFAULT_EPSILON = 0.1


def _case(
    name: str,
    contexts: list[tuple[int, ...]],
    variables: list[str],
    reward_function,
) -> dict[str, object]:
    rewards = compute_true_expected_rewards(
        contexts, [0, 1], reward_function
    )
    return {
        "name": name,
        "contexts": contexts,
        "variables": variables,
        "rewards": rewards,
        "distribution": [1.0 / len(contexts)] * len(contexts),
    }


def build_cases(seed: int) -> list[dict[str, object]]:
    """Build fixed synthetic cases, shuffling rows reproducibly by ``seed``."""
    rng = np.random.default_rng(seed)
    base_cases = [
        _case(
            "irrelevant_context",
            [(0,), (1,)],
            ["C"],
            lambda _context, action: 0.7 if action == 0 else 0.3,
        ),
        _case(
            "policy_relevant_context",
            [(0,), (1,)],
            ["C"],
            lambda context, action: 0.8 if context[0] == action else 0.2,
        ),
        _case(
            "interacting_context",
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            ["A", "B"],
            lambda context, action: (
                0.8 if action == (context[0] ^ context[1]) else 0.2
            ),
        ),
        _case(
            "epsilon_tolerance",
            [(0,), (1,)],
            ["C"],
            lambda context, action: (
                1.0 if context[0] == 0 and action == 0
                else 0.9 if context[0] == 1 and action == 0
                else 1.0 if context[0] == 1 and action == 1
                else 0.0
            ),
        ),
    ]
    for case in base_cases:
        order = rng.permutation(len(case["contexts"]))
        case["contexts"] = [case["contexts"][index] for index in order]
        case["rewards"] = case["rewards"][order]
    return base_cases


def _prototype_context(case: dict[str, object]) -> list[str]:
    variables = case["variables"]
    if case["name"] == "irrelevant_context":
        scores = {variable: 0.0 for variable in variables}
    elif case["name"] == "policy_relevant_context":
        scores = {variable: 0.4 for variable in variables}
    else:
        scores = {variable: 0.9 for variable in variables}
    return reduce_context(variables, [], scores)


def _record(
    case: dict[str, object],
    method: str,
    selected: list[str],
    policy: np.ndarray,
    reference_policy: np.ndarray,
    rewards: np.ndarray,
    distribution: list[float],
    runtime: float,
    epsilon: float,
    selection_feasible: bool | None = None,
) -> dict[str, object]:
    reference_reward = expected_policy_reward(rewards, reference_policy, distribution)
    achieved_reward = expected_policy_reward(rewards, policy, distribution)
    return {
        "case": case["name"],
        "method": method,
        "selected_context": ",".join(selected),
        "context_size": len(selected),
        "reduction_ratio": context_reduction_fraction_removed(
            len(case["variables"]), len(selected)
        ),
        "reference_reward": reference_reward,
        "achieved_reward": achieved_reward,
        "reward_gap": reference_reward - achieved_reward,
        "policy_agreement": policy_agreement(
            reference_policy, policy, distribution
        ),
        "selection_feasible": (
            achieved_reward >= reference_reward - epsilon
            if selection_feasible is None
            else selection_feasible
        ),
        "evaluation_within_epsilon": achieved_reward >= reference_reward - epsilon,
        "runtime": runtime,
    }


def run_experiment(
    seed: int = DEFAULT_SEED, epsilon: float = DEFAULT_EPSILON
) -> list[dict[str, object]]:
    """Run all methods under identical contexts, rewards, and distributions."""
    rows: list[dict[str, object]] = []
    for case in build_cases(seed):
        contexts = case["contexts"]
        variables = case["variables"]
        rewards = case["rewards"]
        distribution = case["distribution"]
        reference_policy = optimal_actions(rewards)

        started = time.perf_counter()
        rows.append(
            _record(
                case,
                "full_context_reference",
                list(variables),
                reference_policy,
                reference_policy,
                rewards,
                distribution,
                time.perf_counter() - started,
                epsilon,
            )
        )

        started = time.perf_counter()
        prototype_selected = _prototype_context(case)
        prototype_policy = _best_policy_for_context(
            contexts, prototype_selected, variables, rewards, distribution
        )
        rows.append(
            _record(
                case,
                "prototype_context_reduction",
                prototype_selected,
                prototype_policy,
                reference_policy,
                rewards,
                distribution,
                time.perf_counter() - started,
                epsilon,
            )
        )

        started = time.perf_counter()
        reduced = reward_preserving_backward_elimination(
            variables,
            contexts,
            rewards,
            distribution,
            epsilon=epsilon,
        )
        rows.append(
            _record(
                case,
                "reward_preserving_backward_elimination",
                reduced["selected_context"],
                reduced["selected_policy"],
                reference_policy,
                rewards,
                distribution,
                time.perf_counter() - started,
                epsilon,
                reduced["selection_feasible"],
            )
        )
    return rows


def write_results(rows: list[dict[str, object]], path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "method",
        "selected_context",
        "context_size",
        "reduction_ratio",
        "reference_reward",
        "achieved_reward",
        "reward_gap",
        "policy_agreement",
        "selection_feasible",
        "evaluation_within_epsilon",
        "runtime",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    args = parser.parse_args()
    rows = run_experiment(seed=args.seed, epsilon=args.epsilon)
    write_results(rows)
    print(f"Wrote {len(rows)} rows to {RESULTS_PATH}")
    for row in rows:
        print(
            f"{row['case']} / {row['method']}: "
            f"context={row['selected_context'] or 'none'}, "
            f"reward={row['achieved_reward']:.3f}, "
            f"gap={row['reward_gap']:.3f}"
        )


if __name__ == "__main__":
    main()
