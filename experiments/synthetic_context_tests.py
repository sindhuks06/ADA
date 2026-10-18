"""Run controlled synthetic experiments for conditioning-context reduction."""

import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.context_reduction import reduce_context
from src.evaluation import (
    compare_policies,
    context_reduction_fraction_removed,
    context_retention_fraction,
    evaluate_policy_method,
    reporting_estimation_error,
)
from src.synthetic_data import (
    compute_optimal_policy,
    compute_true_expected_rewards,
    estimate_effect_adjusted_for_z,
    estimate_unadjusted_effect,
    generate_confounding_data,
    true_causal_effect,
)

SEED = 20261018
CONTEXT_DISTRIBUTION = [0.5, 0.5]
ORIGINAL_CONTEXT = ["C"]
RESULTS_PATH = ROOT / "results" / "synthetic_context_comparison.csv"


def run_test_a() -> dict[str, object]:
    contexts, actions = [0, 1], [0, 1]
    predictive_selected_context = reduce_context(ORIGINAL_CONTEXT, [], {"C": 0.0})
    prototype_selected_context = reduce_context(ORIGINAL_CONTEXT, [], {"C": 0.0})
    rewards = compute_true_expected_rewards(
        contexts, actions, lambda _context, action: 0.7 if action == 0 else 0.3
    )
    full_policy = compute_optimal_policy(rewards)
    reduced_policy = np.repeat(full_policy[0], len(contexts))
    return {
        "expected_rewards": rewards,
        "full_policy": full_policy,
        "reduced_policy": reduced_policy,
        "selected_context": prototype_selected_context,
        "predictive_selected_context": predictive_selected_context,
        "prototype_selected_context": prototype_selected_context,
        "metrics": compare_policies(
            full_policy, reduced_policy, rewards, CONTEXT_DISTRIBUTION
        ),
        "comparison_rows": _comparison_rows(
            full_policy,
            rewards,
            CONTEXT_DISTRIBUTION,
            predictive_selected_context,
            prototype_selected_context,
        ),
        "context_fraction_removed": context_reduction_fraction_removed(
            1, len(prototype_selected_context)
        ),
        "context_fraction_retained": context_retention_fraction(
            1, len(prototype_selected_context)
        ),
    }


def run_test_b() -> dict[str, object]:
    contexts, actions = [0, 1], [0, 1]
    predictive_selected_context = reduce_context(ORIGINAL_CONTEXT, [], {"C": 0.9})
    # This separate score is intentionally below threshold to expose the
    # prototype's context-removal behavior; it is not causal evidence.
    prototype_selected_context = reduce_context(ORIGINAL_CONTEXT, [], {"C": 0.4})
    rewards = compute_true_expected_rewards(
        contexts, actions, lambda context, action: 0.8 if context == action else 0.2
    )
    full_policy = compute_optimal_policy(rewards)
    # The best context-free action under this distribution is action 0;
    # ties are resolved by the documented low-index rule.
    reduced_action = int(np.argmax(np.dot(CONTEXT_DISTRIBUTION, rewards)))
    context_removed_policy = np.repeat(reduced_action, len(contexts))
    predictive_policy = compute_optimal_policy(rewards)
    return {
        "expected_rewards": rewards,
        "full_policy": full_policy,
        "reduced_policy": context_removed_policy,
        "context_removed_policy": context_removed_policy,
        "predictive_selected_context": predictive_selected_context,
        "prototype_selected_context": prototype_selected_context,
        "predictive_policy": predictive_policy,
        "predictive_metrics": compare_policies(
            full_policy, predictive_policy, rewards, CONTEXT_DISTRIBUTION
        ),
        "metrics": compare_policies(
            full_policy, context_removed_policy, rewards, CONTEXT_DISTRIBUTION
        ),
        "comparison_rows": _comparison_rows(
            full_policy,
            rewards,
            CONTEXT_DISTRIBUTION,
            predictive_selected_context,
            prototype_selected_context,
        ),
        "context_fraction_removed": context_reduction_fraction_removed(1, 0),
        "context_fraction_retained": context_retention_fraction(1, 0),
    }


def _comparison_rows(
    full_policy: np.ndarray,
    rewards: np.ndarray,
    context_distribution: list[float],
    predictive_selected_context: list[str],
    prototype_selected_context: list[str],
) -> list[dict[str, object]]:
    """Build all methods from one reward table and context distribution."""
    return [
        evaluate_policy_method(
            "full_context_reference",
            ORIGINAL_CONTEXT,
            full_policy,
            rewards,
            context_distribution,
            ORIGINAL_CONTEXT,
        ),
        evaluate_policy_method(
            "predictive_only",
            predictive_selected_context,
            full_policy,
            rewards,
            context_distribution,
            ORIGINAL_CONTEXT,
        ),
        evaluate_policy_method(
            "prototype_context_reduction",
            prototype_selected_context,
            full_policy,
            rewards,
            context_distribution,
            ORIGINAL_CONTEXT,
        ),
    ]


def run_policy_comparisons() -> list[dict[str, object]]:
    """Return Test A and B rows for the reproducible comparison CSV."""
    return [
        {"test_case": "A", **row}
        for row in run_test_a()["comparison_rows"]
    ] + [
        {"test_case": "B", **row}
        for row in run_test_b()["comparison_rows"]
    ]


def write_comparison_csv(
    rows: list[dict[str, object]], path: Path = RESULTS_PATH
) -> None:
    """Write comparison rows without changing existing baseline plots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "test_case",
        "method",
        "retained_variables",
        "original_context_count",
        "retained_context_count",
        "context_fraction_removed",
        "context_fraction_retained",
        "policy_agreement",
        "expected_reward",
        "full_context_reward",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_test_c() -> dict[str, object]:
    data = generate_confounding_data(seed=SEED)
    unadjusted = estimate_unadjusted_effect(data)
    adjusted = estimate_effect_adjusted_for_z(data)
    true_effect = true_causal_effect()
    return {
        "n_samples": len(data),
        "seed": SEED,
        "true_effect": true_effect,
        "unadjusted_effect": unadjusted,
        "adjusted_effect": adjusted,
        "unadjusted_error": reporting_estimation_error(unadjusted, true_effect),
        "adjusted_error": reporting_estimation_error(adjusted, true_effect),
        "z_action_association": float(data.groupby("Z")["X"].mean().diff().iloc[-1]),
    }


def main() -> None:
    a, b, c = run_test_a(), run_test_b(), run_test_c()
    comparison_rows = run_policy_comparisons()
    write_comparison_csv(comparison_rows)
    print("Test A — irrelevant context")
    print(f"  prototype-selected context: {a['selected_context']}")
    print(
        f"  policy: {a['full_policy'].tolist()}, "
        f"agreement: {a['metrics']['policy_agreement']:.3f}"
    )
    print(
        f"  expected reward: {a['metrics']['full_context_reward']:.3f} "
        f"-> {a['metrics']['reduced_context_reward']:.3f}"
    )
    print("Test B — policy-relevant context")
    print(f"  predictive-only selected context: {b['predictive_selected_context']}")
    print(f"  prototype selected context: {b['prototype_selected_context']}")
    print(
        f"  full policy: {b['full_policy'].tolist()}, "
        f"predictive-only policy: {b['predictive_policy'].tolist()}"
    )
    print(f"  context-removed policy: {b['context_removed_policy'].tolist()}")
    print(
        f"  predictive-only agreement: "
        f"{b['predictive_metrics']['policy_agreement']:.3f}; "
        f"reward: {b['predictive_metrics']['reduced_context_reward']:.3f}"
    )
    print(
        f"  context-removed agreement: {b['metrics']['policy_agreement']:.3f}; "
        f"reward loss: {b['metrics']['reward_loss']:.3f}"
    )
    print("Test C — weakly predictive but confounder")
    print(f"  seed: {c['seed']}, samples: {c['n_samples']}")
    print(f"  true effect: {c['true_effect']:.6f}")
    print(
        f"  unadjusted: {c['unadjusted_effect']:.6f} "
        f"(error {c['unadjusted_error']:.6f})"
    )
    print(
        f"  adjusted for Z: {c['adjusted_effect']:.6f} "
        f"(error {c['adjusted_error']:.6f})"
    )
    print(f"Comparison CSV: {RESULTS_PATH}")
    for row in comparison_rows:
        print(
            f"  {row['test_case']} / {row['method']}: "
            f"retained={row['retained_variables'] or 'none'}, "
            f"removed={row['context_fraction_removed']:.3f}, "
            f"retained_fraction={row['context_fraction_retained']:.3f}, "
            f"agreement={row['policy_agreement']:.3f}, "
            f"reward={row['expected_reward']:.3f}"
        )


if __name__ == "__main__":
    main()
