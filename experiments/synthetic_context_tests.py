"""Run controlled synthetic experiments for conditioning-context reduction."""

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


def run_test_a() -> dict[str, object]:
    contexts, actions = [0, 1], [0, 1]
    selected_context = reduce_context(["C"], [], {"C": 0.0})
    rewards = compute_true_expected_rewards(
        contexts, actions, lambda _context, action: 0.7 if action == 0 else 0.3
    )
    full_policy = compute_optimal_policy(rewards)
    reduced_policy = np.repeat(full_policy[0], len(contexts))
    return {
        "expected_rewards": rewards,
        "full_policy": full_policy,
        "reduced_policy": reduced_policy,
        "selected_context": selected_context,
        "metrics": compare_policies(full_policy, reduced_policy, rewards, [0.5, 0.5]),
        "context_fraction_removed": context_reduction_fraction_removed(
            1, len(selected_context)
        ),
        "context_fraction_retained": context_retention_fraction(1, len(selected_context)),
    }


def run_test_b() -> dict[str, object]:
    contexts, actions = [0, 1], [0, 1]
    predictive_selected_context = reduce_context(["C"], [], {"C": 0.9})
    rewards = compute_true_expected_rewards(
        contexts,
        actions,
        lambda context, action: (
            0.8 if context == action else 0.2
        ),
    )
    full_policy = compute_optimal_policy(rewards)
    context_distribution = [0.5, 0.5]
    # The best context-free action under the context distribution is action 0
    # (ties are resolved by the documented low-index rule).
    reduced_action = int(np.argmax(np.dot(context_distribution, rewards)))
    context_removed_policy = np.repeat(reduced_action, len(contexts))
    predictive_policy = compute_optimal_policy(rewards)
    return {
        "expected_rewards": rewards,
        "full_policy": full_policy,
        "reduced_policy": context_removed_policy,
        "context_removed_policy": context_removed_policy,
        "predictive_selected_context": predictive_selected_context,
        "predictive_policy": predictive_policy,
        "predictive_metrics": compare_policies(
            full_policy, predictive_policy, rewards, context_distribution
        ),
        "metrics": compare_policies(
            full_policy, context_removed_policy, rewards, context_distribution
        ),
        "context_fraction_removed": context_reduction_fraction_removed(1, 0),
        "context_fraction_retained": context_retention_fraction(1, 0),
    }


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
    print("Test A — irrelevant context")
    print(f"  prototype-selected context: {a['selected_context']}")
    print(f"  policy: {a['full_policy'].tolist()}, agreement: {a['metrics']['policy_agreement']:.3f}")
    print(f"  expected reward: {a['metrics']['full_context_reward']:.3f} -> {a['metrics']['reduced_context_reward']:.3f}")
    print(
        f"  context fraction removed: {a['context_fraction_removed']:.3f}; "
        f"retained: {a['context_fraction_retained']:.3f}"
    )
    print("Test B — policy-relevant context")
    print(f"predictive-only selected context: {b['predictive_selected_context']}")
    print(
        f"  full policy: {b['full_policy'].tolist()}, "
        f"predictive-only policy: {b['predictive_policy'].tolist()}"
    )
    print(
        f"  context-removed policy: {b['context_removed_policy'].tolist()}"
    )
    print(
        f"  predictive-only agreement: "
        f"{b['predictive_metrics']['policy_agreement']:.3f}; "
        f"reward: {b['predictive_metrics']['reduced_context_reward']:.3f}"
    )
    print(f"  agreement: {b['metrics']['policy_agreement']:.3f}, reward loss: {b['metrics']['reward_loss']:.3f}")
    print(
        f"  context fraction removed: {b['context_fraction_removed']:.3f}; "
        f"retained: {b['context_fraction_retained']:.3f}"
    )
    print("Test C — weak outcome predictor but confounder")
    print(f"  seed: {c['seed']}, samples: {c['n_samples']}")
    print(f"  true effect: {c['true_effect']:.6f}")
    print(f"  unadjusted: {c['unadjusted_effect']:.6f} (error {c['unadjusted_error']:.6f})")
    print(f"  adjusted for Z: {c['adjusted_effect']:.6f} (error {c['adjusted_error']:.6f})")


if __name__ == "__main__":
    main()
