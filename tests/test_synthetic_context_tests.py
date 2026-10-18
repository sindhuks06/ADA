import numpy as np

from experiments.synthetic_context_tests import run_test_a, run_test_b, run_test_c
from src.context_reduction import reduce_context
from src.evaluation import (
    context_reduction_fraction_removed,
    context_retention_fraction,
    policy_agreement,
)
from src.synthetic_data import generate_confounding_data, true_causal_effect


def test_test_a_has_same_optimal_action_for_both_contexts():
    result = run_test_a()
    assert result["full_policy"].tolist() == [0, 0]
    assert result["metrics"]["policy_agreement"] == 1.0


def test_test_b_has_different_optimal_actions():
    assert run_test_b()["full_policy"].tolist() == [0, 1]


def test_removing_policy_relevant_context_loses_reward():
    result = run_test_b()
    assert result["metrics"]["reward_loss"] > 0
    assert result["metrics"]["reduced_context_reward"] < result["metrics"]["full_context_reward"]


def test_retaining_policy_relevant_context_matches_full_policy():
    result = run_test_b()
    assert result["predictive_selected_context"] == ["C"]
    assert result["predictive_policy"].tolist() == result["full_policy"].tolist()
    assert result["predictive_metrics"]["policy_agreement"] == 1.0
    assert (
        result["predictive_metrics"]["reduced_context_reward"]
        == result["predictive_metrics"]["full_context_reward"]
    )


def test_reduction_retains_protected_variables_and_order():
    selected = reduce_context(["Z", "A", "B"], ["Z"], {"A": 0.8, "B": 0.2}, threshold=0.5)
    assert selected == ["Z", "A"]


def test_policy_agreement_and_context_fractions():
    assert policy_agreement([0, 1], [0, 0], [0.25, 0.75]) == 0.25
    assert context_reduction_fraction_removed(4, 3) == 0.25
    assert context_retention_fraction(4, 3) == 0.75


def test_confounding_data_follows_binary_dgp():
    data = generate_confounding_data(n_samples=1000, seed=7)
    assert set(data.columns) == {"Z", "X", "Y"}
    assert set(np.unique(data["Z"])) <= {0, 1}
    assert data.groupby("Z")["X"].mean().loc[1] > data.groupby("Z")["X"].mean().loc[0]


def test_causal_estimates_are_compared_with_known_effect():
    result = run_test_c()
    assert result["true_effect"] == true_causal_effect()
    assert result["unadjusted_error"] >= 0
    assert result["adjusted_error"] >= 0
    assert result["adjusted_error"] < result["unadjusted_error"]
