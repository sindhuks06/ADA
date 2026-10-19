import numpy as np
import pytest

from src.context_reduction import reward_preserving_backward_elimination


def _run(contexts, rewards, epsilon=0.0, variables=("A", "B")):
    return reward_preserving_backward_elimination(
        variables,
        contexts,
        np.asarray(rewards, dtype=float),
        [1 / len(contexts)] * len(contexts),
        epsilon=epsilon,
    )


def test_irrelevant_context_is_removed():
    result = _run([(0, 0), (0, 1), (1, 0), (1, 1)], [[1, 0]] * 4)
    assert result["selected_context"] == []
    assert result["selected_reward"] == result["reference_reward"] == 1.0
    assert result["retained_variables"] == 0


def test_policy_relevant_context_is_retained():
    result = _run([(0, 0), (0, 1), (1, 0), (1, 1)], [[1, 0], [1, 0], [0, 1], [0, 1]])
    assert result["selected_context"] == ["A"]
    assert result["policy_agreement"] == 1.0


def test_epsilon_allows_a_small_reward_loss():
    result = _run([(0,), (1,)], [[1.0, 0.0], [0.9, 1.0]], epsilon=0.1, variables=("A",))
    assert result["selected_context"] == []
    assert np.isclose(result["reward_gap"], 0.05)
    assert result["feasible"] is True


def test_reward_gap_is_reference_minus_selected_reward():
    result = _run(
        [(0,), (1,)], [[1.0, 0.0], [0.0, 1.0]], epsilon=0.5, variables=("A",)
    )
    assert np.isclose(result["reward_gap"], 0.5)
    assert np.isclose(result["selected_reward"], 0.5)


def test_ties_are_broken_by_lexicographically_first_retained_context():
    result = _run([(0, 0), (1, 1)], [[1, 0], [0, 1]])
    # Both one-variable candidates are equivalent; removing B is deterministic.
    assert result["selected_context"] == ["A"]


def test_interaction_requires_both_variables():
    contexts = [(0, 0), (0, 1), (1, 0), (1, 1)]
    rewards = [[1, 0], [0, 1], [0, 1], [1, 0]]
    result = _run(contexts, rewards)
    assert result["selected_context"] == ["A", "B"]
    assert result["selected_reward"] == result["reference_reward"]


def test_selection_and_evaluation_tables_are_separate():
    result = reward_preserving_backward_elimination(
        ["A"],
        [(0,), (1,)],
        np.asarray([[1, 0], [0, 1]], dtype=float),
        [0.5, 0.5],
        evaluation_expected_rewards=np.asarray([[1, 0], [0.2, 1]], dtype=float),
    )
    assert result["selected_context"] == ["A"]
    assert np.isclose(result["reference_reward"], 1.0)


def test_selection_feasibility_is_distinct_from_evaluation_tolerance():
    result = reward_preserving_backward_elimination(
        ["A"],
        [(0,), (1,)],
        np.asarray([[1, 0], [0, 1]], dtype=float),
        [0.5, 0.5],
        epsilon=0.5,
        evaluation_expected_rewards=np.asarray([[1, 0], [0, 10]], dtype=float),
    )
    assert result["selected_context"] == []
    assert result["selection_feasible"] is True
    assert result["feasible"] is True
    assert result["evaluation_within_epsilon"] is False
    assert result["evaluation_reward_gap"] > 0.5
    assert result["reward_gap"] == result["evaluation_reward_gap"]


def test_zero_probability_context_group_is_rejected():
    with pytest.raises(ValueError, match="zero-probability"):
        reward_preserving_backward_elimination(
            ["A"],
            [(0,), (1,)],
            np.asarray([[1, 0], [0, 1]], dtype=float),
            [1.0, 0.0],
        )


def test_zero_probability_group_is_rejected_even_when_variable_is_retained():
    with pytest.raises(ValueError, match="zero-probability"):
        reward_preserving_backward_elimination(
            ["A"],
            [(0,), (1,)],
            np.asarray([[1, 0], [0, 1]], dtype=float),
            [1.0, 0.0],
        )
