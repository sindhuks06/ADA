"""Small, explicitly non-causal context-selection prototype."""

from collections.abc import Mapping, Sequence
import math


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