def reduce_context(
    candidate_context,
    protected_variables,
    scores,
    threshold=0.5,
):
    """
    Return a reduced context while retaining protected variables.
    """
    selected = [
        variable
        for variable in candidate_context
        if variable in protected_variables
        or scores.get(variable, 0.0) >= threshold
    ]

    return selected