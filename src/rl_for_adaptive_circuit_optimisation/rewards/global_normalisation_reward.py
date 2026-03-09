def global_normalisation_reward(
    num_2q_gates_before: int, num_2q_gates_after: int, normalisation: int = 1
) -> float:
    """Reward normalised with the initial number of 2-qubit gates at the start of the circuit.

    Args:
        num_2q_gates_before (int): The number of 2-qubit gates before the pass
        num_2q_gates_after (int): The number of 2-qubit gates after the pass
    Returns:
        float: The reward, which is the difference between the number of 2-qubit gates
        before and after the pass
    """
    if normalisation == 0:
        normalisation = 1  # avoid division by zero
    return (num_2q_gates_before - num_2q_gates_after) / normalisation