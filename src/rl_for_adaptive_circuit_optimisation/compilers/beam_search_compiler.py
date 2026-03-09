"""
A compiler performing beam search over a selection of possible
compiler passes.
"""

import heapq
from collections.abc import Callable
from copy import deepcopy

from pytket import Circuit, OpType
from pytket.passes import CustomPass

from .named_compiler import NamedCompiler

from rl_for_adaptive_circuit_optimisation.rewards import global_normalisation_reward


def get_pass_sequence(
    circuit: Circuit,
    max_steps: int,
    action_space: list[NamedCompiler],
    beam_width: int,
    reward_function: Callable[[int, int, int], int],
) -> tuple[int, Circuit, list[NamedCompiler]]:
    """Perform a beam search over the given compilers
        to find the best sequence and compiled circuit.

    :param circuit: Circuit to compile.
    :param max_steps: The maximum number of steps in the search.
    :param action_space: The compiler passes which can be applied.
    :param beam_width: Beam width. The number of circuits surviving each
        step of the search.
    :param reward_function: Reward function to be applied.
    :return: Triple of: the reward for the highest reward circuit found,
        the circuit with the lowest number of 2q gates found during the search,
        and the sequence of passes applied to generate the highest
        reward circuit.
    """

    def circuit_reward_function(
        circuit_original: Circuit, circuit_compiled: Circuit
    ) -> int:
        """Reward function as function of compiled circuit.

        :param circuit_original: Original Circuit.
        :type circuit_original: Circuit
        :param circuit_compiled: Compiled Circuit
        :type circuit_compiled: Circuit
        :return: Reward
        :rtype: int
        """

        return reward_function(
            circuit_original.n_gates_of_type(OpType.ZZPhase),
            circuit_compiled.n_gates_of_type(OpType.ZZPhase),
            circuit.n_gates_of_type(OpType.ZZPhase),
        )

    # (reward, circuit, passes which have been applied)
    beams: list[tuple[int, Circuit, list[NamedCompiler]]] = [(0, deepcopy(circuit), [])]
    for _ in range(max_steps + 1):

        new_beams = beams.copy()
        for acc_reward, circuit_original, pass_seq in beams:

            for action in action_space:

                circuit_compiled = deepcopy(circuit_original)
                circuit_compiled = action.apply(circuit_compiled)

                reward = circuit_reward_function(
                    circuit_original,
                    circuit_compiled,
                )
                new_beams.append(
                    (
                        acc_reward + reward,
                        circuit_compiled,
                        pass_seq + [action],
                    )
                )

        beams = heapq.nlargest(beam_width, new_beams, key=lambda x: x[0])

    return beams[0]


def BeamSearch(
    max_steps: int,
    action_space: list[NamedCompiler],
    beam_width: int,
    reward_function: Callable[[int, int, int], int] = global_normalisation_reward,
) -> NamedCompiler:
    """Compiler performing beam search over the given action space.

    :param max_steps: The maximum number of steps in the search.
    :type max_steps: int
    :param action_space: The compiler passes which can be applied.
    :type action_space: list[NamedCompilers]
    :param beam_width: Beam width. The number of circuits surviving each
        step of the search.
    :type beam_width: int
    :param reward_function: Reward function to be applied.
    :type reward_function: Callable[[int, int, int], int]
    """

    def transform(circuit: Circuit) -> Circuit:
        beam = get_pass_sequence(
            circuit, max_steps, action_space, beam_width, reward_function
        )
        return beam[1]

    return NamedCompiler(
        name="BeamSearch", compiler_pass=CustomPass(transform=transform)
    )
