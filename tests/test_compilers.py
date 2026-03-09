from rl_for_adaptive_circuit_optimisation.compilers import (
    RebaseKAKDecomposition,
    RebaseCliffordResynthesis,
    RebaseCliffordSimp,
    RebaseZXGraphlikeOptimisation,
    RebasePauliSimp,
    RebaseThreeQubitSquash,
    RebaseFullPeephole,
    BeamSearch, 
    RLRebase,
    DoNothing
)
from pytket import Circuit
import pytest


@pytest.mark.parametrize(
    ("optimiser"),
    [
        RebaseKAKDecomposition(),
        RebaseCliffordResynthesis(),
        RebaseCliffordSimp(),
        RebaseZXGraphlikeOptimisation(),
        RebasePauliSimp(),
        RebaseThreeQubitSquash(),
        RebaseFullPeephole(),
        BeamSearch(max_steps=10, action_space=[RebaseKAKDecomposition(), RebaseCliffordResynthesis()], beam_width=2),
        DoNothing(),
    ],
)
def test_compilers_run(optimiser) -> None:
    """Test that all compilers run without error on a simple circuit."""
    original_circuit = Circuit(2).CX(0, 1).Rz(0.1, 1).CX(0, 1)
    rebased_circuit = RLRebase().apply(original_circuit)
    optimiser.apply(rebased_circuit)