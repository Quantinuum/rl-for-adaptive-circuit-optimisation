"""
Collection of compilers for selection
"""

from pytket.circuit import Circuit, OpType
from pytket.extensions.quantinuum import QuantinuumAPIOffline, QuantinuumBackend
from pytket.passes import (
    AutoRebase,
    CliffordResynthesis,
    CliffordSimp,
    CommuteThroughMultis,
    CustomPass,
    EulerAngleReduction,
    FullPeepholeOptimise,
    GreedyPauliSimp,
    KAKDecomposition,
    RebaseTket,
    RemoveRedundancies,
    RepeatPass,
    SequencePass,
    ZXGraphlikeOptimisation,
)
from pytket.transform import Transform

from .gateset import GATESET
from .named_compiler import NamedCompiler


def RLRebase() -> NamedCompiler:
    """Default rebase pass for repository."""
    return NamedCompiler(name="RLRebase", compiler_pass=AutoRebase(gateset=GATESET))


def RebaseRemoveRedundancies() -> NamedCompiler:
    """
    Returns pass removing redundancies and rebasing.
    """
    compiler_pass = SequencePass(
        pass_list=[RemoveRedundancies(), RLRebase().compiler_pass],
    )
    return NamedCompiler(name="RebaseRemoveRedundancies", compiler_pass=compiler_pass)


def RebaseCommuteThroughMultis() -> NamedCompiler:
    """
    Returns pass commuting single qubit gates through two qubit gates and rebasing.
    """
    compiler_pass = SequencePass(
        pass_list=[CommuteThroughMultis(), RLRebase().compiler_pass],
    )
    return NamedCompiler(name="RebaseCommuteThroughMultis", compiler_pass=compiler_pass)


def RebaseCleanUp() -> NamedCompiler:
    """
    Returns a pass that rebases the circuit to the given gateset then removes redundancies
    """
    clean_up = SequencePass(
        pass_list=[
            RebaseRemoveRedundancies().compiler_pass,
            RebaseCommuteThroughMultis().compiler_pass,
        ]
    )
    compiler_pass = SequencePass(
        pass_list=[RLRebase().compiler_pass, RepeatPass(compilation_pass=clean_up)]
    )
    return NamedCompiler(name="RebaseCleanUp", compiler_pass=compiler_pass)


def RebaseCliffordResynthesis() -> NamedCompiler:
    """
    Returns pass resynthesising clifford subcircuits and rebasing.
    """
    compiler_pass = SequencePass(
        pass_list=[
            AutoRebase({OpType.ZZPhase, OpType.Rx, OpType.Rz}),
            EulerAngleReduction(p=OpType.Rx, q=OpType.Rz),
            CliffordResynthesis(),
            RLRebase().compiler_pass,
        ],
    )
    return NamedCompiler(name="RebaseCliffordResynthesis", compiler_pass=compiler_pass)


def RebaseKAKDecomposition() -> NamedCompiler:
    """
    Returns a pass that simplifies the circuit by applying
    KAKDecomposition.
    """
    compiler_pass = SequencePass(
        pass_list=[
            KAKDecomposition(target_2qb_gate=OpType.TK2),
            RLRebase().compiler_pass,
        ],
    )
    return NamedCompiler(name="RebaseKAKDecomposition", compiler_pass=compiler_pass)


def RebaseZXGraphlikeOptimisation() -> NamedCompiler:
    """
    Returns a pass that simplifies the circuit by applying
    ZXGraphlikeOptimisation.
    """
    zx_optimisation = ZXGraphlikeOptimisation()

    gateset = zx_optimisation.get_gate_set()
    assert gateset is not None, "ZXGraphlikeOptimisation has no gateset"

    compiler_pass = SequencePass(
        pass_list=[
            AutoRebase(gateset=gateset),
            zx_optimisation,
            RLRebase().compiler_pass,
        ],
        strict=False,
    )
    return NamedCompiler(
        name="RebaseZXGraphlikeOptimisation", compiler_pass=compiler_pass
    )


def RebaseCliffordSimp() -> NamedCompiler:
    """
    Returns a pass that simplifies the circuit by applying Clifford simplification
    """
    compiler_pass = SequencePass(
        pass_list=[
            # This rebase into TKET, and CX in particular, is required
            # as otherwise CliffordSimp will not recognise 2 qubit
            # Clifford gates. This may be resolved in newer versions
            # of pytket.
            RebaseTket(),
            CliffordSimp(target_2qb_gate=OpType.TK2),
            RLRebase().compiler_pass,
        ],
    )
    return NamedCompiler(name="RebaseCliffordSimp", compiler_pass=compiler_pass)


def RebasePauliSimp() -> NamedCompiler:
    """
    Returns a pass that simplifies the circuit by applying Pauli simplification
    """
    pauli_simp = GreedyPauliSimp(allow_zzphase=True)

    gateset = pauli_simp.get_gate_set()
    assert gateset is not None, "GreedyPauliSimp has no gateset."

    compiler_pass = SequencePass(
        pass_list=[
            AutoRebase(gateset=gateset),
            pauli_simp,
            RLRebase().compiler_pass,
        ],
        strict=False,
    )
    return NamedCompiler(name="RebasePauliSimp", compiler_pass=compiler_pass)


def RebaseFullPeephole() -> NamedCompiler:
    """
    Return pass that simplifies circuit by applying FullPeepholeOptimiser.
    """
    compiler_pass = SequencePass(
        pass_list=[
            FullPeepholeOptimise(target_2qb_gate=OpType.TK2),
            RLRebase().compiler_pass,
        ]
    )
    return NamedCompiler(name="RebaseFullPeephole", compiler_pass=compiler_pass)


def CleanUpQuantinuumDefault(
    optimisation_level: int = 3, device_name: str = "H2-1LE"
) -> NamedCompiler:
    """Returns a pass performing default Quantinuum compilation.

    :param optimisation_level: Quantinuum optimisation level, defaults to 3
    :type optimisation_level: int, optional
    :param device_name: Name of the Quantinuum device, defaults to "H2-1LE"
    :type device_name: str, optional
    :return: Pass performing default Quantinuum compilation.
    :rtype: NamedCompiler
    """
    compiler_pass = SequencePass(
        pass_list=[
            QuantinuumBackend(
                device_name=device_name, api_handler=QuantinuumAPIOffline()  # type: ignore
            ).default_compilation_pass(optimisation_level=optimisation_level),
            RebaseCleanUp().compiler_pass,
        ]
    )
    return NamedCompiler(name="CleanUpQuantinuumDefault", compiler_pass=compiler_pass)


def RebaseThreeQubitSquash() -> NamedCompiler:
    """
    Returns a pass that simplifies the circuit by applying
    ThreeQubitSquash.
    """

    def ThreeQubitSquash():

        def transform(circuit):

            circuit_copy = circuit.copy()

            gateset = set(cmd.op.type for cmd in circuit_copy if len(cmd.args) == 1)
            gateset.add(OpType.TK2)
            gateset.add(OpType.TK1)

            AutoRebase(gateset=gateset).apply(circuit_copy)
            Transform.ThreeQubitSquash(target_2qb_gate=OpType.TK2).apply(circuit_copy)

            return circuit_copy

        return CustomPass(transform=transform)

    compiler_pass = SequencePass(
        pass_list=[
            ThreeQubitSquash(),
            RLRebase().compiler_pass,
        ],
        strict=False,
    )
    return NamedCompiler(name="RebaseThreeQubitSquash", compiler_pass=compiler_pass)


def DoNothing() -> NamedCompiler:
    """
    Returns a pass which does nothing
    """

    def do_nothing(circuit: Circuit) -> Circuit:
        """Returns given circuit.

        :param circuit: A circuit to immediately return.
        :type circuit: Circuit
        :return: Original circuit.
        :rtype: Circuit
        """
        return circuit

    return NamedCompiler(
        name="DoNothing",
        compiler_pass=CustomPass(transform=do_nothing),
    )
