"""
Standard gateset for H-series pass selection
"""

from pytket.circuit import OpType
from pytket.passes import AutoRebase
from pytket.predicates import GateSetPredicate, Predicate

GATESET = {OpType.ZZPhase, OpType.Rz, OpType.PhasedX}


def MLCPSGateSetPredicate() -> Predicate:
    """Predicate to check that circuit is in
    Machine Learning Compiler Pass Sequence gateset.
    """
    return GateSetPredicate(allowed_types=GATESET)


def MLCPSRebasePass():
    """Machine Learning Compiler Pass Sequence rebase pass.
    Rebases to ZZPhase, Rz, PhasedX gateset.
    """
    return AutoRebase(gateset=GATESET)
