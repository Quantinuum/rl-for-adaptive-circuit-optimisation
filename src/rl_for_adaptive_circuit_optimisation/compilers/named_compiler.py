"""
Base class for compilers
"""

from dataclasses import dataclass

from pytket.circuit import Circuit
from pytket.passes import BasePass


@dataclass
class NamedCompiler:
    """Data class containing the compiler to act and its name."""

    name: str
    compiler_pass: BasePass

    def apply(self, circuit: Circuit) -> Circuit:
        """Apply compiler pass. Note that the compiled circuit is returned,
        rather than the compilation happening in place.

        :param circuit: Circuit to apply pass to.
        :type circuit: Circuit
        :return: Compiled circuit.
        :rtype: Circuit
        """
        compiled_circuit = circuit.copy()
        self.compiler_pass.apply(compiled_circuit)
        return compiled_circuit

    def __str__(self):
        return self.name
