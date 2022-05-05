from __future__ import annotations
from bqskit import Circuit
from bqskit import Compiler
from bqskit import UnitaryMatrix
from bqskit import CompilationTask
from bqskit.passes import QSearchSynthesisPass, QFASTDecompositionPass, UnfoldPass, ForEachBlockPass, ScanningGateRemovalPass
from bqskit.ir.gates import RXXGate, RXGate, RYGate, RZGate, CircuitGate, U3Gate, CNOTGate
from bqskit.passes.search import SimpleLayerGenerator, LayerGenerator
from bqskit.compiler.basepass import BasePass
from bqskit2qiskit import bqskit2qiskit
from qiskit import QuantumCircuit, assemble, Aer, QuantumRegister, ClassicalRegister, transpile
from math import pi, sqrt
from qiskit.visualization import plot_bloch_multivector, plot_histogram, array_to_latex, circuit_drawer
import numpy as np
from rxxx import RXXXGate

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gate import Gate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import U3Gate
from bqskit.passes.search.generator import LayerGenerator
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

_logger = logging.getLogger(__name__)


def print_circ(circuit):
    op_str = ""
    for op in list(circuit.operations()):
        op_str += str(op) + " " + str(op.params[0]) + ","
    print(op_str)


class ThreeLayerGenerator(LayerGenerator):
    def __init__(
        self,
        three_qudit_gate: Gate = RXXXGate(),
        two_qudit_gate: Gate = RXXGate(),
        single_qudit_gate_1: Gate = U3Gate(),
        single_qudit_gate_2: Gate | None = None,
        initial_layer_gate: Gate | None = None,
    ) -> None:
        if single_qudit_gate_2 is None:
            single_qudit_gate_2 = single_qudit_gate_1

        if initial_layer_gate is None:
            initial_layer_gate = single_qudit_gate_1

        two_radix_1 = two_qudit_gate.radixes[0]
        two_radix_2 = two_qudit_gate.radixes[1]

        self.three_qudit_gate = three_qudit_gate
        self.two_qudit_gate = two_qudit_gate
        self.single_qudit_gate_1 = single_qudit_gate_1
        self.single_qudit_gate_2 = single_qudit_gate_2
        self.initial_layer_gate = initial_layer_gate

    def gen_initial_layer(
        self,
        target: UnitaryMatrix | StateVector,
        data: dict[str, Any],
    ) -> Circuit:
        """
        Generate the initial layer, see LayerGenerator for more.

        Raises:
            ValueError: If `target` has a radix mismatch with
                `self.initial_layer_gate`.
        """

        if not isinstance(target, (UnitaryMatrix, StateVector)):
            raise TypeError(
                'Expected unitary or state, got %s.' % type(target), )

        for radix in target.radixes:
            if radix != self.initial_layer_gate.radixes[0]:
                raise ValueError(
                    'Radix mismatch between target and initial_layer_gate.', )
        init_circuit = Circuit(target.num_qudits, target.radixes)
        for i in range(init_circuit.num_qudits):
            init_circuit.append_gate(self.initial_layer_gate, [i])
        return init_circuit

    def gen_successors(
        self,
        circuit: Circuit,
        data: dict[str, Any],
    ) -> list[Circuit]:
        """
        Generate the successors of a circuit node.

        Raises:
            ValueError: If circuit is a single-qudit circuit.
        """

        if not isinstance(circuit, Circuit):
            raise TypeError('Expected circuit, got %s.' % type(circuit))

        if circuit.num_qudits < 3:
            raise ValueError('Cannot expand a single-qudit circuit.')

        # Get the machine model
        model = BasePass.get_model(circuit, data)

        # Generate successors that use the three qudit gate
        # does this by sliding the gate down the circuit
        # this won't allow for three non-consququitive qubits though
        successors = []
        num_windows = circuit.num_qudits - 2
        for i in range(num_windows):
            successor = circuit.copy()
            successor.append_gate(self.three_qudit_gate, [0 + i, 1 + i, 2 + i])
            successor.append_gate(self.single_qudit_gate_1, 0 + i)
            successor.append_gate(self.single_qudit_gate_2, 1 + i)
            successor.append_gate(self.single_qudit_gate_2, 2 + i)
            successors.append(successor)

        # Generate two qudit successors as normal
        for edge in model.coupling_graph:
            successor = circuit.copy()
            successor.append_gate(self.two_qudit_gate, [edge[0], edge[1]])
            successor.append_gate(self.single_qudit_gate_1, edge[0])
            successor.append_gate(self.single_qudit_gate_2, edge[1])
            successors.append(successor)

        #print("one pass done with circuit: ")
        # print_circ(circuit)
        return successors


simp_layer_gen = ThreeLayerGenerator(three_qudit_gate=RXXXGate(),
                                     two_qudit_gate=RXXGate(),
                                     single_qudit_gate_1=U3Gate())

configed_search = QSearchSynthesisPass(layer_generator=simp_layer_gen)

configured_qfast_pass = QFASTDecompositionPass()

# autopep8: off
cccx_ancilla_unitary = UnitaryMatrix([[
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0
],
                                      [
                                          0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 1
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          1, 0, 0, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 1, 0, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 1, 0, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 1, 0, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 1, 0, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 1, 0, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 1, 0
                                      ],
                                      [
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                          0, 0, 0, 0, 0, 0, 0, 0
                                      ]])

# X-Gate with three controls (CCCX)
cccx_unitary = UnitaryMatrix([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
])

# Encode the toffoli unitary
toffoli_unitary = UnitaryMatrix([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0],
])
toffoli_ancilla_unitary = UnitaryMatrix(
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])

cx_unitary = UnitaryMatrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
])
# autopep8: on

chosen_unitary = toffoli_unitary

circuit = Circuit.from_unitary(chosen_unitary)

if __name__ == '__main__':
    with Compiler() as compiler:
        task = CompilationTask(circuit, [configed_search])
        '''
        task = CompilationTask(circuit, [
            configured_qfast_pass,
            ForEachBlockPass([configed_search]),
            UnfoldPass(),
            ScanningGateRemovalPass(),
        ])
        '''
        synthesized_circuit = compiler.compile(task)
        print(
            "% error: ",
            synthesized_circuit.get_unitary().get_distance_from(
                chosen_unitary))

        sim = Aer.get_backend('aer_simulator_unitary')
        qc = bqskit2qiskit(synthesized_circuit)
        circuit_drawer(qc, output='mpl', filename="synthOutput")
        transpiled_qc = transpile(qc, sim)
        result = sim.run(transpiled_qc).result()
        unitary = result.get_unitary(transpiled_qc)
        print("Circuit unitary:\n", unitary.round(5))
