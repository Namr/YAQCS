from yaqcs import RXGate, RXXGate, Circuit
from math import pi, sqrt
import copy
import time
import itertools
import numpy as np

USING_MPI = False
if USING_MPI:
    import threading
    from mpi4py import MPI

allowed_gates = [
    RXXGate([pi / 2]),
    RXGate([pi / 2]),
    RXGate([-pi / 2]),
]

#target_unitary = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
target_unitary = [[
    0.35355 - 0.35355j, -0.35355 + 0.35355j, 0.35355 - 0.35355j,
    0.35355 - 0.35355j
],
                  [
                      -0.35355 + 0.35355j, 0.35355 - 0.35355j,
                      0.35355 - 0.35355j, 0.35355 - 0.35355j
                  ],
                  [
                      0.35355 - 0.35355j, 0.35355 - 0.35355j,
                      0.35355 - 0.35355j, -0.35355 + 0.35355j
                  ],
                  [
                      0.35355 - 0.35355j, 0.35355 - 0.35355j,
                      -0.35355 + 0.35355j, 0.35355 - 0.35355j
                  ]]


def test_circuit(qubits: int, circuit: list) -> bool:
    c = Circuit(qubits, qiskit_style=False)
    for gate in circuit:
        if gate[0].num_qbits > qubits - gate[1]:
            return False
        c.append_gate(gate[0], gate[1])

    if (np.around(c.get_unitary(), 3) == np.around(target_unitary, 3)).all():
        print("============================================")
        print("found a solution with the following unitary")
        print(c.get_unitary())
        print("created with this circuit")
        print(circuit)
        print("============================================")
        return True
    return False


def pure_brute_force(qubits: int,
                     gates: list,
                     max_depth: int,
                     stride: int = 1) -> bool:

    gates_with_positions = list(itertools.product(allowed_gates,
                                                  range(qubits)))

    count = 0
    for circuit in itertools.product(gates_with_positions, repeat=max_depth):
        count = count + 1
        if count == stride:
            count = 0
            if test_circuit(qubits, circuit):
                return True

    return False


if __name__ == "__main__":
    if USING_MPI:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

    pure_brute_force(2, allowed_gates, 3, stride=rank - 1)

    if USING_MPI:
        comm.Barrier()
