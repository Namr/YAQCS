from yaqcs import RXGate, RXXGate, Circuit, RYGate, RZGate
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
    RYGate([pi / 2]),
    RYGate([-pi / 2]),
    RZGate([pi / 2]),
    RZGate([-pi / 2]),
]

timerS = time.time()

r2o2 = sqrt(2.0) / 2.0
ww = r2o2 + r2o2 * 1.0j
www = -r2o2 + -r2o2 * 1.0j
target_unitary = [[www, 0, 0, 0], [0, www, 0, 0], [0, 0, 0, ww], [0, 0, ww, 0]]
'''
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

'''


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
    gCount = 0
    print(len(list(itertools.product(gates_with_positions, repeat=max_depth))))

    for circuit in itertools.product(gates_with_positions, repeat=max_depth):
        count = count + 1
        gCount = gCount + 1
        if gCount % 1000 == 0:
            print(time.time() - timerS)
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
        comm.Barrier()
    else:
        pure_brute_force(2, allowed_gates, 5)
