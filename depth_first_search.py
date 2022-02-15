from yaqcs import RXGate, RXXGate, Circuit, RYGate, RZGate, unitary_distance, Gate
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


def test_circuit(qubits: int, circuit: list) -> float:
    c = Circuit(qubits, qiskit_style=False)
    for gate in circuit:
        if gate[0].num_qbits > qubits - gate[1]:
            return False
        c.append_gate(gate[0], gate[1])

    unitary = c.get_unitary()
    if (np.around(c.get_unitary(), 3) == np.around(target_unitary, 3)).all():
        print("============================================")
        print("found a solution with the following unitary")
        print(c.get_unitary())
        print("created with this circuit")
        for g in circuit:
            print(g[0].__class__.__name__ + " angle: " + str(g[0].params[0]))
        print("============================================")
        return 0.0
    else:
        return unitary_distance(unitary, target_unitary)


def find_min_dist_choice(qubits: int,
                         circuit: list,
                         possibilities: list,
                         blacklist: list,
                         depth: int = 1) -> Gate:
    best_dist = 100000000
    best_gate = None

    possibilities = list(itertools.product(possibilities, repeat=depth))

    for gate in possibilities:
        # try this gate
        circuit += gate
        dist = test_circuit(qubits, circuit)
        # if this has the best performance so far, log it
        if dist < best_dist and gate not in blacklist:
            best_dist = dist
            best_gate = gate

        # cleanup before next iteration
        for i in range(depth):
            circuit.pop()

    return best_gate


def depth_first_search(qubits: int,
                       gates: list,
                       max_depth: int,
                       stride: int = 1) -> bool:

    gates_with_positions = list(itertools.product(allowed_gates,
                                                  range(qubits)))
    gates_with_positions = [
        gate for gate in gates_with_positions
        if not (gate[0].__class__.__name__ == "RXXGate" and gate[1] == 1)
    ]

    for d in range(max_depth):
        stack = []
        blacklist = []
        print("did one iter")
        while len(blacklist) < len(gates_with_positions)**(d + 1):
            # add the next best gate for this step
            start_set = find_min_dist_choice(qubits,
                                             stack,
                                             gates_with_positions,
                                             blacklist,
                                             depth=d + 1)
            stack += start_set

            # blindly add the minimum distance choices until you reach the desired depth or get a full unitary match
            while len(stack) < max_depth:
                stack += find_min_dist_choice(qubits, stack,
                                              gates_with_positions, [])

            print("===========================")
            print(stack)
            print("===========================")

            # if we have gotten to this point without quitting it means that this set of choices did not work
            # clear the circuit
            stack = []

            blacklist.append(start_set)

    return False


if __name__ == "__main__":
    depth_first_search(2, allowed_gates, 5)
