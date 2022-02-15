from yaqcs import RXGate, RXXGate, Circuit, RYGate, RZGate, unitary_distance
import math
from math import pi
import copy
import time
import itertools
import numpy as np

r2o2 = math.sqrt(2.0) / 2.0
ww = r2o2 + r2o2 * 1.0j
www = -r2o2 + -r2o2 * 1.0j
target_unitary = [[-www, 0, 0, 0], [0, -www, 0, 0], [0, 0, 0, ww],
                  [0, 0, ww, 0]]

allowed_gates = [
    RXXGate([pi / 2]),
    RXGate([pi / 2]),
    RXGate([-pi / 2]),
    RYGate([pi / 2]),
    RYGate([-pi / 2]),
    RZGate([pi / 2]),
    RZGate([-pi / 2]),
]


def quick_dist(circuit):
    print("========================================")
    print(np.around(circuit.get_unitary(), 2))
    print("distance: " +
          str(unitary_distance(circuit.get_unitary(), target_unitary)))
    print("========================================")


def lowest_at_step(input_circuit, pos=0):
    lowest = 1.0
    lowestGate = None
    for gate in allowed_gates:
        circuit = copy.deepcopy(input_circuit)
        circuit.append_gate(gate, pos)
        dist = unitary_distance(circuit.get_unitary(), target_unitary)
        if dist < lowest:
            lowest = dist
            lowestGate = gate
    print(str(lowest) + " with gate: " + str(lowestGate))


if __name__ == "__main__":
    rx = RXGate([-math.pi / 2.0])
    ry = RYGate([-math.pi / 2.0])
    ryo = RYGate([math.pi / 2.0])
    rz = RZGate([-math.pi / 2.0])
    rxx = RXXGate([math.pi / 2.0])

    circuit = Circuit(2, qiskit_style=False)
    circuit.append_gate(ryo, 0)
    circuit.append_gate(rxx, 0)
    circuit.append_gate(rx, 0)
    circuit.append_gate(ry, 0)
    circuit.append_gate(rx, 1)
    quick_dist(circuit)

    circuit = Circuit(2, qiskit_style=False)
    print("step 1: ")
    print("lowest with dist:")
    lowest_at_step(circuit)
    circuit.append_gate(ryo, 0)
    print("true value: ")
    print(
        str(unitary_distance(circuit.get_unitary(), target_unitary)) +
        " with gate: " + str(ryo))

    print("step 2: ")
    print("lowest with dist:")
    lowest_at_step(circuit)
    circuit.append_gate(rxx, 0)
    print("true value: ")
    print(
        str(unitary_distance(circuit.get_unitary(), target_unitary)) +
        " with gate: " + str(rxx))

    print("step 3: ")
    print("lowest with dist:")
    lowest_at_step(circuit)
    circuit.append_gate(rx, 0)
    print("true value: ")
    print(
        str(unitary_distance(circuit.get_unitary(), target_unitary)) +
        " with gate: " + str(rx))

    print("step 4: ")
    print("lowest with dist:")
    lowest_at_step(circuit)
    circuit.append_gate(ry, 0)
    print("true value: ")
    print(
        str(unitary_distance(circuit.get_unitary(), target_unitary)) +
        " with gate: " + str(ry))

    print("step 5: ")
    print("lowest with dist:")
    lowest_at_step(circuit, 1)
    circuit.append_gate(rx, 1)
    print("true value: ")
    print(
        str(unitary_distance(circuit.get_unitary(), target_unitary)) +
        " with gate: " + str(rx))
    quick_dist(circuit)
