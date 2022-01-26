from qiskit import QuantumCircuit, assemble, Aer, QuantumRegister, ClassicalRegister, transpile
from math import pi, sqrt
from qiskit.visualization import plot_bloch_multivector, plot_histogram, array_to_latex
import itertools
import copy
import time

allowed_gates = [("rxx", pi / 2), ("rx", pi / 2), ("rx", -pi / 2),
                 ("ry", pi / 2), ("ry", -pi / 2), ("rz", pi / 2),
                 ("rz", -pi / 2)]

sim = Aer.get_backend('aer_simulator_unitary')
target_unitary = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
count = 0
sTime = time.time()


def test_circuit(qubits: int, circuit: list) -> bool:
    c = QuantumCircuit(qubits)
    for op in circuit:
        if op[0][0] == "rxx":
            c.rxx(op[0][1], op[1], op[1] + 1)
        if op[0][0] == "rx":
            c.rx(op[0][1], op[1])
        if op[0][0] == "ry":
            c.ry(op[0][1], op[1])
        if op[0][0] == "rz":
            c.rz(op[0][1], op[1])

    # see if the unitary is the same
    c.save_unitary()
    transpiled_qc = transpile(c, sim)
    result = sim.run(transpiled_qc).result().get_unitary(transpiled_qc)

    if (result == target_unitary).all():
        print(circuit)
        print("we got em")
        quit()
        return True
    return False


def pure_brute_force(qubits: int, gates: list, max_depth: int,
                     current_depth: int, circuit: list) -> bool:

    global count
    global time

    if current_depth > max_depth:
        return False

    for q in range(qubits):
        for gate in allowed_gates:
            if gate[0] == "rxx" and q == qubits - 1:
                continue

            # save the state of our circuit before our modification
            save_old = copy.deepcopy(circuit)

            # add the new gate and see if it makes it work
            circuit.append((gate, q))
            test_circuit(qubits, circuit)

            count += 1
            if count % 1000 == 0:
                print(count)
                print(time.time() - sTime)

            # if not then recursively add a new gate
            if pure_brute_force(qubits, gates, max_depth, current_depth + 1,
                                circuit) == False:
                circuit = save_old


if __name__ == "__main__":
    if pure_brute_force(2, allowed_gates, 3, 0, []) == False:
        print("no solutions?")
