import numpy as np
import math


class Gate:
    num_qbits = 0
    num_params = 0
    # how many configurations of this gate are supported by hardware
    num_variations = 0
    variations = []

    def __init__(self) -> None:
        return

    def get_unitary(self, variation_index: int = 0) -> np.ndarray:
        return None


class Circuit:
    num_qbits = 0
    # an operation is a tuple of (gate, position_placed), this is also known as a "placed_gate"
    operations = []
    is_qiskit_ordering = False

    def __init__(self, qbits, qiskit_style=False):
        self.num_qbits = qbits
        self.is_qiskit_ordering = qiskit_style

    def append_gate(self, gate: Gate, position):
        if self.is_qiskit_ordering:
            position = abs(position - (self.num_qbits - 1))

        if gate.num_qbits > self.num_qbits - position:
            print("cant add a multiqbit gate at position " + str(position))
            return None

        self.operations.append((gate, position))

    def get_displaced_gate_unitary(self, placed_gate: list):
        pos = placed_gate[1]
        unitary = placed_gate[0].get_unitary()
        if pos != 0:
            unitary = np.kron(np.identity(pow(2, pos)), unitary)

        if self.num_qbits - pos - placed_gate[0].num_qbits != 0:
            unitary = np.kron(
                unitary,
                np.identity(
                    pow(2, self.num_qbits - pos - placed_gate[0].num_qbits)))

        return unitary

    def get_unitary(self) -> np.ndarray:
        if self.operations == 0:
            print("Can't get Unitary of a circuit with no operations")
            return None

        true_operations = self.operations[::-1]

        unitary = self.get_displaced_gate_unitary(true_operations[0])
        if (len(true_operations) > 1):
            for op in range(1, len(true_operations)):
                unitary = unitary * \
                    self.get_displaced_gate_unitary(true_operations[op])
        return unitary


class RXGate(Gate):
    def __init__(self, params=[math.pi / 2.0]) -> None:
        self.num_qbits = 1
        self.num_params = 1
        if (len(params) != self.num_params):
            print("invalid number of params in RXGate")

        self.params = params

        return None

    def get_unitary(self) -> np.ndarray:
        ct = np.cos(self.params[0] / 2.0)
        st = np.sin(self.params[0] / 2.0)

        return np.matrix([[ct, -1j * st], [-1j * st, ct]])


class RYGate(Gate):
    def __init__(self, params=[math.pi / 2.0]) -> None:
        self.num_qbits = 1
        self.num_params = 1
        if (len(params) != self.num_params):
            print("invalid number of params in RXGate")

        self.params = params

        return None

    def get_unitary(self) -> np.ndarray:
        ct = np.cos(self.params[0] / 2.0)
        st = np.sin(self.params[0] / 2.0)

        return np.matrix([[ct, -1 * st], [st, ct]])


class RZGate(Gate):
    def __init__(self, params=[math.pi / 2.0]) -> None:
        self.num_qbits = 1
        self.num_params = 1
        if (len(params) != self.num_params):
            print("invalid number of params in RXGate")

        self.params = params

        return None

    def get_unitary(self) -> np.ndarray:
        return np.matrix([[np.exp(-1j * self.params[0] / 2.0), 0],
                          [0, np.exp(1j * self.params[0] / 2.0)]])


class RXXGate(Gate):
    def __init__(self, params=[math.pi / 2.0]) -> None:
        self.num_qbits = 2
        self.num_params = 1
        if (len(params) != self.num_params):
            print("invalid number of params in RXGate")

        self.params = params
        return None

    def get_unitary(self) -> np.ndarray:
        ct = np.cos(self.params[0] / 2.0)
        st = np.sin(self.params[0] / 2.0)

        return np.matrix([[ct, 0, 0, -1j * st], [0, ct, -1j * st, 0],
                          [0, -1j * st, ct, 0], [-1j * st, 0, 0, ct]])


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

    print(np.around(circuit.get_unitary(), 2))
