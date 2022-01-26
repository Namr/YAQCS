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

    def __init__(self, qbits):
        self.num_qbits = qbits

    def append_gate(self, gate: Gate, position):
        if gate.num_qbits > self.num_qbits - position:
            print("cant add a multiqbit gate at position " + position)
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

        unitary = self.get_displaced_gate_unitary(self.operations[0])
        if (len(self.operations) > 1):
            for op in range(1, len(self.operations)):
                unitary *= self.get_displaced_gate_unitary(self.operations[op])
        return unitary


class RXGate(Gate):
    def __init__(self) -> None:
        self.num_qbits = 1
        self.num_params = 1
        self.num_variations = 1
        self.variations = [math.pi / 2.0, -math.pi / 2.0]
        return None

    def get_unitary(self, variation_index: int = 0) -> np.ndarray:
        ct = np.cos(self.variations[variation_index] / 2.0)
        st = np.sin(self.variations[variation_index] / 2.0)

        return np.matrix([[ct, -1j * st], [-1j * st, ct]])


class RXXGate(Gate):
    def __init__(self) -> None:
        self.num_qbits = 2
        self.num_params = 1
        self.num_variations = 1
        self.variations = [math.pi / 2.0, -math.pi / 2.0]
        return None

    def get_unitary(self, variation_index: int = 0) -> np.ndarray:
        ct = np.cos(self.variations[variation_index] / 2.0)
        st = np.sin(self.variations[variation_index] / 2.0)

        return np.matrix([[ct, 0, 0, -1j * st], [0, ct, -1j * st, 0],
                          [0, -1j * st, ct, 0], [-1j * st, 0, 0, ct]])


if __name__ == "__main__":
    rx = RXGate()
    rxx = RXXGate()

    circuit = Circuit(4)
    circuit.append_gate(rx, 1)
    circuit.append_gate(rxx, 0)
    circuit.append_gate(rx, 2)
    print(np.around(circuit.get_unitary(), 2))
