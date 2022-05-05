from qiskit import QuantumCircuit, assemble, Aer, QuantumRegister, ClassicalRegister, transpile
from math import pi, sqrt
from qiskit.visualization import plot_bloch_multivector, plot_histogram, array_to_latex
import numpy as np
sim = Aer.get_backend('aer_simulator_unitary')


def myX(qc, theta, target):
    qc.r(theta, 0, target)


def myY(qc, theta, target):
    qc.r(theta, np.pi / 2, target)


def myZ(qc, theta, target):
    qc.r(np.pi, 0, target)
    qc.r(np.pi, theta / 2, target)


def myH(qc, target):
    qc.r(-np.pi, 0, target)
    qc.r(np.pi/2, np.pi/2, target)


def myT(qc, target):
    myZ(qc, np.pi / 4, target)


def myTdg(qc, target):
    myZ(qc, -np.pi / 4, target)


def mycx(qc, c, t):
    myY(qc, np.pi/2, c)
    qc.rxx(np.pi/2, c, t)
    myY(qc, -np.pi/2, c)
    myX(qc, -np.pi/2, t)
    myZ(qc, -np.pi/2, c)


# 3 quantum qubits
qc = QuantumCircuit(3)

qc.ccx(2, 1, 0)

# qc.x(0)
# qc.x(1)
myH(qc, 0)
mycx(qc, 1, 0)
myTdg(qc, 0)
mycx(qc, 2, 0)
myT(qc, 0)
mycx(qc, 1, 0)
myTdg(qc, 0)
mycx(qc, 2, 0)
myT(qc, 1)
myT(qc, 0)
mycx(qc, 2, 1)
myT(qc, 2)
myTdg(qc, 1)
mycx(qc, 2, 1)
myH(qc, 0)

qc.save_unitary()
qc.draw(output='mpl')

transpiled_qc = transpile(qc, sim)
result = sim.run(transpiled_qc).result()
unitary = result.get_unitary(transpiled_qc)
print("Circuit unitary:\n", np.asarray(unitary).round(5))
