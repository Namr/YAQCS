import numpy as np
from bqskit.compiler import Compiler
from bqskit.compiler import CompilationTask
from bqskit.passes import QuickPartitioner
from bqskit.passes import ForEachBlockPass
from bqskit.passes import QSearchSynthesisPass
from bqskit.passes import ScanningGateRemovalPass
from bqskit.passes import UnfoldPass
from bqskit.ir.gates import CXGate
from bqskit import UnitaryMatrix
from bqskit import Circuit

def less_2q_gates(result_circuit, initial_block_as_op):
    begin_cx_count = initial_block_as_op.gate._circuit.count(CXGate())
    end_cx_count = result_circuit.count(CXGate())
    return end_cx_count < begin_cx_count

# Construct the unitary as an NumPy array
unitary = UnitaryMatrix(
    [
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0],
])
#
#[[-0.707+0.707j,  0.0+0.j,    -0.0+0.j,    -0.0+0.j,    -0.0+0.j,
#   0.0-0.j,     0.0-0.j,    -0.0-0.j],
# [ 0.0-0.j,    -0.707+0.707j,  0.0-0.j,    -0.0-0.j,    -0.0-0.j,
#  -0.0+0.j,    -0.0+0.j,    -0.0+0.j],
# [ 0.0-0.j,    -0.0+0.j,    -0.707+0.707j, -0.0+0.j,     0.0-0.j,
#   0.0-0.j,    -0.0-0.j,    -0.0-0.j],
# [ 0.0-0.j,     0.0-0.j,     0.0-0.j,    -0.707+0.707j, -0.0+0.j,
#  -0.0+0.j,    -0.0+0.j,    -0.0+0.j],
# [ 0.0-0.j,     0.0+0.j,     0.0-0.j,     0.0-0.j,    -0.707+0.707j,
#  -0.0-0.j,    -0.0+0.j,    -0.0+0.j],
# [ 0.0-0.j,     0.0-0.j,     0.0-0.j,    -0.0+0.j,     0.0-0.j,
#  -0.707+0.707j, -0.0+0.j,    -0.0-0.j],
# [-0.0-0.j,     0.0+0.j,     0.0-0.j,     0.0-0.j,     0.0-0.j,
#   0.0-0.j,    -0.707+0.707j,  0.0-0.j],
# [-0.0-0.j,     0.0-0.j,     0.0-0.j,    -0.0-0.j,     0.0-0.j,
#   0.0-0.j,    -0.0-0.j,    -0.707+0.707j]]
#)

circuit = Circuit.from_unitary(unitary)

# Create a standard synthesis CompilationTask
task = CompilationTask(circuit, [QSearchSynthesisPass()])
#task = CompilationTask(circuit, [
#    QuickPartitioner(3),
#    ForEachBlockPass(
#        [QSearchSynthesisPass(), ScanningGateRemovalPass()],
#        replace_filter=less_2q_gates
#    ),
#    UnfoldPass(),
#])

# Spawn a compiler and compile the task
if __name__ == '__main__':
    with Compiler() as compiler:
        synthesized_circuit = compiler.compile(task)
    print(synthesized_circuit)
    for gate in synthesized_circuit.gate_set:
        print(f"{gate} Count:", synthesized_circuit.count(gate))
    dist = synthesized_circuit.get_unitary().get_distance_from(unitary)
    print("distance", dist, "delta", 1e-10, "small", dist < 1e-10)
    print("Critial Path Length:", synthesized_circuit.depth)
    print("Parallelism:", synthesized_circuit.parallelism)
    print("Total Number of Operations:", synthesized_circuit.num_operations)
    
