import numpy as np
from bqskit.compiler import Compiler
from bqskit.compiler import CompilationTask

# Construct the unitary as an NumPy array
toffoli = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0],
])

# Create a standard synthesis CompilationTask
task = CompilationTask.synthesize(toffoli)

# Spawn a compiler and compile the task
if __name__ == '__main__':
    comp = Compiler()
    synthesized_circuit = comp.compile(task)
