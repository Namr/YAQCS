# YAQCS
Yet Another Quantum Circuit Synthesizer

We have found that other quantum circuit synthesizers do not work well with Ion-Trap problems. This QCS aims to work on those solutions. There are two main files you should be aware of for YAQS.

* `brute_force.py`
  * this can be run with MPI, if you want to utilize MPI edit line 8 to make `USING_MPI = True`
* `depth_first_search.py`

both files will search for the unitary stored in `target_unitary` which is generally defined at the top of the file. You can change this and run the program and it will attempt to synthesize a circuit for it.

# BQSKit Extensions

The BQSKit Extensions folder has python files that add ion trap functionality to BQSKit. You will need to have BQSKit installed on your system to run these programs. The `examples` directory just has some circuit definitions that can be helpful to see, but don't directly relate to circuit synthesis. The other important files are:

* `bqskit2qiskit.py` this is a module that can be imporated into another file. It allows conversion from a BQSKit circuit to a Qiskit circuit.
* `rxxx.py` this module defines an MS gate across 3 qubits.
* `searchspace.py` this is the most important file in the project and is where the actual searching happens. 
  * The class `ThreeLayerGenerator` is the custom layer generator that allows MS gates to be used in QSearch. 
  * `configed_search` is the QSearchPass instantiated with that layer generator.
  * `chosen_unitary` on line 364 is the unitary that will actually be synthesized.
  * In the `__name__ == "__main__"` section of the code synthesis is run and the circuit is stored visually in a file. A QFast task is commented out but can be run if the layer generator is compatible.
