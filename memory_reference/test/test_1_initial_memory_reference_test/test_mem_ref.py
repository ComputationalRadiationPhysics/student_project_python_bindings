from algogpu import AlgoGPU
from Test_Mem_Ref import AlgoCPU
import cupy_ref
import cupy as cp
import numpy as np

def test_memory_reference_for_cpu():
    algo = AlgoCPU()

    algo.whoami()
    algo.initialize_array(10)

    input = algo.get_input_memory()
    output = algo.get_output_memory()

    print(input)
    print(type(input))

    for i in range(10):
        input[i] = i/5

    algo.compute(input, output)

    print(output)
    print(type(output))

    assert(np.array_equal(input,output))
    assert(str(type(input)) == "<class 'numpy.ndarray'>")
    assert(str(type(output)) == "<class 'numpy.ndarray'>")

def test_memory_reference_for_gpu():
    algo = AlgoGPU()

    algo.whoami()
    algo.initialize_array(10)

    input = algo.get_input_memory()
    output = algo.get_output_memory()

    print(input)
    print(type(input))

    for i in range(10):
        input[i] = i/5

    algo.compute(input, output)

    print(output)
    print(type(output))

    assert(cp.array_equal(input,output))
    assert(str(type(input)) == "<class 'cupy._core.core.ndarray'>")
    assert(str(type(output)) == "<class 'cupy._core.core.ndarray'>")