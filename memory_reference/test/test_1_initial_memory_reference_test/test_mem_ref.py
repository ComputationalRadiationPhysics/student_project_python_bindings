import Test_Mem_Ref
import numpy as np

if Test_Mem_Ref.is_cuda_available() == True:
    from algogpu import AlgoGPU
    import cupy_ref
    import cupy as cp

def test_memory_reference_for_cpu():
    algo = Test_Mem_Ref.AlgoCPU()

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
    assert(isinstance(input, np.ndarray))
    assert(isinstance(output, np.ndarray))

def test_memory_reference_for_gpu():
    if Test_Mem_Ref.is_cuda_available() == True:
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
        assert(isinstance(input, cp._core.core.ndarray))
        assert(isinstance(output, cp._core.core.ndarray))
    
    else:
        print("CUDA is not available")

    
def test_get_algo():
    print(Test_Mem_Ref.get_available_device())