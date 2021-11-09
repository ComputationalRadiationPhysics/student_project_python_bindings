import binding
import cupy as cp
import numpy as np


# algo can be easily replace by 
# algo = binding.AlgoCPU()
algo = binding.AlgoCUDA()

algo.whoami()
algo.initialize_array(10)

input = algo.get_input_memory()
output = algo.get_output_memory()

cupy_input = cp.array(input, copy = False)
cupy_output = cp.array(output, copy = False)

print(cupy_input)
print(type(cupy_input))

for i in range(10):
    cupy_input[i] = 2.0

algo.compute(input, output)

print(cupy_output)
print(type(cupy_output))