import binding
import cupy as cp
import numpy as np


# algo can be easily replace by 
algo = binding.AlgoCPU()
# algo = binding.AlgoCUDA

algo.whoami()
algo.initialize_array(10)

input = algo.get_input_memory()
output = algo.get_output_memory()

print(input)
print(type(input))

for i in range(10):
    input[i] = 2.0

algo.compute(input, output)

print(output)
print(type(output))