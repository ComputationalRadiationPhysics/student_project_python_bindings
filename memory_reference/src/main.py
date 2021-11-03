import binding
import cupy as cp
import numpy as np


# algo can be easily replace by 
algo = binding.AlgoCPU()
# algo = binding.AlgoCUDA

algo.initialize_array(10)
input = algo.get_input_memory().get_data()
for i in range(10):
    input[i] = 2.0

algo.compute(algo.get_input_memory(), algo.get_output_memory())

print(algo.get_output_memory().get_data())