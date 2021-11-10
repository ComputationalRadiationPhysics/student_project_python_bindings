import binding
import cupy_ref
import cupy as cp

class AlgoGPU:
    def __init__(self):
        self.algocuda = binding.AlgoCUDA()
    
    def whoami(self):
        self.algocuda.whoami()
    
    def initialize_array(self, size):
        self.algocuda.initialize_array(size)

    def get_input_memory(self):
        return cp.array(self.algocuda.get_input_memory(), copy = False)
    
    def get_output_memory(self):
        return cp.array(self.algocuda.get_output_memory(), copy = False)

    def compute(self, input, output):

        cupy_ref_input = cupy_ref.Cupy_Ref(ptr = input.data.ptr, shape = input.shape, dtype = input.dtype, typestr = input.dtype.str)
        cupy_ref_output = cupy_ref.Cupy_Ref(ptr = output.data.ptr, shape = output.shape, dtype = output.dtype, typestr = output.dtype.str)
        self.algocuda.compute(cupy_ref_input, cupy_ref_output)