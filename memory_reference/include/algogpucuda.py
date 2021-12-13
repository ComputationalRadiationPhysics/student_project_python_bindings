from binding import AlgoCUDA
import cupy_ref
import cupy as cp

class AlgoGPUCUDA:
    def __init__(self):
        self.algocuda = AlgoCUDA()
    
    def whoami(self):
        self.algocuda.whoami()

    def is_synced_mem(self):
        return False
    
    def initialize_array(self, size):
        self.algocuda.initialize_array(size)
        self.input = cp.array(self.algocuda.get_input_memory(), copy = False)
        self.output = cp.array(self.algocuda.get_output_memory(), copy = False)

    def get_input_memory(self):
        return self.input
    
    def get_output_memory(self):
        return self.output

    def compute(self, input = None, output = None):

        if(input is None and output is None):
            self.algocuda.compute(self.algocuda.get_input_memory(), self.algocuda.get_output_memory())
        else:
            cupy_ref_input = cupy_ref.Cupy_Ref(ptr = input.data.ptr, shape = input.shape, dtype = input.dtype, typestr = input.dtype.str)
            cupy_ref_output = cupy_ref.Cupy_Ref(ptr = output.data.ptr, shape = output.shape, dtype = output.dtype, typestr = output.dtype.str)
            self.algocuda.compute(cupy_ref_input, cupy_ref_output)
