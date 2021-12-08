import hip_ref
import binding
import numpy as np

class Hip_Mem:
    def __init__(self, size):
        self.implementation = binding.Hip_Mem_Impl()
        self.device_mem = self.implementation.get_hip_array(size)
        self.buffer = np.zeros(size)
        
    def read(self):
        self.implementation.read(self.buffer, self.device_mem)

    def write(self):
        self.implementation.write(self.buffer, self.device_mem)