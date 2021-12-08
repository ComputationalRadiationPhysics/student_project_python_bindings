from binding import AlgoHIP
import hip_mem

class AlgoGPUHIP:
    def __init__(self):
        self.algohip = AlgoHIP()
    
    def whoami(self):
        self.algohip.whoami()

    def is_synced_mem(self):
        return True
    
    def initialize_array(self, size):
        self.size = size
        self.input = hip_mem.Hip_Mem(size)
        self.output = hip_mem.Hip_Mem(size)

    def get_input_memory(self):
        return self.input
    
    def get_output_memory(self):
        return self.output

    def compute(self, input, output):

        self.algohip.compute(input.device_mem, output.device_mem, self.size)