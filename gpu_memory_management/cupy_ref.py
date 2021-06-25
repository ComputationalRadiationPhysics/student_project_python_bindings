class Custom_Cupy_Ref:
    def __init__(self, ptr, size, dtype):
        self.ptr = ptr
        self.size = size
        self.dtype = dtype