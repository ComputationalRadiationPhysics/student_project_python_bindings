class Hip_Ref:
    def __init__(self, ptr, dtype, shape):
        self.ptr = ptr
        self.dtype = str(dtype)
        self.shape = tuple(shape)