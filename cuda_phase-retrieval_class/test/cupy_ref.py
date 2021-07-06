class Custom_Cupy_Ref:
    def __init__(self, ptr, size, dtype, shape_x, shape_y):
        self.ptr = ptr
        self.size = size
        self.dtype = dtype
        self.shape_x = shape_x
        self.shape_y = shape_y