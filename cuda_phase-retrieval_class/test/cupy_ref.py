class Custom_Cupy_Ref:
    def __init__(self, ptr, size, dtype, shape):
        self.ptr = ptr
        self.size = size
        self.dtype = dtype
        self.shape = tuple(shape) #the shape properties of numpy is a tuple. This part become a list if it received a vector from c++.
                                  #this is causing an assertion error in test because tuple != list. so tuple conversion is needed.