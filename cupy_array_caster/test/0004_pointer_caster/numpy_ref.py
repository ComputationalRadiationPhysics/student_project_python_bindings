class Simple_Numpy_Ref:
    def __init__(self, ptr):
        self.ptr = ptr


class Advanced_Numpy_Ref:
    def __init__(self, ptr, size):
        self.ptr = ptr
        self.size = size
