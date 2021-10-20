class Cupy_Ref:
    def __init__(self, ptr, dtype, shape):
        self.ptr = ptr
        self.dtype = dtype
        self.shape = tuple(shape) #the shape properties of numpy is a tuple. This part become a list if it received a vector from c++.
                                  #this is causing an assertion error in test because tuple != list. so tuple conversion is needed.
        
    # For now, I still dont know how to get the original cupy object (with the original address (ptr) from __init__).
    # What I do in this function is I copy the value of original cupy to a new cupy object using cudaMemcpy operation in c++,
    # and then return the new cupy object
    def as_cupy(self):
        # Because get() is using pybind, I need to choose a pybind project to access it (it use test 8_project for now). 
        # How do I make it a flexible pybind function so it can be used in multiple project?
        try:
            import Test_Interface
            data = Test_Interface.get(self.ptr, self.shape)
            return data
        except ModuleNotFoundError:
            print("This function is currently only for test_8")


    def __repr__(self):
        return str(self.as_cupy())