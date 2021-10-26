class Cupy_Ref:
    def __init__(self, ptr, dtype, shape, typestr = None, data_pointer = None, readonly = None):
        self.ptr = ptr
        self.dtype = dtype
        self.shape = tuple(shape) #the shape properties of numpy is a tuple. This part become a list if it received a vector from c++.
                                  #this is causing an assertion error in test because tuple != list. so tuple conversion is needed.

        # cuda array interface implementation. without this, we cannot create a new cupy array from cupy_ref
        if(typestr != None):
            self.__cuda_array_interface__ = {}
            self.__cuda_array_interface__['shape'] = shape
            self.__cuda_array_interface__['typestr'] = typestr
            self.__cuda_array_interface__['data'] = (data_pointer, readonly)
            self.__cuda_array_interface__['version'] = 3