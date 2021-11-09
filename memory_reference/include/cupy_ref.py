import cupy as cp

class Cupy_Ref:
    def __init__(self, ptr, dtype, shape, typestr):
        self.ptr = ptr
        self.dtype = str(dtype)
        self.typestr  = str(typestr)
        self.shape = tuple(shape) #the shape properties of numpy is a tuple. This part become a list if it received a vector from c++.
                                  #this is causing an assertion error in test because tuple != list. so tuple conversion is needed.

        # cuda array interface implementation. without this, we cannot create a new cupy array from cupy_ref
        self.__cuda_array_interface__ = {}
        self.__cuda_array_interface__['shape'] = tuple(shape)
        self.__cuda_array_interface__['typestr'] = str(typestr)
        self.__cuda_array_interface__['data'] = (ptr, False)
        self.__cuda_array_interface__['version'] = 3

    def get_cupy_array(self):
        return cp.array(self, dtype=self.dtype, copy=False)