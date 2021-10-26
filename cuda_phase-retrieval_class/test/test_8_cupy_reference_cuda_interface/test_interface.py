import cupy as cp
import numpy as np
import cupy_ref
import pytest

def test_return_as_cupy_copy():
    a = cp.array([4,5,6])
    b = cupy_ref.Cupy_Ref(ptr = a.data.ptr, shape = a.shape, dtype = str(a.dtype), typestr = a.dtype.str, 
                          data_pointer = a.__cuda_array_interface__['data'][0], readonly = a.__cuda_array_interface__['data'][1])

    c = cp.array(b, dtype=b.dtype, copy=True) # we use copy to create a new cupy object with the same value of "a"
   
    assert(cp.array_equal(a,c))
    assert(a.data.ptr != c.data.ptr) # make sure it is copied

def test_return_as_cupy_not_copy():
    a = cp.array([4,5,6])
    b = cupy_ref.Cupy_Ref(ptr = a.data.ptr, shape = a.shape, dtype = str(a.dtype), typestr = a.dtype.str, 
                          data_pointer = a.__cuda_array_interface__['data'][0], readonly = a.__cuda_array_interface__['data'][1])

    c = cp.array(b, dtype=b.dtype, copy=False) # we dont use copy to get the original "a"

    assert(cp.array_equal(a,c))
    assert(a.data.ptr == c.data.ptr) # make sure it is not copied

