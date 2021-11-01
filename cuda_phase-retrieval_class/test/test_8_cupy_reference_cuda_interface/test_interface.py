import cupy as cp
import numpy as np
import cupy_ref
import pytest
import Test_Interface

def test_return_as_cupy_copy():
    a = cp.array([1.1, 3.14, 42.69])
    b = cupy_ref.Cupy_Ref(ptr = a.data.ptr, shape = a.shape, dtype = a.dtype)

    c = cp.array(b, dtype=b.dtype, copy=True) # we use copy to create a new cupy object with the same value of "a"
    
    assert(cp.array_equal(a,c))
    assert(a.data.ptr != c.data.ptr) # make sure it is copied

def test_return_as_cupy_not_copy():
    a = cp.array([1.1, 3.14, 42.69])
    b = cupy_ref.Cupy_Ref(ptr = a.data.ptr, shape = a.shape, dtype = a.dtype)

    c = cp.array(b, dtype=b.dtype, copy=False) # we dont use copy to get the original "a"
     
    assert(cp.array_equal(a,c))
    assert(a.data.ptr == c.data.ptr) # make sure it is not copied

def test_memory_holder():
    a = Test_Interface.GPU_memory_holder([3])
    b = a.get_memory_reference()
    c = cp.array(b, dtype=b.dtype, copy=False)

    a_2 = cp.array([0.0, 0.0, 0.0])

    assert(c.shape == a_2.shape)
    assert(cp.array_equal(c, a_2))

def test_add_cupy_and_numpy():
    a = Test_Interface.GPU_memory_holder([3])
    b = a.get_memory_reference()
    c = cp.array(b, dtype=b.dtype, copy=False)

    data = np.array([1.0, 1.0, 1.0])

    c_inc = cp.array([1.0, 1.0, 1.0])

    for i in range(3):
        c[i] = data[i]

    assert cp.array_equal(c, c_inc)


def test_memory_sychronization():
    """
    Check if automatic copy to host and copy to device are correct done,
    if c++ cuda kernel is called via python binding
    """
    size = 30

    mem_holder = Test_Interface.GPU_memory_holder([size])
    mem_holder_ref = mem_holder.get_memory_reference()
    mem_holder_array = cp.array(mem_holder_ref, dtype=mem_holder_ref.dtype, copy=False)

    data = np.arange(1, size + 1, 1, np.float64)
    expected_result = cp.array(data + 1)

    for i in range(size):
        mem_holder_array[i] = data[i]

    Test_Interface.incOne(mem_holder_ref, size)

    assert cp.array_equal(expected_result, mem_holder_array)
