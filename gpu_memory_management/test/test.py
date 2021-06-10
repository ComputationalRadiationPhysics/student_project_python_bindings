import pytest
import cupy as cp
import numpy as np

import gpuMemManagement
import cupy_ref

#test 1 : there is two way to get the cupy pointer, but are they give a same result?
def test_two_version_for_getting_cupy_pointer():
    a = cp.ones(5)

    assert(a.data.ptr == a.__cuda_array_interface__["data"][0])

#test 2 : test if the size attribute get the right size
def test_getting_cupy_size():
    elem = 5   
    a  = cp.ones(elem)

    assert(elem == a.size)

#test 3 : test if reinterpret cast with real cupy and custom cupy will result a same value
def test_if_reinterpret_ptr_is_the_same():
    a = cp.ones(5)
    b = cupy_ref.Custom_Cupy_Ref(ptr = a.data.ptr, size = a.size)
    c = gpuMemManagement.test_if_reinterpret_ptr_is_the_same(a.data.ptr, b)

    assert(c == 1)

#test 4 : test if the result of the  reinterpret cast of a real cupy pointer is a device pointer
def test_if_real_cupy_reinterpret_ptr_is_a_gpu_array():
    a = cp.ones(5)
    res = gpuMemManagement.test_if_real_cupy_reinterpret_ptr_is_a_gpu_array(a.data.ptr)

    assert(res == 1)

#test 5 : test if the pointer of a custom cupy pointer attributes is a device pointer
def test_if_custom_cupy_reinterpret_ptr_is_a_gpu_array():
    a = cp.ones(5)
    b = cupy_ref.Custom_Cupy_Ref(ptr = a.data.ptr, size = a.size)
    res = gpuMemManagement.test_if_custom_cupy_reinterpret_ptr_is_a_gpu_array(b)

    assert(res == 1)

#test 6 : copy array of ones from real cupy to cpu
def test_copy_cupy_of_ones_to_cpu():
    a = cp.ones(5)
    b = gpuMemManagement.test_copy_cupy_of_ones_to_cpu(a.data.ptr, a.size)

    assert(b == 1)

#test 7 : copy array of ones from custom cupy to cpu
def test_copy_custom_cupy_of_ones_to_cpu():
    a = cp.ones(5)
    b = cupy_ref.Custom_Cupy_Ref(ptr = a.data.ptr, size = a.size)
    c = gpuMemManagement.test_copy_custom_cupy_of_ones_to_cpu(b)

    assert(c == 1)

#test 8 : copy array of float from custom cupy to cpu
def test_copy_custom_cupy_of_float_to_cpu():
    a = cp.array([3.14, 3.14, 3.14])
    b = cupy_ref.Custom_Cupy_Ref(ptr = a.data.ptr, size = a.size)
    c = gpuMemManagement.test_copy_custom_cupy_of_float_to_cpu(b)

    assert(c == 1)

#test 9 : test 2 reinterpret cast with different data type and see if both has a same value
#this part need more explanation
def test_2_different_reiterpret_cast():
    a = cp.array([3.14, 3.14, 3.14])
    b = gpuMemManagement.test_2_different_reiterpret_cast(a.data.ptr)

    assert(b == 1)

