import pytest
import cupy as cp
import numpy as np

import gpuMemManagement
import cupy_ref

#test 1 : there is two way to get the cupy pointer, but are they give a same result?
def test_two_version_for_getting_cupy_pointer():
    cp.cuda.Device(0).use()
    a = cp.array([3.14, 3.14, 3.14])

    assert(a.data.ptr == a.__cuda_array_interface__["data"][0])

#test 2 : test if the size attribute get the right size
def test_getting_cupy_size():
    cp.cuda.Device(0).use()
    elem = 5   
    a  = cp.full(elem, 3.14)

    assert(elem == a.size)

#test 3 : test if reinterpret cast with real cupy and custom cupy will result a same value
def test_if_reinterpret_ptr_is_the_same():
    cp.cuda.Device(0).use()
    a = cp.array([3.14, 3.14, 3.14])
    b = cupy_ref.Custom_Cupy_Ref(ptr = a.data.ptr, size = a.size)
    c = gpuMemManagement.test_if_reinterpret_ptr_is_the_same(a.data.ptr, b)

    assert(c == 1)

#test 4 : test if the result of the  reinterpret cast of a real cupy pointer is a device pointer
def test_if_real_cupy_reinterpret_ptr_is_a_gpu_array():
    cp.cuda.Device(0).use()
    a = cp.array([3.14, 3.14, 3.14])
    res = gpuMemManagement.test_if_real_cupy_reinterpret_ptr_is_a_gpu_array(a.data.ptr)

    assert(res == 1)

#test 5 : test if the pointer of a custom cupy pointer attributes is a device pointer
def test_if_custom_cupy_reinterpret_ptr_is_a_gpu_array():
    cp.cuda.Device(0).use()
    a = cp.array([3.14, 3.14, 3.14])
    b = cupy_ref.Custom_Cupy_Ref(ptr = a.data.ptr, size = a.size)
    res = gpuMemManagement.test_if_custom_cupy_reinterpret_ptr_is_a_gpu_array(b)

    assert(res == 1)

#test 6 : copy array of float from real cupy to cpu
def test_copy_real_cupy_to_cpu():
    cp.cuda.Device(0).use()
    a = cp.array([3.14, 3.14, 3.14])
    b = gpuMemManagement.test_copy_real_cupy_to_cpu(a.data.ptr, a.size)

    assert(b == 1)

#test 7 : copy array of float from custom cupy to cpu
def test_copy_custom_cupy_to_cpu():
    cp.cuda.Device(0).use()
    a = cp.array([3.14, 3.14, 3.14])
    b = cupy_ref.Custom_Cupy_Ref(ptr = a.data.ptr, size = a.size)
    c = gpuMemManagement.test_copy_custom_cupy_to_cpu(b)

    assert(c == 1)

#test 8 : test if real cupy can run successfully in cuda kernel
def test_real_cupy_pointer_with_cuda_kernel():
    a = cp.array([4.5, 4.5, 4.5])
    b = gpuMemManagement.real_cupy_increment_all_data_by_1(a.data.ptr, a.size)

    assert(b == 3*(4.5 + 1))

#test 9 : test if custom cupy can run successfully in cuda kernel
def test_custom_cupy_pointer_with_cuda_kernel():
    a = cp.array([4.5, 4.5, 4.5])
    b = cupy_ref.Custom_Cupy_Ref(ptr = a.data.ptr, size = a.size)
    c = gpuMemManagement.custom_cupy_increment_all_data_by_1(b)

    assert(c == 3*(4.5 + 1))

#one of the reason why test 1 - 9 was successfull is because all cupy array value in the test is always double precision python float,
#the reinterpret cast of the real cupy is always casted to double, and the custom cupy current design is only accepting double type.
#test 10 : what if instead of python float, array of integer is sent by real cupy. expectation : failure
def test_reinterpret_integer_cupy_to_double_with_real_cupy():
    a = cp.array([4, 4, 4, 4])
    b = gpuMemManagement.test_copy_real_cupy_to_cpu(a.data.ptr, a.size)

    assert(b == 0) #fail

#test 11 : what if instead of python float, array of integer is sent by real cupy to custom cupy. expectation : failure
def test_reinterpret_integer_cupy_to_double_with_custom_cupy():
    a = cp.array([4, 4, 4, 4])
    b = cupy_ref.Custom_Cupy_Ref(ptr = a.data.ptr, size = a.size)
    c = gpuMemManagement.test_copy_custom_cupy_to_cpu(b)

    assert(c == 0) #fail
