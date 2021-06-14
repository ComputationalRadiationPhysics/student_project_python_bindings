import cupy as cp
import numpy as np

import gpuMemManagement
import cupy_ref

#test 1 : there is two way to get the cupy pointer, but are they give a same result?
def test_two_version_for_getting_cupy_pointer():
    cp.cuda.Device(0).use()
    a = cp.array([3.14, 4.25, 5.36])

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
    a = cp.array([3.14, 4.25, 5.36])
    b = cupy_ref.Custom_Cupy_Ref(ptr = a.data.ptr, size = a.size)
    c = gpuMemManagement.test_if_reinterpret_ptr_is_the_same(a.data.ptr, b)

    assert(c == True)

#test 4 : test if the result of the  reinterpret cast of a real cupy pointer is a device pointer
def test_if_real_cupy_reinterpret_ptr_is_a_gpu_array():
    cp.cuda.Device(0).use()
    a = cp.array([3.14, 4.25, 5.36])
    res = gpuMemManagement.test_if_real_cupy_reinterpret_ptr_is_a_gpu_array(a.data.ptr)

    assert(res == True)

#test 5 : test if the pointer of a custom cupy pointer attributes is a device pointer
def test_if_custom_cupy_reinterpret_ptr_is_a_gpu_array():
    cp.cuda.Device(0).use()
    a = cp.array([3.14, 4.25, 5.36])
    b = cupy_ref.Custom_Cupy_Ref(ptr = a.data.ptr, size = a.size)
    res = gpuMemManagement.test_if_custom_cupy_reinterpret_ptr_is_a_gpu_array(b)

    assert(res == True)

#test 6 : copy array of float from real cupy to cpu
def test_copy_real_cupy_to_cpu():
    cp.cuda.Device(0).use()
    a = cp.array([3.14, 4.25, 5.36])
    b = gpuMemManagement.test_copy_real_cupy_to_cpu(a.data.ptr, a.size)

    assert(b == True)

#test 7 : copy array of float from custom cupy to cpu
def test_copy_custom_cupy_to_cpu():
    cp.cuda.Device(0).use()
    a = cp.array([3.14, 4.25, 5.36])
    b = cupy_ref.Custom_Cupy_Ref(ptr = a.data.ptr, size = a.size)
    c = gpuMemManagement.test_copy_custom_cupy_to_cpu(b)

    assert(c == True)

#test 8 : test if real cupy can run successfully in cuda kernel
def test_real_cupy_pointer_with_cuda_kernel():
    a = cp.array([3.14, 4.25, 5.36])
    b = gpuMemManagement.real_cupy_increment_all_data_by_1(a.data.ptr, a.size)

    assert(b == True)

#test 9 : test if custom cupy can run successfully in cuda kernel
def test_custom_cupy_pointer_with_cuda_kernel():
    a = cp.array([3.14, 4.25, 5.36])
    b = cupy_ref.Custom_Cupy_Ref(ptr = a.data.ptr, size = a.size)
    c = gpuMemManagement.custom_cupy_increment_all_data_by_1(b)

    assert(c == True)
