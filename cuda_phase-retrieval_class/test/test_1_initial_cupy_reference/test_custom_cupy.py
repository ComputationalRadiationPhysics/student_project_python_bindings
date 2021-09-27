import cupy as cp

import gpuMemManagement
import cupy_ref
import pytest

#test 1 : there is two way to get the cupy pointer, but are they give a same result?
def test_two_version_for_getting_cupy_pointer():
    cp.cuda.Device(0).use()
    a = cp.array([3.14, 4.25, 5.36], dtype=cp.float64)

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
    a = cp.array([3, 4, 5], dtype=cp.float64)
    b = cupy_ref.Cupy_Ref(ptr = a.data.ptr, shape = a.shape, dtype = str(a.dtype)) # the type of a cupy type is a numpy type class, 
                                                                                        # so I need to convert the type to string first
    c = gpuMemManagement.test_if_reinterpret_ptr_is_the_same(a.data.ptr, b)

    assert(c == True)

#test 4 : test if the result of the  reinterpret cast of a real cupy pointer is a device pointer
def test_if_real_cupy_reinterpret_ptr_is_a_gpu_array():
    cp.cuda.Device(0).use()
    a = cp.array([3.14, 4.25, 5.36], dtype=cp.float64)
    res = gpuMemManagement.test_if_real_cupy_reinterpret_ptr_is_a_gpu_array(a.data.ptr)

    assert(res == True)

#test 5 : test if the pointer of a custom cupy pointer attributes is a device pointer
def test_if_custom_cupy_reinterpret_ptr_is_a_gpu_array():
    cp.cuda.Device(0).use()
    a = cp.array([3.14, 4.25, 5.36], dtype=cp.float64)
    b = cupy_ref.Cupy_Ref(ptr = a.data.ptr, shape = a.shape, dtype = str(a.dtype))
    res = gpuMemManagement.test_if_custom_cupy_reinterpret_ptr_is_a_gpu_array(b)

    assert(res == True)

#test 6 : copy array of float from real cupy to cpu
def test_copy_real_cupy_to_cpu():
    cp.cuda.Device(0).use()
    a = cp.array([3.14, 4.25, 5.36], dtype=cp.float64)
    b = gpuMemManagement.test_copy_real_cupy_to_cpu(a.data.ptr, a.size)

    assert(b == True)

#test 7 : copy array of float from custom cupy to cpu
def test_copy_custom_cupy_to_cpu():
    cp.cuda.Device(0).use()
    a = cp.array([3.14, 4.25, 5.36], dtype=cp.float64)
    b = cupy_ref.Cupy_Ref(ptr = a.data.ptr, shape = a.shape, dtype = str(a.dtype))
    c = gpuMemManagement.test_copy_custom_cupy_to_cpu(b)

    assert(c == True)

#test 8 : test if real cupy can run successfully in cuda kernel
def test_real_cupy_pointer_with_cuda_kernel():
    a = cp.array([3.14, 4.25, 5.36], dtype=cp.float64)
    b = gpuMemManagement.real_cupy_increment_all_data_by_1(a.data.ptr, a.size)

    assert(b == True)

    # python side test
    # compare float with tolerance
    # each element of "a" should be incremented by 1 at this point, but because of cuda kernel, there is a very very small value change, 
    # so allclose is needed insteall of array_equal
    a_inc = cp.array([4.14, 5.25, 6.36])
    assert(cp.allclose(a, a_inc)) 

#test 9 : test if custom cupy can run successfully in cuda kernel
def test_custom_cupy_pointer_with_cuda_kernel():
    a = cp.array([3.14, 4.25, 5.36], dtype=cp.float64)
    b = cupy_ref.Cupy_Ref(ptr = a.data.ptr, shape = a.shape, dtype = str(a.dtype))
    c = gpuMemManagement.custom_cupy_increment_all_data_by_1(b)

    assert(c == True)

    # python side test
    # compare float with tolerance
    # By using "b", each element of "a" should be incremented by 1 at this point, same with test 8. But because of cuda kernel, there is a very very small value change, 
    # so allclose is needed insteall of array_equal
    a_inc = cp.array([4.14, 5.25, 6.36], dtype=cp.float64)
    assert(cp.allclose(a, a_inc)) 

#test 10 : create real cupy in c++, return it to python
def test_create_real_cupy_from_c():
    a = cp.array([3.14, 4.25, 5.36], dtype=cp.float64)
    b = gpuMemManagement.test_create_real_cupy_from_c()

    assert(cp.array_equal(a, b))
    assert(cp.asnumpy(b).sum() == b.sum()) #see if the normal cupy from c++ can be converted to numpy like a normal cupy
                                           #and see if sum of b is equal to sum of numpy version of b


#test 11 : create custom cupy in python, send it to c++ and return it again
def test_copy_custom_cupy_to_custom_cupy():

    a = cp.array([3.14, 4.25, 5.36], dtype=cp.float64)
    b = cupy_ref.Cupy_Ref(ptr = a.data.ptr, shape = a.shape, dtype = str(a.dtype))
    c = gpuMemManagement.test_copy_custom_cupy_to_custom_cupy(b)

    assert(b.ptr == c.ptr and b.dtype == c.dtype)

#test 12 : test memory usage, still not sure if this is a right way
def test_memory():
    assert(cp.get_default_memory_pool().used_bytes() == 0)
    a = gpuMemManagement.test_create_real_cupy_from_c()

    b = a*2
    assert(cp.array_equal(b.sum(), a.sum()*2))

    a = None
    b = None

    assert(cp.get_default_memory_pool().used_bytes() == 0)

#test 13 : send wrong float types
def test_wrong_dtype_float():
    with pytest.raises(TypeError):
        a = cp.array([3.14, 4.25, 5.36], dtype=cp.float32) #this cupy is set to float32, but the c++ function is float64
        b = cupy_ref.Cupy_Ref(ptr = a.data.ptr, shape = a.shape, dtype = str(a.dtype))
        gpuMemManagement.test_wrong_dtype_float(b) #this should raise an exception

#test 14 : send wrong integer types
def test_wrong_dtype_int():
    with pytest.raises(TypeError):
        a = cp.array([3, 4, 5], dtype=cp.uint32) #this cupy is set to uint32, but the c++ function is uint16
        b = cupy_ref.Cupy_Ref(ptr = a.data.ptr, shape = a.shape, dtype = str(a.dtype))
        gpuMemManagement.test_wrong_dtype_int(b) #this should raise an exception

#test 15 : send wrong complex types
def test_wrong_dtype_complex():
    with pytest.raises(TypeError):
        a = cp.array([2.+3.j,  0.+0.j, 4.+1.j], dtype=cp.complex64) #this cupy is set to float32, but the c++ function is float64
        b = cupy_ref.Cupy_Ref(ptr = a.data.ptr, shape = a.shape, dtype = str(a.dtype))
        gpuMemManagement.test_wrong_dtype_complex(b) #this should raise an exception

#test 16 : test template function with pybind 11
#there is a weird missmatch error happening 
def test_custom_cupy_template_function():
    cp.cuda.Device(0).use()
    a = cp.array([3, 4, 5], dtype=cp.complex128)
    b = cupy_ref.Cupy_Ref(ptr = a.data.ptr, shape = a.shape, dtype = str(a.dtype)) # the type of a cupy type is a numpy type class, 
                                                                                        # so I need to convert the type to string first
    c = gpuMemManagement.test_custom_cupy_template_function(a.data.ptr, b)

    assert(c == True)
   
    
