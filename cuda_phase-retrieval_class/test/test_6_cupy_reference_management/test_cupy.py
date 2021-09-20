import cuPhaseRet_Test
import cupy as cp
import cupy_ref
import pytest

#test 1. Generate 1D cupy of complex double from c++
def test_generating_cupy_of_complex_double_from_c():
    a = cp.array([3.14, 4.25, 5.36], dtype=cp.complex128)
    b = cuPhaseRet_Test.test_generating_cupy_of_complex_double_from_c()

    assert(cp.array_equal(a, b))

#test 2. Generate 2D cupy of complex double from python, send it to c++, and get it back
def test_send_cupy_complex_to_c_and_send_it_back():
    a = cp.array([[3.14, 4.25, 5.36], [4, 5, 6], [1.23, 4.56, 7.89]], dtype=cp.complex128)
    b = cuPhaseRet_Test.test_send_cupy_complex_to_c_and_send_it_back(a.data.ptr, a.size, a.shape[0], a.shape[1]) #is there any way to receive shape a single variable?

    assert(cp.array_equal(a, b))

#test 3. Use CUFFT inverse + forward into 2D cupy of complex double, copy the result to a new cupy from c++, then send the result back
def test_cupy_cufft_inverse_forward():
    a = cp.array([[3.14, 4.25, 5.36], [4, 5, 6], [1.23, 4.56, 7.89]], dtype=cp.complex128)
    b = cuPhaseRet_Test.test_cupy_cufft_inverse_forward(a.data.ptr, a.size, a.shape[0], a.shape[1])

    print()
    print("Test 3")
    print(a)
    print(b)

    assert(cp.allclose(a, b)) #array_uqual wont work because there is still a very very small difference

#test 4. Test create a custom cupy from a non cupy object in C++
def test_create_a_custom_cupy_from_a_non_cupy_object_in_c():
    with pytest.raises(Exception) as excinfo:
        cuPhaseRet_Test.test_create_a_custom_cupy_from_a_non_cupy_object_in_c()
    
    print()
    print("Test 4")
    print(excinfo.value)


#test 5. create a custom cupy with flexible dimension (TDim is using the default value "0" in C++)
def test_create_a_custom_cupy_with_flexible_dimension():
    a = cp.ones((2,2,2), dtype=cp.complex128)
    b = cupy_ref.Custom_Cupy_Ref(ptr = a.data.ptr, dtype = str(a.dtype), shape = a.shape)
    c= cuPhaseRet_Test.test_create_a_custom_cupy_with_flexible_dimension(b)

    assert(c == 3)

    a = cp.ones((3,3,3,3), dtype=cp.complex128)
    b = cupy_ref.Custom_Cupy_Ref(ptr = a.data.ptr, dtype = str(a.dtype), shape = a.shape)
    c= cuPhaseRet_Test.test_create_a_custom_cupy_with_flexible_dimension(b)

    assert(c == 4)

#test 6. test for succesfully create a custom cupy with fixed dimension (TDim is equal to the dimensiom of this functiom cupy "a")
def test_create_a_custom_cupy_with_fixed_dimension_success():
    a = cp.ones((2,2,2), dtype=cp.complex128)
    b = cupy_ref.Custom_Cupy_Ref(ptr = a.data.ptr, dtype = str(a.dtype), shape = a.shape)
    c= cuPhaseRet_Test.test_create_a_custom_cupy_with_fixed_dimension_success(b)

    assert(c == 3)

#test 7. test for failing to create a custom cupy with fixed dimension (TDim is not equal to the dimensiom of this functiom cupy "a")
def test_create_a_custom_cupy_with_fixed_dimension_fail():
    with pytest.raises(Exception):
        a = cp.ones((2,2,2), dtype=cp.complex128)
        b = cupy_ref.Custom_Cupy_Ref(ptr = a.data.ptr, dtype = str(a.dtype), shape = a.shape)
        c = cuPhaseRet_Test.test_create_a_custom_cupy_with_fixed_dimension_fail(b)

        assert(c == 3)


#test 8. same with test 3, but with cupy caster
def test_cupy_cufft_inverse_forward_with_caster():
    a = cp.array([[3.14, 4.25, 5.36], [4, 5, 6], [1.23, 4.56, 7.89]], dtype=cp.complex128)
    b = cupy_ref.Custom_Cupy_Ref(ptr = a.data.ptr, dtype = str(a.dtype), shape = a.shape)
    c = cuPhaseRet_Test.test_cupy_cufft_inverse_forward_with_caster(b)

    print()
    print("Test 6")
    print(a)
    print(c)

    assert(cp.allclose(a, c)) #array_uqual wont work because there is still a very very small difference

#test 9. send cupy caster to c++ and send it back to python
#although the result caster (c) doesnt have its own cupy, this test may be useful
def test_send_cupy_caster_to_c_and_get_it_back():
    a = cp.array([[3.14, 4.25, 5.36], [4, 5, 6], [1.23, 4.56, 7.89]], dtype=cp.complex128)
    b = cupy_ref.Custom_Cupy_Ref(ptr = a.data.ptr, dtype = str(a.dtype), shape = a.shape)
    c = cuPhaseRet_Test.test_send_cupy_caster_to_c_and_get_it_back(b)

    assert(a.data.ptr == c.ptr and a.dtype == c.dtype and a.shape == c.shape)

#test 10. check if c++ is properly removing the cupy object that is created in c++ after an end of a function
def test_cupy_from_c_memory():
    assert(cp.get_default_memory_pool().used_bytes() == 0)
    
    cuPhaseRet_Test.test_cupy_from_c_memory()
    
    assert(cp.get_default_memory_pool().used_bytes() == 0)

#test 11. Test Enum with pybind 
def test_enum():
    assert(cuPhaseRet_Test.test_enum(cuPhaseRet_Test.Hybrid) == 1)
    assert(cuPhaseRet_Test.test_enum(cuPhaseRet_Test.InputOutput) == 2)
    assert(cuPhaseRet_Test.test_enum(cuPhaseRet_Test.OutputOutput) == 3)

#test 12. Test create a 1D cupy object with a custom allocate function
def test_custom_cupy_object_creator_1d():
   b = cuPhaseRet_Test.test_custom_cupy_object_creator_1d()
   print()
   print("Test 12, 1D")
   assert(b.size == 42)
   print(b.size)
   assert(b.dtype == "uint64")
   print(b.dtype)
   assert(b.shape == (42,))
   print(b.shape)
   assert(b.ndim == 1)
   print(b.ndim)

#test 13. Test create a 2D cupy object with a custom allocate function
def test_custom_cupy_object_creator_2d():
   b = cuPhaseRet_Test.test_custom_cupy_object_creator_2d()
   print()
   print("Test 13, 2D")
   assert(b.size == 16)
   print(b.size)
   assert(b.dtype == "complex128")
   print(b.dtype)
   assert(b.shape == (4,4))
   print(b.shape)
   assert(b.ndim == 2)
   print(b.ndim)

#test 14. Test create a 3D cupy object with a custom allocate function
def test_custom_cupy_object_creator_3d():
   b = cuPhaseRet_Test.test_custom_cupy_object_creator_3d()
   print()
   print("Test 14, 3D")
   assert(b.size == 60)
   print(b.size)
   assert(b.dtype == "int64")
   print(b.dtype)
   assert(b.shape == (3,4,5))
   print(b.shape)
   assert(b.ndim == 3)
   print(b.ndim)

#test 15. Test modify generated cupy
def test_modify_generated_cupy():
   b = cuPhaseRet_Test.test_custom_cupy_object_creator_1d()
   print()
   print("Test 15, 1D")
   print(b)
   print("First Element + 10")
   b[0] = b[0] + 10
   print(b)
   print("Last Element + 50")
   b[b.size-1] = b[b.size-1] + 50
   print(b)
   print("Create new cupy from python")
   c = cp.ones(42)
   print(c)
   print("Add test 15 cupy with new cupy")
   b = b + c
   print(b)
   print("Get Sum")
   print(cp.sum(b))
   print("Sort")
   b = cp.sort(b)
   print(b)
   print("Copy to CPU")
   b_cpu = cp.asnumpy(b)
   print(b_cpu)