import cuPhaseRet_Test
import cupy as cp
import cupy_ref

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
    print(a)
    print(b)

    assert(cp.allclose(a, b)) #array_uqual wont work because there is still a very very small difference

#test 4. same with test 3, but with cupy caster
def test_cupy_cufft_inverse_forward_with_caster():
    a = cp.array([[3.14, 4.25, 5.36], [4, 5, 6], [1.23, 4.56, 7.89]], dtype=cp.complex128)
    b = cupy_ref.Custom_Cupy_Ref(ptr = a.data.ptr, size = a.size, dtype = str(a.dtype), shape = a.shape)
    c = cuPhaseRet_Test.test_cupy_cufft_inverse_forward_with_caster(b)

    print()
    print(a)
    print(c)

    assert(cp.allclose(a, c)) #array_uqual wont work because there is still a very very small difference

#test 5. send cupy caster to c++ and send it back to python
#although the result caster (c) doesnt have its own cupy, this test may be useful
def test_send_cupy_caster_to_c_and_get_it_back():
    a = cp.array([[3.14, 4.25, 5.36], [4, 5, 6], [1.23, 4.56, 7.89]], dtype=cp.complex128)
    b = cupy_ref.Custom_Cupy_Ref(ptr = a.data.ptr, size = a.size, dtype = str(a.dtype), shape = a.shape)
    c = cuPhaseRet_Test.test_send_cupy_caster_to_c_and_get_it_back(b)

    assert(a.data.ptr == c.ptr and a.size == c.size and a.dtype == c.dtype and a.shape == c.shape)

#test 6. check if c++ is properly removing the cupy object that is created in c++ after an end of a function
def test_cupy_from_c_memory():
    assert(cp.get_default_memory_pool().used_bytes() == 0)
    
    cuPhaseRet_Test.test_cupy_from_c_memory()

    assert(cp.get_default_memory_pool().used_bytes() == 0)