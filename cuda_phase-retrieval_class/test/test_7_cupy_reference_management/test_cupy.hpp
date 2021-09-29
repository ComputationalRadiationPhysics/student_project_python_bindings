#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cufft.h>
#include <cstdio>
#include <iostream>
#include <complex>

#include "cupy_ref.hpp"
#include "cupy_caster.hpp"
#include "cuda_algo.hpp"

using namespace std::literals::complex_literals;
using namespace pybind11::literals;

enum Mode { Hybrid = 1, InputOutput = 2, OutputOutput = 3};

//test 1. Generate 1D cupy of complex double from c++
pybind11::object test_generating_cupy_of_complex_double_from_c()
{
    std::vector<std::complex<double>> v{3.14, 4.25, 5.36};
    auto cp = pybind11::module::import("cupy").attr("array")(v, "dtype"_a="complex128");

    return cp;
}

//test 2. Generate 2D cupy of complex double from python, send it to c++, and get it back
pybind11::object test_send_cupy_complex_to_c_and_send_it_back(std::size_t a_address, std::size_t a_size, std::size_t a_x, std::size_t a_y) //can I use a single variable for shape?
{
    std::complex<double> *gpu_data = reinterpret_cast<std::complex<double> *>(a_address);

    std::vector<std::complex<double>> cpu_data(a_size);

    CUDA_CHECK(cudaMemcpy(cpu_data.data(), gpu_data, a_size*sizeof(std::complex<double>), cudaMemcpyDeviceToHost));

    auto cp = pybind11::module::import("cupy").attr("array")(cpu_data, "dtype"_a="complex128").attr("reshape")(a_x, a_y); //if I dont do this, the return will be 1D instead of 2D

    return cp;    
}

//test 3. Use CUFFT inverse + forward into 2D cupy of complex double, copy the result to a new cupy from c++, then send the result back
pybind11::object test_cupy_cufft_inverse_forward(std::size_t a_address, std::size_t a_size, std::size_t a_x, std::size_t a_y)
{
    //find number of SM
    int devId, numSMs;
    cudaGetDevice(&devId);
    cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, devId);

    int size_x = static_cast<int>(a_x); //Convert X to integer to prevent getting warning from CUFFT
    int size_y = static_cast<int>(a_y); //Convert Y to integer to prevent getting warning from CUFFT
    int dimension = static_cast<int>(a_size); //Convert size to integer to prevent getting warning from CUFFT

    //convert source cupy from double complex to cufftDoubleComplex so CUFFT can use it
    cufftDoubleComplex *gpu_data = reinterpret_cast<cufftDoubleComplex *>(a_address);

    //create a new 2D cupy double complex with the same size and shape as source cupy
    auto cp = pybind11::module::import("cupy").attr("zeros")(dimension, "dtype"_a="complex128").attr("reshape")(size_x, size_y);

    //convert new cupy from double complex to cufftDoubleComplex so CUFFT can use it
    cufftDoubleComplex *gpu_data_result = reinterpret_cast<cufftDoubleComplex *>(cp.attr("data").attr("ptr").cast<std::size_t>());

    //coopy data from source cupy to new cupy
    CUDA_CHECK(cudaMemcpy(gpu_data_result, gpu_data, dimension*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice));

    // ---------------CUFFT process---------------------------------
    cufftHandle plan; //create cufft plan
        
    CUFFT_CHECK(cufftPlan2d(&plan, size_x, size_y, CUFFT_Z2Z));

    //CUFFT Inverse
    CUFFT_CHECK(cufftExecZ2Z(plan, gpu_data_result, gpu_data_result, CUFFT_INVERSE));

    //normalize cufft after inverse. This function comes from test_algo.hpp
    normalize_array<<<8*numSMs, 256>>>(gpu_data_result, gpu_data_result, dimension);

    //CUFFT forward
    CUFFT_CHECK(cufftExecZ2Z(plan, gpu_data_result, gpu_data_result, CUFFT_FORWARD));

    cufftDestroy(plan);

    return cp;    
}

//test 4. Test create a custom cupy from a non cupy object in C++
template <typename T>
void test_create_a_custom_cupy_from_a_non_cupy_object_in_c()
{
    auto cp = pybind11::module::import("numpy").attr("ones")(4, "dtype"_a="complex128").attr("reshape")(2, 2);
    Cupy_Ref<T> casted_cp = Cupy_Ref<T>::getCupyRef(cp);
}

//test 5. create a custom cupy with flexible dimension (TDim is using the default value "0")
template <typename T>
std::size_t test_create_a_custom_cupy_with_flexible_dimension(Cupy_Ref<T> b)
{
    return b.shape.size();
}

//test 6. test for succesfully create a custom cupy with fixed dimension (TDim is equal to the dimensiom of the source cupy)
template <typename T>
std::size_t test_create_a_custom_cupy_with_fixed_dimension_success(Cupy_Ref<T, 3> b)
{
    return b.shape.size();
}

//test 7. test for failing to create a custom cupy with fixed dimension (TDim is not equal to the dimensiom of the source cupy)
template <typename T>
std::size_t test_create_a_custom_cupy_with_fixed_dimension_fail(Cupy_Ref<T, 4> b)
{
    return b.shape.size();
}


//test 8. same with test 3, but with cupy caster
template <typename T>
pybind11::object test_cupy_cufft_inverse_forward_with_caster(Cupy_Ref<T> b)
{
    //find number of SM
    int devId, numSMs;
    cudaGetDevice(&devId);
    cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, devId);

    int size_x = static_cast<int>(b.shape[0]); //Convert X to integer to prevent getting warning from CUFFT
    int size_y = static_cast<int>(b.shape[1]); //Convert Y to integer to prevent getting warning from CUFFT
    int dimension = static_cast<int>(size_x*size_y); //Convert size to integer to prevent getting warning from CUFFT

    //convert source cupy from double complex to cufftDoubleComplex so CUFFT can use it
    // cufftDoubleComplex *gpu_data = reinterpret_cast<cufftDoubleComplex *>(b.ptr);
    cufftDoubleComplex *gpu_data = convertToCUFFT<T, cufftDoubleComplex>(b.ptr);

    //create a new 2D cupy double complex with the same size and shape as source cupy
    auto cp = pybind11::module::import("cupy").attr("zeros")(dimension, "dtype"_a="complex128").attr("reshape")(size_x, size_y);
    Cupy_Ref<T> casted_cp = Cupy_Ref<T>::getCupyRef(cp);

    //convert new cupy from double complex to cufftDoubleComplex so CUFFT can use it
    cufftDoubleComplex *gpu_data_result = convertToCUFFT<T, cufftDoubleComplex>(casted_cp.ptr);

    //coopy data from source cupy to new cupy
    CUDA_CHECK(cudaMemcpy(gpu_data_result, gpu_data, dimension*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice));

    // ---------------CUFFT process---------------------------------
    cufftHandle plan; //create cufft plan
        
    CUFFT_CHECK(cufftPlan2d(&plan, size_x, size_y, CUFFT_Z2Z));

    //CUFFT Inverse
    CUFFT_CHECK(cufftExecZ2Z(plan, gpu_data_result, gpu_data_result, CUFFT_INVERSE));

    //normalize cufft after inverse. This function comes from test_algo.hpp
    normalize_array<<<8*numSMs, 256>>>(gpu_data_result, gpu_data_result, dimension);

    //CUFFT forward
    CUFFT_CHECK(cufftExecZ2Z(plan, gpu_data_result, gpu_data_result, CUFFT_FORWARD));

    cufftDestroy(plan);

    return cp;    
}

//test 9. send cupy caster to c++ and send it back to python
//although the result caster (c) doesnt have its own cupy, this test may be useful
template <typename T>
Cupy_Ref<T> test_send_cupy_caster_to_c_and_get_it_back(Cupy_Ref<T> b)
{
    Cupy_Ref<T> c = b;
    return c;
}

//test 10. check if c++ is properly removing the cupy object that is created in c++ after an end of a function
void test_cupy_from_c_memory()
{
    std::vector<std::complex<double>> v{3.14, 4.25, 5.36};
    auto cp = pybind11::module::import("cupy").attr("array")(v, "dtype"_a="complex128");

    //make sure cupy "cp" is really using a memory
    assert(pybind11::module::import("cupy").attr("get_default_memory_pool")().attr("used_bytes")().cast<std::size_t>() > 0 );

    //cp removed automatically after the next closing bracket
}

//test 11. Test Enum with pybind
int test_enum(Mode phase_mode)
{
    return phase_mode;
}

//test 12. Test create a 1D cupy object with a custom allocate function
pybind11::object test_custom_cupy_object_creator_1d()
{
    pybind11::object cp = cupy_allocate<std::uint64_t>({42});
    return cp;
}

//test 13. Test create a 2D cupy object with a custom allocate function
pybind11::object test_custom_cupy_object_creator_2d()
{
    pybind11::object cp = cupy_allocate<std::complex<double>>({4,4});
    return cp;
}

//test 14. Test create a 3D cupy object with a custom allocate function
pybind11::object test_custom_cupy_object_creator_3d()
{
    pybind11::object cp = cupy_allocate<long long int>({3,4,5});
    return cp;
}