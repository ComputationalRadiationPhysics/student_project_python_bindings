#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cufft.h>
#include <cuComplex.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <complex>
#include <string>
// #include "cupy_ref.hpp"

#define PI 3.1415926535897932384626433
#define CUDA_CHECK(call) {cudaError_t error = call; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}
using namespace std;
using namespace std::literals::complex_literals;
using namespace pybind11::literals;
namespace py = pybind11;

template<typename TInputData, typename TOutputData> TOutputData * convertToCUFFT(TInputData * ptr);
template<> cufftDoubleComplex *convertToCUFFT(complex<double> * ptr);

//test 1. Generate 1D cupy of complex double from c++
py::object test_generating_cupy_of_complex_double_from_c()
{
    vector<complex<double>> v{3.14, 4.25, 5.36};
    auto cp = py::module::import("cupy").attr("array")(v, "dtype"_a="complex128");

    return cp;
}

//test 2. Generate 2D cupy of complex double from python, send it to c++, and get it back
py::object test_send_cupy_complex_to_c_and_send_it_back(size_t a_address, size_t a_size, size_t a_x, size_t a_y) //can I use a single variable for shape?
{
    complex<double> *gpu_data = reinterpret_cast<complex<double> *>(a_address);

    vector<complex<double>> cpu_data(a_size);

    CUDA_CHECK(cudaMemcpy(cpu_data.data(), gpu_data, a_size*sizeof(complex<double>), cudaMemcpyDeviceToHost));

    auto cp = py::module::import("cupy").attr("array")(cpu_data, "dtype"_a="complex128").attr("reshape")(a_x, a_y); //if I dont do this, the return will be 1D instead of 2D

    return cp;    
}

//test 3. Use CUFFT inverse + forward into 2D cupy of complex double, copy the result to a new cupy from c++, then send the result back
py::object test_cupy_cufft_inverse_forward(size_t a_address, size_t a_size, size_t a_x, size_t a_y)
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
    auto cp = py::module::import("cupy").attr("zeros")(dimension, "dtype"_a="complex128").attr("reshape")(size_x, size_y);

    //convert new cupy from double complex to cufftDoubleComplex so CUFFT can use it
    cufftDoubleComplex *gpu_data_result = reinterpret_cast<cufftDoubleComplex *>(cp.attr("data").attr("ptr").cast<size_t>());

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
    auto cp = py::module::import("numpy").attr("ones")(4, "dtype"_a="complex128").attr("reshape")(2, 2);
    Custom_Cupy_Ref<T> casted_cp = Custom_Cupy_Ref<T>::getCustomCupyRef(cp);
}

//test 5. create a custom cupy with flexible dimension
template <typename T>
size_t test_create_a_custom_cupy_with_flexible_dimension(Custom_Cupy_Ref<T> b)
{
    return b.shape.size();
}


//test 6. same with test 3, but with cupy caster
template <typename T>
py::object test_cupy_cufft_inverse_forward_with_caster(Custom_Cupy_Ref<T> b)
{
    //find number of SM
    int devId, numSMs;
    cudaGetDevice(&devId);
    cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, devId);

    int size_x = static_cast<int>(b.shape[0]); //Convert X to integer to prevent getting warning from CUFFT
    int size_y = static_cast<int>(b.shape[1]); //Convert Y to integer to prevent getting warning from CUFFT
    int dimension = static_cast<int>(b.size); //Convert size to integer to prevent getting warning from CUFFT

    //convert source cupy from double complex to cufftDoubleComplex so CUFFT can use it
    // cufftDoubleComplex *gpu_data = reinterpret_cast<cufftDoubleComplex *>(b.ptr);
    cufftDoubleComplex *gpu_data = convertToCUFFT<T, cufftDoubleComplex>(b.ptr);

    //create a new 2D cupy double complex with the same size and shape as source cupy
    auto cp = py::module::import("cupy").attr("zeros")(dimension, "dtype"_a="complex128").attr("reshape")(size_x, size_y);
    Custom_Cupy_Ref<T> casted_cp = Custom_Cupy_Ref<T>::getCustomCupyRef(cp);

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

//test 7. send cupy caster to c++ and send it back to python
//although the result caster (c) doesnt have its own cupy, this test may be useful
template <typename T>
Custom_Cupy_Ref<T> test_send_cupy_caster_to_c_and_get_it_back(Custom_Cupy_Ref<T> b)
{
    Custom_Cupy_Ref<T> c = b;
    return c;
}

//test 8. check if c++ is properly removing the cupy object that is created in c++ after an end of a function
void test_cupy_from_c_memory()
{
    vector<complex<double>> v{3.14, 4.25, 5.36};
    auto cp = py::module::import("cupy").attr("array")(v, "dtype"_a="complex128");

    //make sure cupy "cp" is really using a memory
    assert(py::module::import("cupy").attr("get_default_memory_pool")().attr("used_bytes")().cast<size_t>() > 0 );

    //cp removed automatically after the next closing bracket
}

template<typename TInputData, typename TOutputData>
TOutputData * convertToCUFFT(TInputData * ptr){}

template<>
cufftDoubleComplex *convertToCUFFT(complex<double> * ptr)
{  
    return reinterpret_cast<cufftDoubleComplex *>(ptr);
}