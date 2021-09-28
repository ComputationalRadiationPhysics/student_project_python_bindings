#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cufft.h>
#include <cuComplex.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>
#include <cstdio>
#include <time.h>
#include <iostream>
#include <complex>
#include <string>
#include <random>

#define PI 3.1415926535897932384626433
#define CUDA_CHECK(call) {cudaError_t error = call; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}

__global__ void get_complex_array(double *real_array, cufftDoubleComplex *complex_array, int dimension);
__global__ void get_absolute_array(cufftDoubleComplex *complex_array, double *real_array , int dimension);
template<typename TInputData, typename TOutputData> TOutputData * convertToCUFFT(TInputData * ptr);
template<> cufftDoubleComplex *convertToCUFFT(std::complex<double> * ptr);
template<typename T> pybind11::object cupy_allocate(std::vector<int> shape);
void CUFFT_CHECK(cufftResult cufft_process);

/**
* \brief Convert array of real number into array of complex number
* \param real_array Array of double real number
* \param complex_array Array of double complex number, implemented using CUFFT library
* \param dimension Size of all arrays
*/
__global__ void get_complex_array(double *real_array, cufftDoubleComplex *complex_array, int dimension)
{
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dimension; idx += blockDim.x * gridDim.x)  
    {
        cufftDoubleComplex complex_number;
        complex_number.x = real_array[idx];
        complex_number.y = 0;
        complex_array[idx] = complex_number;
    }
}

/**
* \brief Get array of absolute value from array of complex number
* \param complex_array Array of double complex number, implemented using CUFFT library
* \param real_array Array of absolute value of the complex number
* \param dimension Size of all arrays
*/
__global__ void get_absolute_array(cufftDoubleComplex *complex_array, double *real_array , int dimension)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dimension; idx += blockDim.x * gridDim.x)  
    {
        real_array[idx] = cuCabs(complex_array[idx]);
    }
}

/**
* \brief CUFFT error checking
* \param cufft_process Result of a CUFFT operation
*/ 
void CUFFT_CHECK(cufftResult cufft_process)
{
    if(cufft_process != CUFFT_SUCCESS) std::cout<<cufft_process<<std::endl;
}

/**
* \brief Reinterpret a complex pointer from standard complex to CUDA FFT
* \param ptr a standard complex number
* \return CUDA FFT version of the standard complex number
*/ 
template<typename TInputData, typename TOutputData>
TOutputData * convertToCUFFT(TInputData * ptr){}

template<>
cufftDoubleComplex *convertToCUFFT(std::complex<double> * ptr)
{  
    return reinterpret_cast<cufftDoubleComplex *>(ptr);
}

/**
* \brief Generate a cupy array with 0 as the values
* \param shape vector representing dimension and size
* \return python object, which is a cupy array
*/
template<typename T> 
pybind11::object cupy_allocate(std::vector<int> shape)
{
    int linear_size = 1;
    for(int const &s : shape) linear_size *= s;
    pybind11::object cp = pybind11::module::import("cupy").attr("zeros")(linear_size, "dtype"_a=cupy_ref_get_dtype<T>()).attr("reshape")(shape);
    return cp;
}