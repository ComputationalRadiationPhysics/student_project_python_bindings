#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <cstdio>
#include <time.h>
#include <iostream>
#include <complex>
#include <string>
#include <random>


#define CUDA_CHECK(call) {cudaError_t error = call; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}
using namespace std::literals::complex_literals;
using namespace pybind11::literals;

__global__ void gpu_increment_all_data_by_1(double *gpu_data)
{
    int i = blockIdx.x;
    gpu_data[i] = gpu_data[i] + 1.0; 
}


bool AreVeryClose(double a, double b)
{
    //source : https://stackoverflow.com/questions/4548004/how-to-correctly-and-standardly-compare-floats
    return (fabs(a - b) <= std::numeric_limits<double>::epsilon() * fmax(fabs(a), fabs(b)));
}

//test if a pointer is a device/gpu pointer or not
bool is_device_pointer(const void *ptr)
{
  cudaPointerAttributes attributes;

  CUDA_CHECK(cudaPointerGetAttributes(&attributes, ptr));

  return attributes.type == 2; 
  //according to documentation, if cudaMemoryTypeDevice = 2, then it is a device memory
  //https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaPointerAttributes.html#structcudaPointerAttributes
  //https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g13de56a8fe75569530ecc3a3106e9b6d

}

//test 3 : test if reinterpret cast with real cupy and custom cupy will result a same value
bool test_if_reinterpret_ptr_is_the_same(size_t a_address, Custom_Cupy_Ref<double> b)
{
    double * ptr = reinterpret_cast<double *>(a_address);
    if(b.ptr == ptr) return true;
    else return false;
}

//test 4 : test if the result of the  reinterpret cast of a real cupy pointer is a device pointer
bool test_if_real_cupy_reinterpret_ptr_is_a_gpu_array(size_t a_address)
{
    double * ptr = reinterpret_cast<double *>(a_address);
    bool res = is_device_pointer(ptr);
    
    return res;
}

//test 5 : test if the pointer of a custom cupy pointer attributes is a device pointer
bool test_if_custom_cupy_reinterpret_ptr_is_a_gpu_array(Custom_Cupy_Ref<double> b)
{
    bool res = is_device_pointer(b.ptr);
    
    return res;
}

//test 6 : copy array of float from real cupy to cpu
bool test_copy_real_cupy_to_cpu(size_t a_address, size_t a_size)
{
    double *gpu_data = reinterpret_cast<double *>(a_address);

    std::vector<double> cpu_data(a_size);

    CUDA_CHECK(cudaMemcpy(cpu_data.data(), gpu_data, a_size*sizeof(double), cudaMemcpyDeviceToHost));

    return (AreVeryClose(cpu_data[0], 3.14) && AreVeryClose(cpu_data[1], 4.25) && AreVeryClose(cpu_data[2], 5.36));
}

//test 7 : copy array of float from custom cupy to cpu
bool test_copy_custom_cupy_to_cpu(Custom_Cupy_Ref<double> b)
{
    int linear_size = 1;
    for(int const &s : b.shape) linear_size *= s;

    std::vector<double> cpu_data(linear_size);

    CUDA_CHECK(cudaMemcpy(cpu_data.data(), b.ptr, linear_size*sizeof(double), cudaMemcpyDeviceToHost));

    return (AreVeryClose(cpu_data[0], 3.14) && AreVeryClose(cpu_data[1], 4.25) && AreVeryClose(cpu_data[2], 5.36));
}

//test 8 : increment all real cupy data by 1, and check if each element is true (or very close)
bool real_cupy_increment_all_data_by_1(size_t a_address, size_t a_size) 
{
    std::vector<double> cpu_data(a_size);

    double *gpu_data = reinterpret_cast<double*>(a_address);
    
    gpu_increment_all_data_by_1<<<a_size, 1>>>(gpu_data);

    CUDA_CHECK(cudaMemcpy(cpu_data.data(), gpu_data, a_size*sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaDeviceSynchronize());
    
    return (AreVeryClose(cpu_data[0], 4.14) && AreVeryClose(cpu_data[1], 5.25) && AreVeryClose(cpu_data[2], 6.36));
}

//test 9 : increment all custom cupy data by 1, and check if each element is true (or very close)
bool custom_cupy_increment_all_data_by_1(Custom_Cupy_Ref<double> b) 
{
    int linear_size = 1;
    for(int const &s : b.shape) linear_size *= s;

    std::vector<double> cpu_data(linear_size);
    
    gpu_increment_all_data_by_1<<<linear_size, 1>>>(b.ptr);

    CUDA_CHECK(cudaMemcpy(cpu_data.data(), b.ptr, linear_size*sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaDeviceSynchronize());
    
    return (AreVeryClose(cpu_data[0], 4.14) && AreVeryClose(cpu_data[1], 5.25) && AreVeryClose(cpu_data[2], 6.36));
}

//test 10 : create real cupy in c++, return it to python
pybind11::object test_create_real_cupy_from_c()
{
    std::vector<double> v{3.14, 4.25, 5.36};
    auto cp = pybind11::module::import("cupy").attr("array")(v, "dtype"_a="float64");

    return cp;
}

//test 11 : test 10 : create custom cupy in python, send it to c++ and return it again
Custom_Cupy_Ref<double> test_copy_custom_cupy_to_custom_cupy(Custom_Cupy_Ref<double> b)
{
    Custom_Cupy_Ref<double> c;

    c.ptr = b.ptr;
    c.dtype = b.dtype;

    return c;
}

//test 13 : send wrong float types
void test_wrong_dtype_float(Custom_Cupy_Ref<double> b){}

//test 14 : send wrong integer types
void test_wrong_dtype_int(Custom_Cupy_Ref<uint16_t> b){}

//test 15 : send wrong complex types
void test_wrong_dtype_complex(Custom_Cupy_Ref<std::complex<double>> b){}

//test 16 : template function with pybind11
template <typename T>
bool test_custom_cupy_template_function(size_t a_address, Custom_Cupy_Ref<T> b)
{
    T * ptr = reinterpret_cast<T *>(a_address);
    return (is_device_pointer(ptr) && is_device_pointer(b.ptr) && ptr == b.ptr);
}