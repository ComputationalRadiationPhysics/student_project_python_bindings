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
using namespace std;
using namespace std::literals::complex_literals;
namespace py = pybind11;


__global__ void gpu_increment_all_data_by_1(double *gpu_data)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    gpu_data[i] = gpu_data[i] + 1; 
}

//test if a pointer is a device/gpu pointer or not
int is_device_pointer(const void *ptr)
{
  int is_device_ptr = 0;
  cudaPointerAttributes attributes;

  CUDA_CHECK(cudaPointerGetAttributes(&attributes, ptr));

  if(attributes.type == 2) //according to documentation, if cudaMemoryTypeDevice = 2, then it is a device memory
                           //https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaPointerAttributes.html#structcudaPointerAttributes
                           //https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g13de56a8fe75569530ecc3a3106e9b6d
  {
    is_device_ptr = 1;
  }

  return is_device_ptr;
}

//test 3 : test if reinterpret cast with real cupy and custom cupy will result a same value
uint32_t test_if_reinterpret_ptr_is_the_same(size_t a_address, Custom_Cupy_Ref b)
{
    double * ptr = reinterpret_cast<double *>(a_address);
    if(b.ptr == ptr) return 1;
    else return 0;
}

//test 4 : test if the result of the  reinterpret cast of a real cupy pointer is a device pointer
int test_if_real_cupy_reinterpret_ptr_is_a_gpu_array(size_t a_address)
{
    double * ptr = reinterpret_cast<double *>(a_address);
    int res = is_device_pointer(ptr);
    
    return res;
}

//test 5 : test if the pointer of a custom cupy pointer attributes is a device pointer
int test_if_custom_cupy_reinterpret_ptr_is_a_gpu_array(Custom_Cupy_Ref b)
{
    int res = is_device_pointer(b.ptr);
    
    return res;
}

//test 6 : copy array of float from real cupy to cpu
uint32_t test_copy_real_cupy_to_cpu(size_t a_address, size_t a_size)
{
    double *gpu_data = reinterpret_cast<double *>(a_address);

    vector<double> cpu_data(a_size);

    CUDA_CHECK(cudaMemcpy(cpu_data.data(), gpu_data, a_size*sizeof(double), cudaMemcpyDeviceToHost));

    uint32_t test = 1;

    cout<<endl;
    for(uint32_t i = 0; i < a_size; i++)
    {
        if(cpu_data[i] != 3.14) 
        {
            test = 0;
            cout<<cpu_data[i]<<endl;
        }
    }

    return test;
}

//test 7 : copy array of float from custom cupy to cpu
uint32_t test_copy_custom_cupy_to_cpu(Custom_Cupy_Ref b)
{
    vector<double> cpu_data(b.size);

    CUDA_CHECK(cudaMemcpy(cpu_data.data(), b.ptr, b.size*sizeof(double), cudaMemcpyDeviceToHost));

    uint32_t test = 1;

    cout<<endl;
    for(uint32_t i = 0; i < b.size; i++)
    {
        if(cpu_data[i] != 3.14) 
        {
            test = 0;
            cout<<cpu_data[i]<<endl;
        }
    }

    return test;
}

//test 8 : increment all real cupy data by 1, and return the sum of all data
double real_cupy_increment_all_data_by_1(size_t a_address, size_t a_size) 
{
    vector<double> cpu_data(a_size);

    double *gpu_data = reinterpret_cast<double*>(a_address);
    
    gpu_increment_all_data_by_1<<<a_size, 1>>>(gpu_data);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(cpu_data.data(), gpu_data, a_size*sizeof(double), cudaMemcpyDeviceToHost));

    double sum = 0;

    for(uint32_t i = 0; i <  a_size; i++)
    {
        sum += cpu_data[i];
    }

    return sum;
}

//test 8 : increment all custom cupy data by 1, and return the sum of all data
double custom_cupy_increment_all_data_by_1(Custom_Cupy_Ref b) 
{
    vector<double> cpu_data(b.size);
    
    gpu_increment_all_data_by_1<<<b.size, 1>>>(b.ptr);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(cpu_data.data(), b.ptr, b.size*sizeof(double), cudaMemcpyDeviceToHost));

    double sum = 0;

    for(uint32_t i = 0; i <  b.size; i++)
    {
        sum += cpu_data[i];
    }

    return sum;
}


//note: create test with integer