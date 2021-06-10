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
uint32_t test_if_reinterpret_ptr_is_the_same(size_t a_adress, Custom_Cupy_Ref b)
{
    double * ptr = reinterpret_cast<double *>(a_adress);
    if(b.ptr == ptr) return 1;
    else return 0;
}

//test 4 : test if the result of the  reinterpret cast of a real cupy pointer is a device pointer
int test_if_real_cupy_reinterpret_ptr_is_a_gpu_array(size_t a_adress)
{
    double * ptr = reinterpret_cast<double *>(a_adress);
    int res = is_device_pointer(ptr);
    
    return res;
}

//test 5 : test if the pointer of a custom cupy pointer attributes is a device pointer
int test_if_custom_cupy_reinterpret_ptr_is_a_gpu_array(Custom_Cupy_Ref b)
{
    int res = is_device_pointer(b.ptr);
    
    return res;
}

//test 6 : copy array of ones from real cupy to cpu
uint32_t test_copy_cupy_of_ones_to_cpu(size_t a_adress, size_t a_size)
{
    double *ptr = reinterpret_cast<double *>(a_adress); //altough cp.ones is used, the result of cp.ones is a float, not integer

    double *cpu_data = (double*)malloc(a_size*sizeof(double));

    CUDA_CHECK(cudaMemcpy(cpu_data, ptr, a_size*sizeof(double), cudaMemcpyDeviceToHost));

    uint32_t test = 1;

    for(uint32_t i = 0; i < a_size; i++)
    {
        if(cpu_data[i] != 1) 
        {
            test = 0;
            cout << cpu_data[i] << endl;
        }
    }

    return test;
}

//test 7 : copy array of ones from custom cupy to cpu
uint32_t test_copy_custom_cupy_of_ones_to_cpu(Custom_Cupy_Ref b)
{
    double *cpu_data = (double*)malloc(b.size*sizeof(double));

    CUDA_CHECK(cudaMemcpy(cpu_data, b.ptr, b.size*sizeof(double), cudaMemcpyDeviceToHost));

    uint32_t test = 1;

    for(uint32_t i = 0; i < b.size; i++)
    {
        if(cpu_data[i] != 1) 
        {
            test = 0;
        }
    }
    return test;
}

//test 8 : copy array of float from custom cupy to cpu
uint32_t test_copy_custom_cupy_of_float_to_cpu(Custom_Cupy_Ref b)
{
    double *cpu_data = (double*)malloc(b.size*sizeof(double));

    CUDA_CHECK(cudaMemcpy(cpu_data, b.ptr, b.size*sizeof(double), cudaMemcpyDeviceToHost));

    uint32_t test = 1;

    for(uint32_t i = 0; i < b.size; i++)
    {
        if(cpu_data[i] != 3.14) 
        {
            test = 0;
        }
    }

    return test;
}

//test 9 : test 2 reinterpret cast with different data type and see if both has a same value
//This is still not clear why different type return the same adress value
uint32_t test_2_different_reiterpret_cast(size_t a_adress)
{
    double *ptr1 = reinterpret_cast<double *>(a_adress);;

    uint32_t *ptr2 = reinterpret_cast<uint32_t *>(a_adress);

    stringstream s1, s2;
    s1 << ptr1;
    s2 << ptr2;

    if(s1.str() == s2.str()) 
    {
        cout<<endl;
        cout<<s1.str()<<endl;
        cout<<s2.str()<<endl;
        return 1;
    }
    else return 0;
}