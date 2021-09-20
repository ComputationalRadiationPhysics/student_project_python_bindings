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

__global__ void partial_image_update(double *parted_image, double *partial_update, double update, int size)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x)  
    {
        partial_update[idx] = parted_image[idx] + update;
    } 
}

int getDeviceNumber()
{
    int num;
    CUDA_CHECK(cudaGetDeviceCount(&num));
    return num;
}

int getNumberofSM()
{
    int devId, numSMs;
    CUDA_CHECK(cudaGetDevice(&devId));
    CUDA_CHECK(cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, devId));
    return numSMs;
}

void copy_to_device(std::size_t gpu_image, pybind11::array_t<double, pybind11::array::c_style> image, int size, int device)
{
    pybind11::buffer_info bufImg = image.request();

    CUDA_CHECK(cudaSetDevice(device));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    double *host_image = static_cast<double*>(bufImg.ptr);
    double *device_image = reinterpret_cast<double*>(gpu_image);

    CUDA_CHECK(cudaMemcpyAsync(device_image, host_image, size * sizeof(double), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaDeviceSynchronize());
}

void update_images(std::size_t gpu_image, std::size_t gpu_partial_update, double update, int size, int device) 
{
    CUDA_CHECK(cudaSetDevice(device));
    double *device_image = reinterpret_cast<double*>(gpu_image);
    double *device_partial_update = reinterpret_cast<double*>(gpu_partial_update);

    int numSMs = getNumberofSM();

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    partial_image_update<<<8*numSMs, 256, 0, stream>>>(device_image, device_partial_update, update, size);

    CUDA_CHECK(cudaDeviceSynchronize());
}

void free_gpu_memory(std::size_t device_array, int device) 
{
    CUDA_CHECK(cudaSetDevice(device));
    double *gpu_array = reinterpret_cast<double*>(device_array);

    CUDA_CHECK(cudaFree(gpu_array));
}