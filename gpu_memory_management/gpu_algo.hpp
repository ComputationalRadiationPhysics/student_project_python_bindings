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
    cudaGetDeviceCount(&num);
    return num;
}

int getNumberofSM()
{
    int devId, numSMs;
    cudaGetDevice(&devId);
    cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, devId);
    return numSMs;
}

void copy_to_device(size_t gpu_image, py::array_t<double, py::array::c_style> image, int size, int device)
{
    py::buffer_info bufImg = image.request();

    cudaSetDevice(device);
    double *host_image = static_cast<double*>(bufImg.ptr);
    double *device_image = reinterpret_cast<double*>(gpu_image);

    cudaMemcpy(device_image, host_image, size * sizeof(double), cudaMemcpyHostToDevice);
}

void update_images(size_t gpu_image, size_t gpu_partial_update, double update, int size, int device) 
{
    cudaSetDevice(device);
    double *device_image = reinterpret_cast<double*>(gpu_image);
    double *device_partial_update = reinterpret_cast<double*>(gpu_partial_update);

    int devId, numSMs;
    cudaGetDevice(&devId);
    cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, devId);
    
    partial_image_update<<<8*numSMs, 256>>>(device_image, device_partial_update, update, size);

    cudaDeviceSynchronize();
}

void free_gpu_memory(size_t device_array, int device) 
{
    cudaSetDevice(device);
    double *gpu_array = reinterpret_cast<double*>(device_array);

    cudaFree(gpu_array);
}

//cuda stream test
py::array_t<double, py::array::c_style> update_images_stream(py::array_t<double, py::array::c_style> images, py::array_t<double, py::array::c_style> update, int size)
{
    py::buffer_info bufImg = images.request();
    py::buffer_info bufUpd = update.request();

    double *ptrImg = static_cast<double*>(bufImg.ptr);
    double *ptrUpd = static_cast<double*>(bufUpd.ptr);

    //split images
    vector<double> parted_images_0(ptrImg + 0, ptrImg + ((size/2)+1));
    vector<double> parted_images_1(ptrImg + ((size/2)+1), ptrImg + size);
    int size_0 = parted_images_0.size();
    int size_1 = parted_images_1.size();

    cudaStream_t streamA, streamB;
    double *partial_update_dev_0, *partial_update_dev_1;
    double *partial_images_dev_0, *partial_images_dev_1;

    //pinned memory, cannot use vector to receive partial update from device
    double *partial_update_0 = (double*) malloc(size_0 * sizeof(double));
    double *partial_update_1 = (double*) malloc(size_1 * sizeof(double));
    
    cudaSetDevice(0);
    int devId_0, numSMs_0;
    cudaGetDevice(&devId_0);
    cudaDeviceGetAttribute( &numSMs_0, cudaDevAttrMultiProcessorCount, devId_0);
    cudaStreamCreate(&streamA);
    CUDA_CHECK(cudaMalloc(&partial_update_dev_0, size_0 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&partial_images_dev_0, size_0 * sizeof(double)));

    cudaSetDevice(1);
    int devId_1, numSMs_1;
    cudaGetDevice(&devId_1);
    cudaDeviceGetAttribute( &numSMs_1, cudaDevAttrMultiProcessorCount, devId_1);
    cudaStreamCreate(&streamB);
    CUDA_CHECK(cudaMalloc(&partial_update_dev_1, size_1 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&partial_images_dev_1, size_1 * sizeof(double)));

    CUDA_CHECK(cudaMemcpyAsync(partial_images_dev_1, parted_images_1.data(), size_1 * sizeof(double), cudaMemcpyHostToDevice, streamB )) ;
    partial_image_update<<<8*numSMs_1, 256, 0, streamB>>>(partial_images_dev_1, partial_update_dev_1, ptrUpd[1], size_1);
    CUDA_CHECK(cudaMemcpyAsync(partial_update_1, partial_update_dev_1, size_1 * sizeof(double), cudaMemcpyDeviceToHost, streamB )) ;

    cudaSetDevice(0);
    CUDA_CHECK(cudaMemcpyAsync(partial_images_dev_0, parted_images_0.data(), size_0 * sizeof(double), cudaMemcpyHostToDevice, streamA )) ;
    partial_image_update<<<8*numSMs_0, 256, 0, streamA>>>(partial_images_dev_0, partial_update_dev_0, ptrUpd[0], size_0);
    CUDA_CHECK(cudaMemcpyAsync(partial_update_0, partial_update_dev_0, size_0 * sizeof(double), cudaMemcpyDeviceToHost, streamA )) ;

    vector<double> partial_update;

    partial_update.insert(partial_update.begin(), partial_update_0, partial_update_0 + size_0);
    partial_update.insert(partial_update.end(), partial_update_1, partial_update_1 + size_1);

    py::array_t<double, py::array::c_style> result = py::array_t<double, py::array::c_style>(size);
    py::buffer_info bufRes = result.request();
    double *ptrRes = static_cast<double*>(bufRes.ptr);
    copy(partial_update.begin(), partial_update.end(), ptrRes);

    cudaFree(partial_update_dev_0);
    cudaFree(partial_images_dev_0);
    cudaFree(partial_update_dev_1);
    cudaFree(partial_images_dev_1);

    return result;
}


