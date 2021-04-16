#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
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


__global__ void image_update(double *parted_image, double *partial_update, double update, int size)
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

py::array_t<double, py::array::c_style> update_images(py::array_t<double, py::array::c_style> parted_image, double update, int size, int dev_number)
{
    py::buffer_info bufImg = parted_image.request();
    double *ptrImg = static_cast<double*>(bufImg.ptr);

    double *partial_update;

    cudaSetDevice(dev_number);
    int devId, numSMs;
    cudaGetDevice(&devId);
    cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, devId);

    double *parted_image_dev, *partial_update_dev;
    CUDA_CHECK(cudaMalloc(&parted_image_dev, size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&partial_update_dev, size * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(parted_image_dev, ptrImg, size * sizeof(double), cudaMemcpyHostToDevice));
  
    
    image_update<<<8*numSMs, 256>>>(parted_image_dev, partial_update_dev, update, size);

    py::array_t<double, py::array::c_style> result = py::array_t<double, py::array::c_style>(bufImg.size);
    py::buffer_info bufRes = result.request();
    double *ptrRes = static_cast<double*>(bufRes.ptr);
    CUDA_CHECK(cudaMemcpy(ptrRes, partial_update_dev, size * sizeof(double), cudaMemcpyDeviceToHost));

    return result;
}

