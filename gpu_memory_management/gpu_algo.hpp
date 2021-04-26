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

double *global_parted_image_0_dev;
double *global_parted_image_1_dev;


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

//1st try---------------------------------------
py::array_t<double, py::array::c_style> update_images(py::array_t<double, py::array::c_style> parted_image, double update, int size, int device_number)
{
    py::buffer_info bufImg = parted_image.request();
    double *ptrImg = static_cast<double*>(bufImg.ptr);

    cudaSetDevice(device_number);
    int devId, numSMs;
    cudaGetDevice(&devId);
    cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, devId);

    double *parted_image_dev, *partial_update_dev;
    CUDA_CHECK(cudaMalloc(&parted_image_dev, size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&partial_update_dev, size * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(parted_image_dev, ptrImg, size * sizeof(double), cudaMemcpyHostToDevice));
    
    partial_image_update<<<8*numSMs, 256>>>(parted_image_dev, partial_update_dev, update, size);

    py::array_t<double, py::array::c_style> result = py::array_t<double, py::array::c_style>(bufImg.size);
    py::buffer_info bufRes = result.request();
    double *ptrRes = static_cast<double*>(bufRes.ptr);
    CUDA_CHECK(cudaMemcpy(ptrRes, partial_update_dev, size * sizeof(double), cudaMemcpyDeviceToHost));
    
    cudaFree(parted_image_dev);
    cudaFree(partial_update_dev);

    return result;
}

//2nd try, segmentation fault-------------------------------------------------------
double * copy_to_device(py::array_t<double, py::array::c_style> parted_image, int size, int device_number)
{
    py::buffer_info bufImg = parted_image.request();
    double *ptrImg = static_cast<double*>(bufImg.ptr);
    
    cudaSetDevice(device_number);
    double *parted_image_dev;
    CUDA_CHECK(cudaMalloc(&parted_image_dev, size * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(parted_image_dev, ptrImg, size * sizeof(double), cudaMemcpyHostToDevice));
    return parted_image_dev;
}


//3rd try-----------------------------------
void copy_parted_image_to_device(py::array_t<double, py::array::c_style> parted_image, int size, int device_number)
{
    py::buffer_info bufImg = parted_image.request();
    double *ptrImg = static_cast<double*>(bufImg.ptr);
    
    if(device_number == 0)
    {
        cudaSetDevice(device_number);
        CUDA_CHECK(cudaMalloc(&global_parted_image_0_dev, size * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(global_parted_image_0_dev, ptrImg, size * sizeof(double), cudaMemcpyHostToDevice));
    }
    else if(device_number == 1)
    {
        cudaSetDevice(device_number);
        CUDA_CHECK(cudaMalloc(&global_parted_image_1_dev, size * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(global_parted_image_1_dev, ptrImg, size * sizeof(double), cudaMemcpyHostToDevice));
    }
}

py::array_t<double, py::array::c_style> update_images_v3(double update, int size, int device_number)
{
    cudaSetDevice(device_number);
    int devId, numSMs;
    cudaGetDevice(&devId);
    cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, devId);

    double *partial_update_dev;
    CUDA_CHECK(cudaMalloc(&partial_update_dev, size * sizeof(double)));
  
    py::array_t<double, py::array::c_style> result = py::array_t<double, py::array::c_style>(size);
    py::buffer_info bufRes = result.request();
    double *ptrRes = static_cast<double*>(bufRes.ptr);

    if(device_number == 0)
    {
        partial_image_update<<<8*numSMs, 256>>>(global_parted_image_0_dev, partial_update_dev, update, size);
        CUDA_CHECK(cudaMemcpy(ptrRes, partial_update_dev, size * sizeof(double), cudaMemcpyDeviceToHost));
        
        cudaFree(global_parted_image_0_dev);
        cudaFree(partial_update_dev);
    }
    else if(device_number == 1)
    {
        partial_image_update<<<8*numSMs, 256>>>(global_parted_image_1_dev, partial_update_dev, update, size);
        CUDA_CHECK(cudaMemcpy(ptrRes, partial_update_dev, size * sizeof(double), cudaMemcpyDeviceToHost));
        
        cudaFree(global_parted_image_1_dev);
        cudaFree(partial_update_dev);
    }

    return result;
}

//4th try--------------------------------------
//similiar to 1st try, learning cudastream with multi-GPU
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

