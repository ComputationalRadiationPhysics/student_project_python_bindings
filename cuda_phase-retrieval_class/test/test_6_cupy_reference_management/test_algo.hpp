#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
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

using namespace std::literals::complex_literals;

__global__ void copy_value(double *ptrMag, double *ptrRes, int dimension);
__global__ void copy_value_complex(cufftDoubleComplex *ptrMag, cufftDoubleComplex *ptrRes, int dimension);
__global__ void normalize_array(cufftDoubleComplex *ptrImg, cufftDoubleComplex *ptrRes, int dimension);

//Test whether array of double received by C++ code is the same as initial python value
pybind11::array_t<double, pybind11::array::c_style> array_check(pybind11::array_t<double, pybind11::array::c_style> img)
{
    pybind11::buffer_info bufImg = img.request();
    size_t X = bufImg.shape[0]; //width of magnitude
    size_t Y = bufImg.shape[1]; //height of magnitude

    int size_x = static_cast<int>(X);
    int size_y = static_cast<int>(Y);
    int dimension = size_x*size_y;

    double *ptrImg = static_cast<double*>(bufImg.ptr); //magnitude 1D

    pybind11::array_t<double, pybind11::array::c_style> result = pybind11::array_t<double, pybind11::array::c_style>(bufImg.size);
    pybind11::buffer_info bufRes = result.request();
    double *ptrRes = static_cast<double*>(bufRes.ptr);

    for(int i = 0; i < dimension; i++)
    {
        ptrRes[i] = ptrImg[i];
    }

    //send to python
    result.resize({X, Y});
    return result;
}

//Test whether array of double received by CUDA code is the same as initial python value
pybind11::array_t<double, pybind11::array::c_style> array_check_cuda(pybind11::array_t<double, pybind11::array::c_style> img)
{
    pybind11::buffer_info bufImg = img.request();
    double *ptrImg = static_cast<double*>(bufImg.ptr); //magnitude 1D
    size_t X = bufImg.shape[0]; //width of magnitude
    size_t Y = bufImg.shape[1]; //height of magnitude

    int size_x = static_cast<int>(X);
    int size_y = static_cast<int>(Y);
    int dimension = size_x * size_y;

    pybind11::array_t<double, pybind11::array::c_style> result = pybind11::array_t<double, pybind11::array::c_style>(bufImg.size);
    pybind11::buffer_info bufRes = result.request();
    double *ptrRes = static_cast<double*>(bufRes.ptr);


    int devId, numSMs;
    cudaGetDevice(&devId);
    cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, devId);

    double *ptrImg_dev, *ptrRes_dev;
    CUDA_CHECK(cudaMalloc(&ptrImg_dev, dimension * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ptrRes_dev, dimension * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(ptrImg_dev, ptrImg, dimension * sizeof(double), cudaMemcpyHostToDevice));

    copy_value<<<8*numSMs, 256>>>(ptrImg_dev, ptrRes_dev, dimension);

    CUDA_CHECK(cudaMemcpy(ptrRes, ptrRes_dev, dimension * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(ptrImg_dev);
    cudaFree(ptrRes_dev);

    //send to python
    result.resize({X, Y});
    return result;
}

//Test whether array of complex received by C++ code is the same as initial python value
pybind11::array_t<std::complex<double>, pybind11::array::c_style> array_check_complex(pybind11::array_t<std::complex<double>, pybind11::array::c_style> mag)
{
    pybind11::buffer_info bufMag = mag.request();
    std::complex<double> *ptrMag =  static_cast<std::complex<double>*>(bufMag.ptr); //magnitude 1D
    size_t X = bufMag.shape[0]; //width of magnitude
    size_t Y = bufMag.shape[1]; //height of magnitude

    int size_x = static_cast<int>(X);
    int size_y = static_cast<int>(Y);

    pybind11::array_t<std::complex<double>, pybind11::array::c_style> result = pybind11::array_t<std::complex<double>, pybind11::array::c_style>(bufMag.size);
    pybind11::buffer_info bufRes = result.request();
    std::complex<double> *ptrRes = static_cast<std::complex<double>*>(bufRes.ptr);

    for(int i = 0; i < size_x * size_y; i++)
    {
        ptrRes[i] = ptrMag[i];
    }

    //send to python
    result.resize({X, Y});
    return result;
}

//Test whether array of complex received by CUDA code is the same as initial python value
pybind11::array_t<std::complex<double>, pybind11::array::c_style> array_check_complex_cuda(pybind11::array_t<std::complex<double>, pybind11::array::c_style> mag)
{
    pybind11::buffer_info bufMag = mag.request();
    std::complex<double> *ptrMag = static_cast<std::complex<double>*>(bufMag.ptr); //magnitude 1D
    size_t X = bufMag.shape[0]; //width of magnitude
    size_t Y = bufMag.shape[1]; //height of magnitude

    int size_x = static_cast<int>(X);
    int size_y = static_cast<int>(Y);
    int dimension = size_x * size_y;

    pybind11::array_t<std::complex<double>, pybind11::array::c_style> result = pybind11::array_t<std::complex<double>, pybind11::array::c_style>(bufMag.size);
    pybind11::buffer_info bufRes = result.request();
    std::complex<double> *ptrRes = static_cast<std::complex<double>*>(bufRes.ptr);

    int devId, numSMs;
    cudaGetDevice(&devId);
    cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, devId);

    cufftDoubleComplex *ptrMag_dev, *ptrRes_dev;
    CUDA_CHECK(cudaMalloc(&ptrMag_dev, dimension * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&ptrRes_dev, dimension * sizeof(cufftDoubleComplex)));

    CUDA_CHECK(cudaMemcpy(ptrMag_dev, ptrMag, dimension * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));

    copy_value_complex<<<8*numSMs, 256>>>(ptrMag_dev, ptrRes_dev, dimension);

    CUDA_CHECK(cudaMemcpy(ptrRes, ptrRes_dev, dimension * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));

    cudaFree(ptrMag_dev);
    cudaFree(ptrRes_dev);

    //send to python
    result.resize({X, Y});
    return result;
}

//Test whether array complex received by CUFFT (IFFT then FFT) is very close to the initial python value
pybind11::array_t<std::complex<double>, pybind11::array::c_style> cufft_inverse_forward(pybind11::array_t<std::complex<double>, pybind11::array::c_style> mag)
{
    pybind11::buffer_info bufMag = mag.request();
    std::complex<double> *ptrMag = static_cast<std::complex<double>*>(bufMag.ptr); //magnitude 1D
    size_t X = bufMag.shape[0]; //width of magnitude
    size_t Y = bufMag.shape[1]; //height of magnitude

    int size_x = static_cast<int>(X);
    int size_y = static_cast<int>(Y);
    int dimension = size_x * size_y;

    pybind11::array_t<std::complex<double>, pybind11::array::c_style> result = pybind11::array_t<std::complex<double>, pybind11::array::c_style>(bufMag.size);
    pybind11::buffer_info bufRes = result.request();
    std::complex<double> *ptrRes = static_cast<std::complex<double>*>(bufRes.ptr);

    int devId, numSMs;
    cudaGetDevice(&devId);
    cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, devId);

    cufftDoubleComplex *ptrMag_dev, *ptrRes_dev;
    CUDA_CHECK(cudaMalloc(&ptrMag_dev, dimension * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&ptrRes_dev, dimension * sizeof(cufftDoubleComplex)));

    CUDA_CHECK(cudaMemcpy(ptrMag_dev, ptrMag, dimension * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));

    cufftHandle plan; //create cufft plan

    CUFFT_CHECK(cufftPlan2d(&plan, size_x, size_y, CUFFT_Z2Z));

    CUFFT_CHECK(cufftExecZ2Z(plan, ptrMag_dev, ptrMag_dev, CUFFT_INVERSE));

    normalize_array<<<8*numSMs, 256>>>(ptrMag_dev, ptrRes_dev, dimension);

    CUFFT_CHECK(cufftExecZ2Z(plan, ptrRes_dev, ptrRes_dev, CUFFT_FORWARD));

    cufftDestroy(plan);

    CUDA_CHECK(cudaMemcpy(ptrRes, ptrRes_dev, dimension * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));

    cudaFree(ptrMag_dev);
    cudaFree(ptrRes_dev);

    //send to python
    result.resize({X, Y});
    return result;
}

pybind11::array_t<std::complex<double>, pybind11::array::c_style> abs_cufft_forward(pybind11::array_t<double, pybind11::array::c_style> image)
{
    pybind11::buffer_info bufImg = image.request();
    double *ptrImg = static_cast<double*>(bufImg.ptr); //magnitude 1D
    size_t X = bufImg.shape[0]; //width of magnitude
    size_t Y = bufImg.shape[1]; //height of magnitude

    int size_x = static_cast<int>(X);
    int size_y = static_cast<int>(Y);
    int dimension = size_x * size_y;

    pybind11::array_t<double, pybind11::array::c_style> result = pybind11::array_t<double, pybind11::array::c_style>(bufImg.size);
    pybind11::buffer_info bufRes = result.request();
    double *ptrRes = static_cast<double*>(bufRes.ptr);

    int devId, numSMs;
    cudaGetDevice(&devId);
    cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, devId);

    double *ptrImg_dev, *ptrResAbs_dev;
    cufftDoubleComplex *ptrRes_dev;
    CUDA_CHECK(cudaMalloc(&ptrImg_dev, dimension * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ptrResAbs_dev, dimension * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ptrRes_dev, dimension * sizeof(cufftDoubleComplex)));

    CUDA_CHECK(cudaMemcpy(ptrImg_dev, ptrImg, dimension * sizeof(double), cudaMemcpyHostToDevice));

    get_complex_array<<<8*numSMs, 256>>>(ptrImg_dev, ptrRes_dev, dimension);

    cufftHandle plan; //create cufft plan

    CUFFT_CHECK(cufftPlan2d(&plan, size_x, size_y, CUFFT_Z2Z));

    CUFFT_CHECK(cufftExecZ2Z(plan, ptrRes_dev, ptrRes_dev, CUFFT_FORWARD));

    cufftDestroy(plan);

    get_absolute_array<<<8*numSMs, 256>>>(ptrRes_dev, ptrResAbs_dev, dimension);

    CUDA_CHECK(cudaMemcpy(ptrRes, ptrResAbs_dev, dimension * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(ptrImg_dev);
    cudaFree(ptrRes_dev);
    cudaFree(ptrResAbs_dev);

    //send to python
    result.resize({X, Y});
    return result;
}

//Normalize array of complex number (results of CUFFT INVERSE)
__global__ void normalize_array(cufftDoubleComplex *ptrImg, cufftDoubleComplex *ptrRes, int dimension)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dimension; idx += blockDim.x * gridDim.x)
    {
        ptrRes[idx].x = ptrImg[idx].x / static_cast<double>(dimension);
        ptrRes[idx].y = ptrImg[idx].y / static_cast<double>(dimension);
    }
}

//Simple array of double copy in CUDA
__global__ void copy_value(double *ptrImg, double *ptrRes, int dimension)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dimension; idx += blockDim.x * gridDim.x)
    {
        ptrRes[idx] = ptrImg[idx];
    }
}

//Simple array of complex double copy in CUDA
__global__ void copy_value_complex(cufftDoubleComplex *ptrMag, cufftDoubleComplex *ptrRes, int dimension)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dimension; idx += blockDim.x * gridDim.x)
    {
        ptrRes[idx] = ptrMag[idx];
    }
}
