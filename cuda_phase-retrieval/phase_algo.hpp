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

#define PI 3.1415926535897932384626433
#define CUDA_CHECK(call) {cudaError_t error = call; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}
using namespace std;
using namespace std::literals::complex_literals;
namespace py = pybind11;

__device__ cufftDoubleComplex gpu_exp(cufftDoubleComplex arg);
__device__ cufftDoubleComplex normalize(cufftDoubleComplex comp_data, int size);
__device__ cufftDoubleComplex get_complex(double real_data);
__device__ double get_real(cufftDoubleComplex comp_data);
__global__ void get_complex_array(double *real_array, cufftDoubleComplex *complex_array, int dimension);
__global__ void get_absolute_array(cufftDoubleComplex *complex_array, double *real_array , int dimension);
__global__ void init_random(double seed, curandState_t *states, int dimension);
__global__ void random_phase(double *random, cufftDoubleComplex *y_hat, double *ptrMag, int dimension );
__global__ void random_phase_cudastate(curandState_t *states, cufftDoubleComplex *y_hat, double *ptrMag, int dimension );
__global__ void satisfy_fourier(cufftDoubleComplex *y_hat, cufftDoubleComplex *x_hat, double *ptrMag, int dimension );
__global__ void process_arrays(double *mask, cufftDoubleComplex *y_hat, double *image_x, double *image_x_p, cufftDoubleComplex *image_x_comp, double beta, int mode, int iter, int dimension);
void CUFFT_CHECK(cufftResult cufft_process);

py::array_t<double, py::array::c_style> fienup_phase_retrieval(py::array_t<double, py::array::c_style> image, py::array_t<double, py::array::c_style> masks, int steps, string mode, double beta, py::array_t<double, py::array::c_style> randoms)
{
    //asserting inputs
    assert(beta > 0);
    assert(steps > 0);
    assert(mode == "input-output" || mode == "output-output" || mode == "hybrid");

    py::buffer_info bufImg = image.request();
    py::buffer_info bufMask = masks.request();
    py::buffer_info bufRand = randoms.request();

    int int_mode;
    if(mode.compare("hybrid") == 0) int_mode = 1;
    else if(mode.compare("input-output") == 0) int_mode = 2;
    else if(mode.compare("output-output") == 0) int_mode = 3;

    double *ptrImg = static_cast<double*>(bufImg.ptr); //input image array
    size_t X = bufImg.shape[0]; //width of image
    size_t Y = bufImg.shape[1]; //height of image
    double *mask = static_cast<double*>(bufMask.ptr); //mask array, same size as image
    double *random_value = static_cast<double*>(bufRand.ptr); //array of uniform random number, same size as image

    //alternative fot saving image size, prevent warning while using CUFFT 
    //( warning C4267: 'argument': conversion from 'size_t' to 'int', possible loss of data)"
    //get int version of size instead of size_t, then create dimension (image size)
    int size_x = static_cast<int>(X); 
    int size_y = static_cast<int>(Y);
    int dimension = size_x*size_y;

    //initialize arrays for GPU
    double *src_img_dev, *mag_dev, *mask_dev, *image_x_device, *image_x_p_device, *random_value_dev;
    cufftDoubleComplex *y_hat_dev, *src_img_dev_comp, *image_x_dev_comp;
    CUDA_CHECK(cudaMalloc(&y_hat_dev, dimension * sizeof(cufftDoubleComplex))); //sample random phase
    CUDA_CHECK(cudaMalloc(&src_img_dev_comp, dimension * sizeof(cufftDoubleComplex))); //complex version of source image in device
    CUDA_CHECK(cudaMalloc(&mag_dev, dimension * sizeof(double))); //device magnitudes
    CUDA_CHECK(cudaMalloc(&src_img_dev, dimension * sizeof(double))); //source image in device
    CUDA_CHECK(cudaMalloc(&mask_dev, dimension * sizeof(double))); //device mask
    CUDA_CHECK(cudaMalloc(&image_x_device, dimension * sizeof(double))); //image x in device
    CUDA_CHECK(cudaMalloc(&image_x_p_device, dimension * sizeof(double))); //image x_p in device
    CUDA_CHECK(cudaMalloc(&image_x_dev_comp, dimension * sizeof(cufftDoubleComplex))); //complex version if image x in device
    CUDA_CHECK(cudaMalloc(&random_value_dev, dimension * sizeof(double))); //array of random in device

    //allocating inital values to device
    cudaMemset(image_x_device,  0, dimension * sizeof(double));
    cudaMemset(image_x_p_device, 0, dimension * sizeof(double));

    //get number of SM on GPU
    int devId, numSMs;
    cudaGetDevice(&devId);
    cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, devId);

    //do CUFFT first time to image, then get the absolute value of the result
    // the absolute result is called magnitude
    CUDA_CHECK(cudaMemcpy(src_img_dev, ptrImg, dimension * sizeof(double), cudaMemcpyHostToDevice));
    get_complex_array<<<8*numSMs, 256>>>(src_img_dev, src_img_dev_comp, dimension);
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan2d(&plan, size_x, size_y, CUFFT_Z2Z));
    CUFFT_CHECK(cufftExecZ2Z(plan, src_img_dev_comp, src_img_dev_comp, CUFFT_FORWARD));
    cufftDestroy(plan);
    get_absolute_array<<<8*numSMs, 256>>>(src_img_dev_comp, mag_dev, dimension);


    //copy mask and random value to gpu
    CUDA_CHECK(cudaMemcpy(mask_dev, mask, dimension * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(random_value_dev, random_value, dimension * sizeof(double), cudaMemcpyHostToDevice));

    //initial random phase
    random_phase<<<8*numSMs, 256>>>(random_value_dev, y_hat_dev, mag_dev, dimension);

    //iteration with number of steps------------------------------------------------------------------------------------------------------
    for(int iter = 0; iter < steps; iter++)
    {   
        cufftHandle plan; //create cufft plan
        CUFFT_CHECK(cufftPlan2d(&plan, size_x, size_y, CUFFT_Z2Z));
        CUFFT_CHECK(cufftExecZ2Z(plan, y_hat_dev, y_hat_dev, CUFFT_INVERSE));
        process_arrays<<<8*numSMs, 256>>>(mask_dev, y_hat_dev, image_x_device, image_x_p_device, image_x_dev_comp, beta, int_mode, iter, dimension);
        CUFFT_CHECK(cufftExecZ2Z(plan, image_x_dev_comp, image_x_dev_comp, CUFFT_FORWARD));
        satisfy_fourier<<<8*numSMs, 256>>>(y_hat_dev, image_x_dev_comp, mag_dev, dimension);
        cufftDestroy(plan);
    }

    py::array_t<double, py::array::c_style> result = py::array_t<double, py::array::c_style>(bufImg.size);
    py::buffer_info bufRes = result.request();
    double *ptrRes = static_cast<double*>(bufRes.ptr);
    CUDA_CHECK(cudaMemcpy(ptrRes, image_x_device, dimension * sizeof(double), cudaMemcpyDeviceToHost));

    //free CUDA usage
    cudaFree(y_hat_dev);
    cudaFree(src_img_dev);
    cudaFree(random_value_dev);
    cudaFree(mag_dev);
    cudaFree(mask_dev);
    cudaFree(image_x_device); 
    cudaFree(image_x_p_device);
    cudaFree(image_x_dev_comp);

    result.resize({X, Y});
    return result;
}

py::array_t<double, py::array::c_style> fienup_phase_retrieval(py::array_t<double, py::array::c_style> image, py::array_t<double, py::array::c_style> masks, int steps, string mode, double beta)
{
    //asserting inputs
    assert(beta > 0);
    assert(steps > 0);
    assert(mode == "input-output" || mode == "output-output" || mode == "hybrid");

    py::buffer_info bufImg = image.request();
    py::buffer_info bufMask = masks.request();

    int int_mode;
    if(mode.compare("hybrid") == 0) int_mode = 1;
    else if(mode.compare("input-output") == 0) int_mode = 2;
    else if(mode.compare("output-output") == 0) int_mode = 3;

    double *ptrImg = static_cast<double*>(bufImg.ptr); //input image array
    size_t X = bufImg.shape[0]; //width of image
    size_t Y = bufImg.shape[1]; //height of image
    double *mask = static_cast<double*>(bufMask.ptr); //mask array, same size as image

    //alternative fot saving image size, prevent warning while using CUFFT 
    //( warning C4267: 'argument': conversion from 'size_t' to 'int', possible loss of data)"
    //get int version of size instead of size_t, then create dimension (image size)
    int size_x = static_cast<int>(X); 
    int size_y = static_cast<int>(Y);
    int dimension = size_x*size_y;

    //initialize arrays for GPU
    double *src_img_dev, *mag_dev, *mask_dev, *image_x_device, *image_x_p_device;
    cufftDoubleComplex *y_hat_dev, *src_img_dev_comp, *image_x_dev_comp;
    CUDA_CHECK(cudaMalloc(&y_hat_dev, dimension * sizeof(cufftDoubleComplex))); //sample random phase
    CUDA_CHECK(cudaMalloc(&src_img_dev_comp, dimension * sizeof(cufftDoubleComplex))); //complex version of source image in device
    CUDA_CHECK(cudaMalloc(&mag_dev, dimension * sizeof(double))); //device magnitudes
    CUDA_CHECK(cudaMalloc(&src_img_dev, dimension * sizeof(double))); //source image in device
    CUDA_CHECK(cudaMalloc(&mask_dev, dimension * sizeof(double))); //device mask
    CUDA_CHECK(cudaMalloc(&image_x_device, dimension * sizeof(double))); //image x in device
    CUDA_CHECK(cudaMalloc(&image_x_p_device, dimension * sizeof(double))); //image x_p in device
    CUDA_CHECK(cudaMalloc(&image_x_dev_comp, dimension * sizeof(cufftDoubleComplex))); //complex version if image x in device

    //allocating inital values to device
    cudaMemset(image_x_device,  0, dimension * sizeof(double));
    cudaMemset(image_x_p_device, 0, dimension * sizeof(double));

    //get number of SM on GPU
    int devId, numSMs;
    cudaGetDevice(&devId);
    cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, devId);

    //get states for curand
    srand( (unsigned)time( NULL ) );
    curandState_t* states;
    cudaMalloc(&states, dimension * sizeof(curandState_t));
    init_random<<<8*numSMs, 256>>>(static_cast<double>(time(0)), states, dimension);

    //do CUFFT first time to image, then get the absolute value of the result
    // the absolute result is called magnitude
    CUDA_CHECK(cudaMemcpy(src_img_dev, ptrImg, dimension * sizeof(double), cudaMemcpyHostToDevice));
    get_complex_array<<<8*numSMs, 256>>>(src_img_dev, src_img_dev_comp, dimension);
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan2d(&plan, size_x, size_y, CUFFT_Z2Z));
    CUFFT_CHECK(cufftExecZ2Z(plan, src_img_dev_comp, src_img_dev_comp, CUFFT_FORWARD));
    cufftDestroy(plan);
    get_absolute_array<<<8*numSMs, 256>>>(src_img_dev_comp, mag_dev, dimension);

    //copy mask to gpu
    CUDA_CHECK(cudaMemcpy(mask_dev, mask, dimension * sizeof(double), cudaMemcpyHostToDevice));

    //initial random phase
    random_phase_cudastate<<<8*numSMs, 256>>>(states, y_hat_dev, mag_dev, dimension);

    //iteration with number of steps------------------------------------------------------------------------------------------------------
    for(int iter = 0; iter < steps; iter++)
    {   
        cufftHandle plan; //create cufft plan
        CUFFT_CHECK(cufftPlan2d(&plan, size_x, size_y, CUFFT_Z2Z));
        CUFFT_CHECK(cufftExecZ2Z(plan, y_hat_dev, y_hat_dev, CUFFT_INVERSE));
        process_arrays<<<8*numSMs, 256>>>(mask_dev, y_hat_dev, image_x_device, image_x_p_device, image_x_dev_comp, beta, int_mode, iter, dimension);
        CUFFT_CHECK(cufftExecZ2Z(plan, image_x_dev_comp, image_x_dev_comp, CUFFT_FORWARD));
        satisfy_fourier<<<8*numSMs, 256>>>(y_hat_dev, image_x_dev_comp, mag_dev, dimension);
        cufftDestroy(plan);
    }

    py::array_t<double, py::array::c_style> result = py::array_t<double, py::array::c_style>(bufImg.size);
    py::buffer_info bufRes = result.request();
    double *ptrRes = static_cast<double*>(bufRes.ptr);
    CUDA_CHECK(cudaMemcpy(ptrRes, image_x_device, dimension * sizeof(double), cudaMemcpyDeviceToHost));

    //free CUDA usage
    cudaFree(y_hat_dev);
    cudaFree(src_img_dev);
    cudaFree(mag_dev);
    cudaFree(mask_dev);
    cudaFree(image_x_device); 
    cudaFree(image_x_p_device);
    cudaFree(image_x_dev_comp);

    result.resize({X, Y});
    return result;
}

__device__ cufftDoubleComplex gpu_exp(cufftDoubleComplex arg)
{
   cufftDoubleComplex res;
   float s, c;
   float e = expf(arg.x);
   sincosf(arg.y, &s, &c);
   res.x = c * e;
   res.y = s * e;
   return res;
}

//Normalize every result elements of CUFFT_INVERSE
__device__ cufftDoubleComplex normalize(cufftDoubleComplex comp_data, int dimension)
{
    cufftDoubleComplex norm_data;
    norm_data.x = comp_data.x / static_cast<double>(dimension);
    norm_data.y = comp_data.y / static_cast<double>(dimension);
    
    return norm_data;
}

//Convert real number to CUFFT complex number using 0 as imaginary part
__device__ cufftDoubleComplex get_complex(double real_data)
{
    cufftDoubleComplex comp_data;
    comp_data.x = real_data;
    comp_data.y = 0;

    return comp_data;
}

//Get real number part of a CUFFT complex number 
__device__ double get_real(cufftDoubleComplex comp_data)
{
    return comp_data.x;
}

//Convert array of real number into array of complex number
__global__ void get_complex_array(double *real_array, cufftDoubleComplex *complex_array, int dimension)
{
   for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dimension; idx += blockDim.x * gridDim.x)  
    {
        cufftDoubleComplex comp_data;
        comp_data.x = real_array[idx];
        comp_data.y = 0;
        complex_array[idx] = comp_data;
    }
}

//Get absolute of complex number
__global__ void get_absolute_array(cufftDoubleComplex *complex_array, double *real_array , int dimension)
{
   for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dimension; idx += blockDim.x * gridDim.x)  
    {
        real_array[idx] = cuCabs(complex_array[idx]);
    }
}

//create states for random values
__global__ void init_random(double seed, curandState_t *states, int dimension)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dimension; idx += blockDim.x * gridDim.x) 
    { 
        curand_init(seed, idx, 0, &states[idx]);
    }
}

//sample random phase using array of random from input
__global__ void random_phase(double *random, cufftDoubleComplex *y_hat, double *ptrMag, int dimension ) 
{
    cufftDoubleComplex complex1i, exp_target, mag_comp;
    complex1i.x = 0; complex1i.y = 1;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dimension; idx += blockDim.x * gridDim.x)  
    {
        mag_comp.x = ptrMag[idx];
        mag_comp.y = 0;
        exp_target.x = PI*2.0*random[idx];
        exp_target.y = 0;
        y_hat[idx] = cuCmul(mag_comp, gpu_exp(cuCmul(complex1i, exp_target)));
    }
}

//sample random phase using curandState_t as random value
__global__ void random_phase_cudastate(curandState_t *states, cufftDoubleComplex *y_hat, double *ptrMag, int dimension ) 
{
    cufftDoubleComplex complex1i, exp_target, mag_comp;
    complex1i.x = 0; complex1i.y = 1;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dimension; idx += blockDim.x * gridDim.x)  
    {
        mag_comp.x = ptrMag[idx];
        mag_comp.y = 0;
        exp_target.x = PI*2.0*curand_uniform(&states[idx]);
        exp_target.y = 0;
        y_hat[idx] = cuCmul(mag_comp, gpu_exp(cuCmul(complex1i, exp_target)));
    }
}

//satisfy fourier domain constraints
__global__ void satisfy_fourier(cufftDoubleComplex *y_hat, cufftDoubleComplex *x_hat, double *ptrMag, int dimension ) 
{
    cufftDoubleComplex complex1i, exp_target, mag_comp;
    complex1i.x = 0; complex1i.y = 1;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dimension; idx += blockDim.x * gridDim.x)  
    {
        mag_comp.x = ptrMag[idx];
        mag_comp.y = 0;
        //arg = atan2(imag, real)
        exp_target.x = atan2(x_hat[idx].y, x_hat[idx].x);
        exp_target.y = 0;
        y_hat[idx] = cuCmul(mag_comp, gpu_exp(cuCmul(complex1i, exp_target)));
    }
}

//processing magnitudes with mask
__global__ void process_arrays(double *mask, cufftDoubleComplex *y_hat, double *image_x, double *image_x_p, cufftDoubleComplex *image_x_comp, double beta, int mode, int iter, int dimension)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dimension; idx += blockDim.x * gridDim.x)  
    {
        bool logical_not_mask;
        bool y_less_than_zero;
        bool logical_and;
        bool indices;

        double y = get_real(normalize(y_hat[idx], dimension));

        if(iter == 0) image_x_p[idx] = y;
        else image_x_p[idx] = image_x[idx];

        //updates for elements that satisfy object domain constraints
        if(mode == 3 || mode == 1) image_x[idx] = y;

        //find elements that violate object domain constraints or are not masked 
        if(mask[idx] <= 0) logical_not_mask = true;
        else if(mask[idx] >= 1) logical_not_mask = false;

        //check if any element y is less than zero
        if(y < 0) y_less_than_zero = true;
        else if(y >= 0) y_less_than_zero = false;

        //use "and" logical to check the "less than zero y" and the mask  
        if(y_less_than_zero == true && mask[idx] >= 1) logical_and = true;
        else logical_and = false;

        //create indices with logical "not"
        if(logical_and == false && logical_not_mask == false) indices = false;
        else indices = true;

        //updates for elements that violate object domain constraints
        if(indices == true)
        {
            if(mode == 1 || mode == 2)
            {
                image_x[idx] = image_x_p[idx]-beta*y;
            }
            else if(mode == 3)
            {
                image_x[idx] = y-beta*y;
            }
        }

        image_x_comp[idx] = get_complex(image_x[idx]);
    }
}

//CUFFT error checking
void CUFFT_CHECK(cufftResult cufft_process)
{
    if(cufft_process != CUFFT_SUCCESS) cout<<cufft_process<<endl;
}