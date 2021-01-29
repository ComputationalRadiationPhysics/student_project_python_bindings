#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cufft.h>
#include <cuComplex.h>
#include <cmath>
#include <cstdio>
#include <time.h>
#include <iostream>
#include <complex>
#include <string>
#include <chrono>

#define PI 3.1415926535897932384626433
#define CUDA_CHECK(call) {cudaError_t error = call; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}
using namespace std;

namespace py = pybind11;

//test, still not used
__device__ cuDoubleComplex my_complex_exp (cuDoubleComplex arg)
{
   cuDoubleComplex res;
   float s, c;
   float e = expf(arg.x);
   sincosf(arg.y, &s, &c);
   res.x = c * e;
   res.y = s * e;
   return res;
}

__global__ void normalize(cufftDoubleComplex * data, cufftDoubleComplex * data_res, const int size_y, const int size_x)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = size_y * size_x;
    if (idx < size)
    {
        data_res[idx].x = data[idx].x / static_cast<float>(size);
        data_res[idx].y = data[idx].y / static_cast<float>(size);
    }
}

__global__ void get_real(cufftDoubleComplex *temp_y, double *y, const int size_y, const int size_x)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = size_y * size_x;
    if (idx < size)
    {
        y[idx] = temp_y[idx].x;
    }
}

__global__ void get_complex(double *image_x, cufftDoubleComplex *image_x_comp, const int size_y, const int size_x)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = size_y * size_x;
    if (idx < size)
    {
        image_x_comp[idx].x = image_x[idx];
        image_x_comp[idx].y = 0;
    }
}


__global__ void process_arrays(double *mask, double *y, double *image_x, double *image_x_p, double beta, int mode, int iter, const int size_y, const int size_x)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = size_y * size_x;
    if (idx < size)
    {
        bool logical_not_mask;
        bool y_less_than_zero;
        bool logical_and;
        bool indices; //logical or

        //previous iterate
        if(iter == 0) image_x_p[idx] = y[idx];
        else image_x_p[idx] = image_x[idx];

        //updates for elements that satisfy object domain constraints
        if(mode == 3 || mode == 1) image_x[idx] = y[idx];

        //find elements that violate object domain constraints 
        //or are not masked 
        //1. logical not of mask
        if(mask[idx] <= 0) logical_not_mask = true;
        else if(mask[idx] >= 1) logical_not_mask = false;

        //2. check if any element y is less than zero
        if(y[idx] < 0) y_less_than_zero = true;
        else if(y[idx] >= 0) y_less_than_zero = false;

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
                image_x[idx] = image_x_p[idx]-beta*y[idx];
            }
            else if(mode == 3)
            {
                image_x[idx] = y[idx]-beta*y[idx];
            }
        }
    }
}

py::array_t<double> fienup_phase_retrieval(py::array_t<complex<double>> mag, int steps, bool verbose, string mode, double beta)
{
  using namespace std::literals::complex_literals;
    assert(beta > 0);
    assert(steps > 0);
    assert(mode == "input-output" || mode == "output-output" || mode == "hybrid");

    srand( (unsigned)time( NULL ) );
    py::buffer_info bufMag = mag.request();

    int int_mode;
    if(mode.compare("hybrid") == 0) int_mode = 1;
    else if(mode.compare("input-output") == 0) int_mode = 2;
    else if(mode.compare("output-output") == 0) int_mode = 3;

    complex<double> *ptrMag = (complex<double> *) bufMag.ptr; //magnitude 1D
    size_t X = bufMag.shape[1]; //width of magnitude
    size_t Y = bufMag.shape[0]; //height of magnitude
    
    //alternative fot saving mag size, prevent warning while using CUFFT 
    //( warning C4267: 'argument': conversion from 'size_t' to 'int', possible loss of data)"
    //get int version of size instead of size_t, then create dimension (mag size)
    int size_x = static_cast<int>(X); 
    int size_y = static_cast<int>(Y);
    int dimension = size_x*size_y;
    complex<double> complex1i(0, 1);

    //allocating arrays, all arrays bellow are 1D representation of 2D array
    double *mask = new double[dimension]; //mask array, same size as magnitude
    double *image_x = new double[dimension]; //initial image x, same size as magnitude
    double *image_x_p = new double[dimension]; //previous image for steps
    complex<double> *y_hat = new complex<double>[dimension]; //sample random phase
    complex<double> *x_hat = new complex<double>[dimension];

    auto begin = chrono::high_resolution_clock::now();

    //allocating inital values to arrays
    fill_n(&mask[0], dimension, 1.0); //fill mask with 1.0
    fill_n(&image_x[0], dimension, 0.0); //fill image_x with 0.0 for initial data
    fill_n(&image_x_p[0], dimension, 0.0); //fill image_x_p with 0.0 for initial data
    
    //this is not natively support in CUDA, still finding solution
    for (int i = 0; i < dimension; i++)
    {
        double rand_num = (double) rand()/RAND_MAX;
        y_hat[i] = ptrMag[i]*exp(complex1i*2.0*PI*rand_num); //random phase initial value
    }

    double *y_dev_res;
    cufftDoubleComplex *y_dev_start, *y_dev_start_norm;
    CUDA_CHECK(cudaMalloc((void **) &y_dev_res, dimension * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **) &y_dev_start, dimension * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc((void **) &y_dev_start_norm, dimension * sizeof(cufftDoubleComplex)));
    
    double *mask_dev, *image_x_device, *image_x_p_device;
    CUDA_CHECK(cudaMalloc((void **) &mask_dev, dimension * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **) &image_x_device, dimension * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **) &image_x_p_device, dimension * sizeof(double)));

    cufftDoubleComplex *image_x_dev_comp, *image_x_dev_comp_norm;
    CUDA_CHECK(cudaMalloc((void **) &image_x_dev_comp, dimension * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc((void **) &image_x_dev_comp_norm, dimension * sizeof(cufftDoubleComplex)));

    CUDA_CHECK(cudaMemcpy(mask_dev, mask, dimension * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(image_x_device, image_x, dimension * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(image_x_p_device, image_x_p, dimension * sizeof(double), cudaMemcpyHostToDevice));

    //iteration with number of steps------------------------------------------------------------------------------------------------------
    for(int iter = 0; iter < steps; iter++)
    {
        cufftResult fftresult;
        CUDA_CHECK(cudaMemcpy(y_dev_start, y_hat, dimension * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));

        //create cufft plan
        //use Z2Z for complex double to complex double, use Z2Z because Z2D has no inverse option
        cufftHandle plan;
        fftresult = cufftPlan2d(&plan, size_x, size_y, CUFFT_Z2Z);
        if(fftresult != CUFFT_SUCCESS) cout<<iter<<"\t"<<fftresult<<endl;

        fftresult = cufftExecZ2Z(plan, y_dev_start, y_dev_start, CUFFT_INVERSE);
        if(fftresult != CUFFT_SUCCESS) cout<<iter<<"\t"<<fftresult<<endl;

        normalize<<<size_y, size_x>>>(y_dev_start, y_dev_start, size_y, size_x);

        //change temp_y to y, which is a real number version of temp_y
        get_real<<<size_y, size_x>>>(y_dev_start, y_dev_res, size_y, size_x);

        //processing image_x
        process_arrays<<<size_y, size_x>>>(mask_dev, y_dev_res, image_x_device, image_x_p_device, beta, int_mode, iter, size_y, size_x);

        //fourier transform---------------------------------------------------------------------------------------------------------
        //convert real to complex (using 0 as imaginary)
        get_complex<<<size_y, size_x>>>(image_x_device, image_x_dev_comp, size_y, size_x);

        //there is actually cufftD2Z (double to complex double), but it doesnt work
        fftresult = cufftExecZ2Z(plan, image_x_dev_comp, image_x_dev_comp, CUFFT_FORWARD);
        if(fftresult != CUFFT_SUCCESS) cout<<iter<<"\t"<<fftresult<<endl;

        CUDA_CHECK(cudaMemcpy(x_hat, image_x_dev_comp, dimension * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));

        //this is not natively support in CUDA, still finding solution
        for(int i = 0; i < dimension; i++) 
        {
            y_hat[i] = ptrMag[i]*exp(complex1i*arg(x_hat[i]));
        }
        
        cufftDestroy(plan);
    }

    //copy image_x from device to host
    CUDA_CHECK(cudaMemcpy(image_x, image_x_device, dimension * sizeof(double), cudaMemcpyDeviceToHost));
    
    cudaFree(y_dev_start);
    cudaFree(y_dev_res);
    cudaFree(mask_dev);
    cudaFree(image_x_device); 
    cudaFree(image_x_p_device);
    cudaFree(image_x_dev_comp);

    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);

    printf("Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);

    //free all arrays
    delete[] mask;
    delete[] image_x_p;
    delete[] y_hat;

    py::array_t<double> image_x_2d =  py::array(dimension, image_x);
    //image_x_2d.resize({Y,X});
    return image_x_2d;
}


py::array_t<complex<double>> test_fft(py::array_t<complex<double>> mag)
{
    py::buffer_info bufMag = mag.request();
    complex<double> *ptrMag = (complex<double> *) bufMag.ptr; //magnitude 1D
    size_t X = bufMag.shape[1]; //width of magnitude
    size_t Y = bufMag.shape[0]; //height of magnitude
    int size_x = static_cast<int>(X); 
    int size_y = static_cast<int>(Y);
    int dimension = size_x*size_y;
    complex<double> *ptrMag2 = new complex<double>[dimension];
    complex<double> *ptrMag3 = new complex<double>[dimension];

    cufftDoubleComplex *mag_dev;
    CUDA_CHECK(cudaMalloc((void **) &mag_dev, dimension * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMemcpy(mag_dev, ptrMag, dimension * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));

    cufftHandle plan1;
    cufftResult fftresult;

    fftresult = cufftPlan2d(&plan1, size_x, size_y, CUFFT_Z2Z);
    if(fftresult != CUFFT_SUCCESS) cout<<fftresult<<endl;
    
    fftresult = cufftExecZ2Z(plan1, mag_dev, mag_dev, CUFFT_INVERSE);
    if(fftresult != CUFFT_SUCCESS) cout<<fftresult<<endl;

    normalize<<<size_y, size_x>>>(mag_dev, mag_dev, size_y, size_x);  

    
    cufftDestroy(plan1);

    CUDA_CHECK(cudaMemcpy(ptrMag2, mag_dev, dimension * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));

    py::array_t<complex<double>> image_x_2d =  py::array(dimension, ptrMag2);
    return image_x_2d;

}