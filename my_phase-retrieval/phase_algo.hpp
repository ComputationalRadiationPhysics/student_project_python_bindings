#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cufft.h>
#include <cmath>
#include <cstdio>
#include <time.h>
#include <iostream>
#include <complex>
#include <string>
#include <chrono>

#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}
using namespace std;

namespace py = pybind11;

__global__ void get_real(cufftDoubleComplex *temp_y, double *y)
{
    int idx = blockIdx.x;
    y[idx] = temp_y[idx].x;
}

__global__ void get_complex(double *image_x, cufftDoubleComplex *image_x_comp)
{
    int idx = blockIdx.x;
    image_x_comp[idx].x = image_x[idx];
    image_x_comp[idx].y = 0;
}


//find elements that violate object domain constraints 
//or are not masked
__global__ void update_violated_elements(double *mask, double *y, double *image_x, double *image_x_p, double beta, int mode)
{
    int idx = blockIdx.x;
    bool logical_not_mask;
    bool y_less_than_zero;
    bool logical_and;
    bool indices; //logical or

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
        if(mode == 1)
        {
            image_x[idx] = image_x_p[idx]-beta*y[idx];
        }
        else if(mode == 2)
        {
            image_x[idx] = y[idx]-beta*y[idx];
        }
    }
}

py::array_t<double> fienup_phase_retrieval(py::array_t<double> mag, int steps, bool verbose, string mode, double beta)
{
  using namespace std::literals::complex_literals;
    assert(beta > 0);
    assert(steps > 0);
    assert(mode == "input-output" || mode == "output-output" || mode == "hybrid");

    srand( (unsigned)time( NULL ) );
    py::buffer_info bufMag = mag.request();

    double *ptrMag = (double *) bufMag.ptr; //magnitude 1D
    size_t X = bufMag.shape[1]; //width of magnitude
    size_t Y = bufMag.shape[0]; //heght of magnitude
    
    //alternative fot saving mag size, prevent warning while using CUFFT 
    //( warning C4267: 'argument': conversion from 'size_t' to 'int', possible loss of data)"
    //get int version of size instead of size_t, then create dimension (mag size)
    int size_x = static_cast<int>(X); 
    int size_y = static_cast<int>(Y);
    int dimension = size_x*size_y;

    //allocating arrays, all arrays bellow are 1D representation of 2D array
    double *mask = new double[dimension]; //mask array, same size as magnitude
    double *image_x = new double[dimension]; //initial image x, same size as magnitude
    double *image_x_p = new double[dimension]; //previous image for steps
    double *y = new double[dimension]; //store inverse fourier transform (real number)
    complex<double> *y_hat = new complex<double>[dimension]; //sample random phase
    complex<double> *temp_y =  new complex<double>[dimension]; //temporary complex version of y after CUFFT, before copying the real number to y
    complex<double> *x_hat = new complex<double>[dimension];

    auto begin = chrono::high_resolution_clock::now();

    //allocating inital values to arrays
    fill_n(&mask[0], dimension, 1.0); //fill mask with 1.0
    fill_n(&image_x[0], dimension, 0.0); //fill image_x with 0.0 for initial data
    fill_n(&image_x_p[0], dimension, 0.0); //fill image_x_p with 0.0 for initial data
    fill_n(&y[0], dimension, 0.0); //store inverse fourier transform (real number)
 
    for (int i = 0; i < dimension; i++)
    {
        double rand_num = (double) rand()/RAND_MAX;
        y_hat[i] = ptrMag[i]*exp(1i*2.0*3.14*rand_num); //random phase initial value
    }

    double *y_dev_res;
    cufftHandle plan;
    cufftDoubleComplex *y_dev_start, *temp_y_dev;
    CUDA_CHECK(cudaMalloc((void **) &y_dev_start, dimension * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc((void **) &temp_y_dev, dimension * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&y_dev_res, dimension * sizeof(double)));

    double *mask_dev, *y_device, *image_x_device, *image_x_p_device;
    CUDA_CHECK(cudaMalloc(&mask_dev, dimension * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&y_device, dimension * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&image_x_device, dimension * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&image_x_p_device, dimension * sizeof(double)));

    double *image_x_dev;
    cufftHandle plan2;
    cufftDoubleComplex *image_x_dev_comp, *image_x_dev_res;
    CUDA_CHECK(cudaMalloc((void **) &image_x_dev, dimension * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **) &image_x_dev_comp, dimension * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc((void **) &image_x_dev_res, dimension * sizeof(cufftDoubleComplex)));

    //iteration with number of steps------------------------------------------------------------------------------------------------------
    for(int iter = 0; iter < steps; iter++)
    {
        //create complex arry for cufft, y_dev_start is initial complex, temp_y_dev is the result of inverse fft, 
        //and y_dev_res is a real number version of the result
        CUDA_CHECK(cudaMemcpy(y_dev_start, y_hat, dimension * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));

        //create cufft plan
        //use Z2Z for complex double to complex double, use Z2Z because Z2D has no inverse option
        cufftPlan2d(&plan, size_x, size_y, CUFFT_Z2Z);
        cufftExecZ2Z(plan, y_dev_start, temp_y_dev, CUFFT_INVERSE);

        //change temp_y to y, which is a real number version of temp_y
        get_real<<<dimension, 1>>>(temp_y_dev, y_dev_res);

        //copy back y_device to y
        CUDA_CHECK(cudaMemcpy(y, y_dev_res, dimension * sizeof(double), cudaMemcpyDeviceToHost));
                
        
        //check if x_p is empty (0.0) or not
        int filled = 0;
        for(int i = 0; i < dimension; i++)
        {
            if(image_x_p[i] != 0.0)
            {
                filled = 1;
                break;
            }
        }

        //previous iterate
        if(filled == 0) copy(y, y+dimension, image_x_p);
        else copy(image_x, image_x+dimension, image_x_p);
       
        //updates for elements that satisfy object domain constraints
        if(mode.compare("output-output") == 0 || mode.compare("hybrid") == 0) copy(y, y+dimension, image_x);

        //updates for elements that violate object domain constraints--------------------------------------------------------------------
        CUDA_CHECK(cudaMemcpy(mask_dev, mask, dimension * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(y_device, y, dimension * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(image_x_device, image_x, dimension * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(image_x_p_device, image_x_p, dimension * sizeof(double), cudaMemcpyHostToDevice));

        int int_mode;
        if(mode.compare("hybrid") == 0 || mode.compare("input-output") == 0) int_mode = 1;
        else if(mode.compare("output-output") == 0) int_mode = 2;
        
        update_violated_elements<<<dimension, 1>>>(mask_dev, y_device, image_x_device, image_x_p_device, beta, int_mode);

        CUDA_CHECK(cudaMemcpy(image_x, image_x_device, dimension * sizeof(double), cudaMemcpyDeviceToHost));

        //fourier transform---------------------------------------------------------------------------------------------------------
        CUDA_CHECK(cudaMemcpy(image_x_dev, image_x, dimension * sizeof(double), cudaMemcpyHostToDevice));

        get_complex<<<dimension, 1>>>(image_x_dev, image_x_dev_comp);

        cufftPlan2d(&plan2, size_x, size_y, CUFFT_Z2Z);
        cufftExecZ2Z(plan2, image_x_dev_comp, image_x_dev_res, CUFFT_FORWARD);

        CUDA_CHECK(cudaMemcpy(x_hat, image_x_dev_res, dimension * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));

        for(int i = 0; i < dimension; i++) 
        {
            y_hat[i] = ptrMag[i]*exp(1i*arg(x_hat[i]));
        }
    }
    
    cufftDestroy(plan);
    cudaFree(y_dev_start);
    cudaFree(temp_y_dev);
    cudaFree(y_dev_res);
    cudaFree(mask_dev); 
    cudaFree(y_device); 
    cudaFree(image_x_device); 
    cudaFree(image_x_p_device);
    cufftDestroy(plan2);
    cudaFree(image_x_dev);
    cudaFree(image_x_dev_comp);
    cudaFree(image_x_dev_res);

    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);

    printf("Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);

    //free all arrays
    delete[] mask;
    delete[] image_x_p;
    delete[] y_hat;
    delete[] temp_y;
    delete[] y;

    py::array_t<double> image_x_2d =  py::array(dimension, image_x);
    image_x_2d.resize({X,Y});
    return image_x_2d;
}
