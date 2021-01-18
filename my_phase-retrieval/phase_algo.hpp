#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cufft.h>
#include <cmath>
#include <cstdio>
#include <time.h>
#include <iostream>
#include <complex>
#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}
using namespace std;

namespace py = pybind11;

void fienup_phase_retrieval(py::array_t<double> mag, int steps, bool verbose) 
{
    srand( (unsigned)time( NULL ) );
    py::buffer_info bufMag = mag.request();

    int k=0; 
    int iter;

    double *ptr1 = (double *) bufMag.ptr; //magnitude 1D
    size_t X = bufMag.shape[0]; //width of magnitude
    size_t Y = bufMag.shape[1]; //heght of magnitude
    
    //alternative fot saving mag size, prevent warning while using CUFFT 
    //( warning C4267: 'argument': conversion from 'size_t' to 'int', possible loss of data)"
    //get int version of size instead of size_t, then create dimension (mag size)
    int size_x = static_cast<int>(X); 
    int size_y = static_cast<int>(Y);
    int dimension = size_x*size_y;

    //allocating arrays, all arrays bellow are 1D representation of 2D array
    double *ptrMag = new double[dimension]; //magnitude array
    double *mask = new double[dimension]; //mask array, same size as magnitude
    double *image_x = new double[dimension]; //initial image x, same size as magnitude
    double *image_x_p = new double[dimension]; //previous image for steps
    double *y = new double[dimension]; //store inverse fourier transform (real number)
    complex<double> *y_hat = new complex<double>[dimension]; //sample random phase
    complex<double> *temp_y =  new complex<double>[dimension]; //temporary complex version of y after CUFFT, before copying the real number to y

    //allocating inital values to arrays
    for (int idx = 0; idx < size_x; idx++)
    {
        for (int idy = 0; idy < size_y; idy++)
        {
            double rand_num = (double) rand()/RAND_MAX;
            mask[idx * size_y + idy] = 1.0; //fill mask with 1.0
            image_x[idx * size_y + idy] = 0.0; //fill image_x with 0.0 for initial data
            image_x_p[idx * size_y + idy] = 0.0; //fill image_x_p with 0.0 for initial data
            y[idx * size_y + idy] = 0.0; //store inverse fourier transform (real number)
            ptrMag[idx * size_y + idy] = ptr1[k]; //convert 2D mag to 1D (3)
            y_hat[idx * size_y + idy] = ptr1[k]*exp(1i*2.0*3.14*rand_num); //random phase initial value
            k++;
            //cout<<y_hat[idx * size_y + idy]<<"\t";
        }
        //cout<<endl;
    }
    
    //create complex arry for cufft, y_dev is initial complex, y_dev_res, is the result of inverse fft
    cufftDoubleComplex *y_dev, *y_dev_res;
    CUDA_CHECK(cudaMalloc((void **) &y_dev, dimension * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc((void **) &y_dev_res, dimension * sizeof(cufftDoubleComplex)));

    //iteration with number of steps
    for(iter = 0; iter < 1 /*steps*/; iter++)
    {
        //copy host complex array "y_hat" to device complex array "y_dev"
        CUDA_CHECK(cudaMemcpy(y_dev, y_hat, dimension * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));

        //create cufft plan
        cufftHandle plan;
        //use Z2Z for complex double to complex double, use Z2Z because Z2D has no inverse option
        cufftPlan2d(&plan, size_x, size_y, CUFFT_Z2Z);
        cufftExecZ2Z(plan, y_dev, y_dev_res, CUFFT_INVERSE);

        //copy device complex array "y_dev" to host complex array "y_hat" 
        CUDA_CHECK(cudaMemcpy(temp_y, y_dev_res, dimension * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));

        //change temp_y to y, which is a real number version of temp_y
        for (int idx = 0; idx < size_x; idx++)
        { 
            for (int idy = 0; idy < size_y; idy++)
            {
                y[idx * size_y + idy] = temp_y[idx * size_y + idy].real();
                //cout<<y[idx * size_y + idy]<<"\t";
            }
            //cout<<endl;   
        }
        cufftDestroy(plan);
    }

    //free all arrays
    cudaFree(y_dev);
    cudaFree(y_dev_res);
    free(ptrMag);
    free(mask);
    free(image_x);
    free(image_x_p);
    free(y_hat);
    free(temp_y);
    free(y);
}
