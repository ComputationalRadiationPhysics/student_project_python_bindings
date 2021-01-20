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

void fienup_phase_retrieval(py::array_t<double> mag, int steps, bool verbose, string mode, double beta) 
{

    assert(beta > 0);
    assert(steps > 0);
    assert(mode == "input-output" || mode == "output-output" || mode == "hybrid");

    srand( (unsigned)time( NULL ) );
    py::buffer_info bufMag = mag.request();

    double *ptrMag = (double *) bufMag.ptr; //magnitude 1D
    size_t X = bufMag.shape[0]; //width of magnitude
    size_t Y = bufMag.shape[1]; //heght of magnitude
    
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

    //indices and logicals
    bool *logical_not_mask = new bool[dimension];
    bool *y_less_than_zero = new bool[dimension];
    bool *logical_and = new bool[dimension];
    bool *indices = new bool[dimension]; //logical or

    auto begin = chrono::high_resolution_clock::now();

    //allocating inital values to arrays
    for (int i = 0; i < dimension; i++)
    {
        double rand_num = (double) rand()/RAND_MAX;
        mask[i] = 1.0; //fill mask with 1.0
        image_x[i] = 0.0; //fill image_x with 0.0 for initial data
        image_x_p[i] = 0.0; //fill image_x_p with 0.0 for initial data
        y[i] = 0.0; //store inverse fourier transform (real number)
        y_hat[i] = ptrMag[i]*exp(1i*2.0*3.14*rand_num); //random phase initial value
    }

    //iteration with number of steps
    for(int iter = 0; iter < steps; iter++)
    {
        //create complex arry for cufft, y_dev is initial complex, y_dev_res, is the result of inverse fft
        cufftDoubleComplex *y_dev, *y_dev_res;
        CUDA_CHECK(cudaMalloc((void **) &y_dev, dimension * sizeof(cufftDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **) &y_dev_res, dimension * sizeof(cufftDoubleComplex)));

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
        for(int i = 0; i < dimension; i++) y[i] = temp_y[i].real();
                
        cufftDestroy(plan);
        cudaFree(y_dev);
        cudaFree(y_dev_res);

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

        //get indices and logicals, and updates for elements
        for(int i = 0; i < dimension; i++)
        {
            //find elements that violate object domain constraints 
            //or are not masked

            //1. logical not of mask
            if(mask[i] <= 0) logical_not_mask[i] = true;
            else if(mask[i] >= 1) logical_not_mask[i] = false;

            //2. check if any element y is less than zero
            if(y[i] < 0) y_less_than_zero[i] = true;
            else if(y[i] >= 0) y_less_than_zero[i] = false;

            //use "and" logical to check the "less than zero y" and the mask  
            if(y_less_than_zero[i] == true && mask[i] >= 1) logical_and[i] = true;
            else logical_and[i] = false;

            //create indices with logical "not"
            if(logical_and[i] == false && logical_not_mask[i] == false) indices[i] = false;
            else indices[i] = true;

            //updates for elements that violate object domain constraints
            if(indices[i] == true)
            {
                if(mode.compare("hybrid") == 0 || mode.compare("input-output") == 0)
                {
                    image_x[i] = image_x_p[i]-beta*y[i];
                }
                if(mode.compare("output-output") == 0)
                {
                    image_x[i] = y[i]-beta*y[i];
                }
            }
        }

        //fourier transform
        cufftDoubleReal *image_x_dev;
        cufftDoubleComplex *image_x_dev_res;
        CUDA_CHECK(cudaMalloc((void**) &image_x_dev, dimension * sizeof(cufftDoubleReal)));
        CUDA_CHECK(cudaMalloc((void **) &image_x_dev_res, dimension * sizeof(cufftDoubleComplex)));

        CUDA_CHECK(cudaMemcpy(image_x_dev, image_x, dimension * sizeof(cufftDoubleReal), cudaMemcpyHostToDevice));

        //create cufft plan
        cufftHandle plan2;
        cufftPlan2d(&plan2, size_x, size_y, CUFFT_D2Z);
        cufftExecD2Z(plan2, image_x_dev, image_x_dev_res);

        CUDA_CHECK(cudaMemcpy(x_hat, image_x_dev_res, dimension * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));

        cufftDestroy(plan2);
        cudaFree(image_x_dev);
        cudaFree(image_x_dev_res);

        // for(int i = 0; i < dimension; i++) 
        // {
        //     cout<<i<<"\t"<<image_x[i]<<"\t"<<x_hat[i]<<endl;
        //     if(x_hat[i].real() == 0) break;
        // }

        for(int i = 0; i < dimension; i++) 
        {
            y_hat[i] = ptrMag[i]*exp(1i*arg(x_hat[i]));
        }
    }

    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(end - begin);

    printf("Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);
    // for(int i = 0; i < dimension; i++) 
    // {
    //     cout<<y_hat[i]<<endl;
    // }

    //free all arrays
    delete[] mask;
    delete[] image_x;
    delete[] image_x_p;
    delete[] y_hat;
    delete[] temp_y;
    delete[] y;
    delete[] logical_not_mask;
    delete[] y_less_than_zero;
    delete[] logical_and;
    delete[] indices;
}
