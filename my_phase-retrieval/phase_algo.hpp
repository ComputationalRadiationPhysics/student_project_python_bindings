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

    //alternative fot saving mag size, prevent warning while using CUFFT 
    //( warning C4267: 'argument': conversion from 'size_t' to 'int', possible loss of data)"
    int size_x = 0; 
    int size_y = 0;

    double *ptr1 = (double *) bufMag.ptr; //magnitude 1D
    size_t X = bufMag.shape[0]; //width of magnitude
    size_t Y = bufMag.shape[1]; //heght of magnitude
    
    //get int version of size instead of size_t
    for (size_t idx = 0; idx < X; idx++) size_x++;
    for (size_t idy = 0; idy < Y; idy++) size_y++;


    double **ptrMag = NULL; //magnitude 2D array
    double **mask = NULL; //mask 2D array, same size as magnitude
    double **image_x = NULL; //initial image x, same size as magnitude
    double **image_x_p = NULL; //previous image for steps
    double **y = NULL; //store inverse fourier transform (real number)
    complex<double> **y_hat; //sample random phase
    complex<double> **temp_y; //temporary complex version of y after CUFFT, before copying the real number to y
    
    //allocating first dimension of arrays
    ptrMag = (double**)malloc(size_x * sizeof(double)); //convert 1D mag to 2D (1)
    mask = (double**)malloc(size_x * sizeof(double)); //mask with same size as magnitude
    image_x = (double**)malloc(size_x * sizeof(double)); //image x with same size as magnitude
    image_x_p = (double**)malloc(size_x * sizeof(double)); //previous image for steps
    y = (double**)malloc(size_x * sizeof(double)); //store inverse fourier transform (real number)
    y_hat = (complex<double>**)malloc(size_x * sizeof(complex<double>)); //sample random phase
    temp_y = (complex<double>**)malloc(size_x * sizeof(complex<double>));

    //allocating second dimension of arrays
    for (size_t i = 0; i < size_x; i++) 
    {
        ptrMag[i] = (double*)malloc(size_y * sizeof(double)); //convert 1D mag to 2D (2)
        mask[i] = (double*)malloc(size_y * sizeof(double)); //allocate mask
        image_x[i] = (double*)malloc(size_y * sizeof(double)); //allocate image_x
        image_x_p[i] = (double*)malloc(size_y * sizeof(double)); //allocate image_x_p for previous image
        y[i] = (double*)malloc(size_y * sizeof(double)); //store inverse fourier transform (real number)
        y_hat[i] = (complex<double>*)malloc(size_y * sizeof(complex<double>));//sample random phase
        temp_y[i] = (complex<double>*)malloc(size_y * sizeof(complex<double>));
    }

    //allocating inital values to arrays
    for (int idx = 0; idx < size_x; idx++)
    {
        for (int idy = 0; idy < size_y; idy++)
        {
            double rand_num = (double) rand()/RAND_MAX;
            mask[idx][idy] = 1.0; //fill mask with 1.0
            image_x[idx][idy] = 0.0; //fill image_x with 0.0 for initial data
            image_x_p[idx][idy] = 0.0; //fill image_x_p with 0.0 for initial data
            y[idx][idy] = 0.0; //store inverse fourier transform (real number)
            ptrMag[idx][idy] = ptr1[k]; //convert 1D mag to 2D (3)
            y_hat[idx][idy] = ptr1[k]*exp(1i*2.0*3.14*rand_num); //random phase initial value
            k++;
            // cout<<y_hat[idx][idy]<<"\t";
        }
        // cout<<endl;
    }
    

    // for(iter = 0; iter < steps; iter++)
    
    cufftDoubleComplex *y_dev, *y_devr;
    for(iter = 0; iter < 1; iter++)
    {
        CUDA_CHECK(cudaMalloc((void **) &y_dev, size_x * size_y * sizeof(cufftDoubleComplex)));
        CUDA_CHECK(cudaMalloc((void **) &y_devr, size_x * size_y * sizeof(cufftDoubleComplex)));
        CUDA_CHECK(cudaMemcpy(y_dev, y_hat, size_x * size_y * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));

        cufftHandle plan;
        cufftPlan2d(&plan, size_x, size_y, CUFFT_Z2Z);
        cufftExecZ2Z(plan, y_dev, y_devr, CUFFT_INVERSE);

        CUDA_CHECK(cudaMemcpy(temp_y, y_devr, size_x * size_y * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));
        

        for (int idx = 0; idx < size_x; idx++)
        {
            for (int idy = 0; idy < size_y; idy++)
            {
                // y[idx][idy] = real(temp_y[idx][idy]);
                cout<<temp_y[idx][idy]<<"\t";
            }
            //cout<<endl;
        }

        CUDA_CHECK(cudaFree(y_dev));
        cufftDestroy(plan);
    }

}
