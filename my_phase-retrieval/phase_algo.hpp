#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cufft.h>
#include <cmath>
#include <cstdio>
#include <time.h>
#include <iostream>
#include <complex>
using namespace std;

namespace py = pybind11;

// py::array_t<double> fienup_phase_retrieval(py::array_t<double> mag) {
//    /* read input arrays buffer_info */
//    py::buffer_info bufMag = mag.request();

//    /* allocate the output buffer */
//    py::array_t<double> result = py::array_t<double>(bufMag.size);
//    py::buffer_info bufres = result.request();
//    double *ptr1 = (double *) bufMag.ptr, *ptrres = (double *)bufres.ptr;
//    size_t X = bufMag.shape[0];
//    size_t Y = bufMag.shape[1];

//    /* Add both arrays */
//    for (size_t idx = 0; idx < X; idx++)
//        for (size_t idy = 0; idy < Y; idy++)
//        {
//            ptrres[idx*Y + idy] = ptr1[idx*Y+ idy];
//        }

//    /* Reshape result to have same shape as input */
//    result.resize({X,Y});

//    return result;
// }

void fienup_phase_retrieval(py::array_t<double> mag, int steps, bool verbose) 
{
    srand( (unsigned)time( NULL ) );
    py::buffer_info bufMag = mag.request();

     int k=0;
     int iter;

    double *ptr1 = (double *) bufMag.ptr; //magnitude 1D
    size_t X = bufMag.shape[0]; //width of magnitude
    size_t Y = bufMag.shape[1]; //heght of magnitude

    double **ptrMag = NULL; //magnitude 2D array
    double **mask = NULL; //mask 2D array, same size as magnitude
    double **image_x = NULL; //initial image x, same size as magnitude
    double **image_x_p = NULL; //previous image for steps
    double **y = NULL; //store inverse fourier transform (real number)
    complex<double> **y_hat;
    
    //allocating first dimension of arrays
    ptrMag = (double**)malloc(X * sizeof(double)); //convert 1D mag to 2D (1)
    mask = (double**)malloc(X * sizeof(double)); //mask with same size as magnitude
    image_x = (double**)malloc(X * sizeof(double)); //image x with same size as magnitude
    image_x_p = (double**)malloc(X * sizeof(double)); //previous image for steps
    y = (double**)malloc(X * sizeof(double)); //store inverse fourier transform (real number)
    y_hat = (complex<double>**)malloc(X * sizeof(complex<double>)); //sample random phase

    //allocating second dimension of arrays
    for (size_t i = 0; i < X; i++) 
    {
        ptrMag[i] = (double*)malloc(Y * sizeof(double)); //convert 1D mag to 2D (2)
        mask[i] = (double*)malloc(Y * sizeof(double)); //allocate mask
        image_x[i] = (double*)malloc(Y * sizeof(double)); //allocate image_x
        image_x_p[i] = (double*)malloc(Y * sizeof(double)); //allocate image_x_p for previous image
        y[i] = (double*)malloc(Y * sizeof(double)); //store inverse fourier transform (real number)
        y_hat[i] = (complex<double>*)malloc(Y * sizeof(complex<double>));//sample random phase
    }

    //allocating inital values to arrays
    for (size_t idx = 0; idx < X; idx++)
    {
        for (size_t idy = 0; idy < Y; idy++)
        {
            double rand_num = (double) rand()/RAND_MAX;
            mask[idx][idy] = 1.0; //fill mask with 1.0
            image_x[idx][idy] = 0.0; //fill image_x with 0.0 for initial data
            image_x_p[idx][idy] = 0.0; //fill image_x_p with 0.0 for initial data
            y[idx][idy] = 0.0; //store inverse fourier transform (real number)
            ptrMag[idx][idy] = ptr1[k]; //convert 1D mag to 2D (3)
            y_hat[idx][idy] = ptr1[k]*exp(1i*2.0*3.14*rand_num); //random phase initial value
            k++;
            //cout<<y_hat[idx][idy]<<"\t";

        }
        //printf("\n");
    }

    for(iter = 0; iter < steps; iter++)
    {

    }

}
