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
#if defined _MSC_VER
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

#define PI 3.1415926535897932384626433
#define CUDA_CHECK(call) {cudaError_t error = call; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}
using namespace std;
using namespace std::literals::complex_literals;
namespace py = pybind11;

void createDir(string dir);
string get_current_dir();
void print_image(string workdir, vector<double> image, int iter, int size_x, int size_y);
__global__ void copy_value(double *ptrMag, double *ptrRes, int dimension);
__global__ void copy_value_complex(cufftDoubleComplex *ptrMag, cufftDoubleComplex *ptrRes, int dimension);
__global__ void normalize_array(cufftDoubleComplex *ptrImg, cufftDoubleComplex *ptrRes, int dimension);
__global__ void random_phase_v2(double *random, cufftDoubleComplex *y_hat, cufftDoubleComplex *ptrMag, int dimension );

//Test whether array of double received by C++ code is the same as initial python value
py::array_t<double, py::array::c_style> array_check(py::array_t<double, py::array::c_style> img)
{
    py::buffer_info bufImg = img.request();
    size_t X = bufImg.shape[0]; //width of magnitude
    size_t Y = bufImg.shape[1]; //height of magnitude

    int size_x = static_cast<int>(X); 
    int size_y = static_cast<int>(Y);
    int dimension = size_x*size_y;

    double *ptrImg = static_cast<double*>(bufImg.ptr); //magnitude 1D

    py::array_t<double, py::array::c_style> result = py::array_t<double, py::array::c_style>(bufImg.size);
    py::buffer_info bufRes = result.request();
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
py::array_t<double, py::array::c_style> array_check_cuda(py::array_t<double, py::array::c_style> img)
{
    py::buffer_info bufImg = img.request();
    double *ptrImg = static_cast<double*>(bufImg.ptr); //magnitude 1D
    size_t X = bufImg.shape[0]; //width of magnitude
    size_t Y = bufImg.shape[1]; //height of magnitude

    int size_x = static_cast<int>(X); 
    int size_y = static_cast<int>(Y);
    int dimension = size_x * size_y;

    py::array_t<double, py::array::c_style> result = py::array_t<double, py::array::c_style>(bufImg.size);
    py::buffer_info bufRes = result.request();
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
py::array_t<complex<double>, py::array::c_style> array_check_complex(py::array_t<complex<double>, py::array::c_style> mag)
{
    py::buffer_info bufMag = mag.request();
    complex<double> *ptrMag =  static_cast<complex<double>*>(bufMag.ptr); //magnitude 1D
    size_t X = bufMag.shape[0]; //width of magnitude
    size_t Y = bufMag.shape[1]; //height of magnitude

    int size_x = static_cast<int>(X); 
    int size_y = static_cast<int>(Y);

    py::array_t<complex<double>, py::array::c_style> result = py::array_t<complex<double>, py::array::c_style>(bufMag.size);
    py::buffer_info bufRes = result.request();
    complex<double> *ptrRes = static_cast<complex<double>*>(bufRes.ptr);

    for(int i = 0; i < size_x * size_y; i++)
    {
        ptrRes[i] = ptrMag[i];
    }

    //send to python
    result.resize({X, Y});
    return result;
}

//Test whether array of complex received by CUDA code is the same as initial python value
py::array_t<complex<double>, py::array::c_style> array_check_complex_cuda(py::array_t<complex<double>, py::array::c_style> mag)
{
    py::buffer_info bufMag = mag.request();
    complex<double> *ptrMag = static_cast<complex<double>*>(bufMag.ptr); //magnitude 1D
    size_t X = bufMag.shape[0]; //width of magnitude
    size_t Y = bufMag.shape[1]; //height of magnitude

    int size_x = static_cast<int>(X); 
    int size_y = static_cast<int>(Y);
    int dimension = size_x * size_y;

    py::array_t<complex<double>, py::array::c_style> result = py::array_t<complex<double>, py::array::c_style>(bufMag.size);
    py::buffer_info bufRes = result.request();
    complex<double> *ptrRes = static_cast<complex<double>*>(bufRes.ptr);

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
py::array_t<complex<double>, py::array::c_style> cufft_inverse_forward(py::array_t<complex<double>, py::array::c_style> mag)
{
    py::buffer_info bufMag = mag.request();
    complex<double> *ptrMag = static_cast<complex<double>*>(bufMag.ptr); //magnitude 1D
    size_t X = bufMag.shape[0]; //width of magnitude
    size_t Y = bufMag.shape[1]; //height of magnitude

    int size_x = static_cast<int>(X); 
    int size_y = static_cast<int>(Y);
    int dimension = size_x * size_y;

    py::array_t<complex<double>, py::array::c_style> result = py::array_t<complex<double>, py::array::c_style>(bufMag.size);
    py::buffer_info bufRes = result.request();
    complex<double> *ptrRes = static_cast<complex<double>*>(bufRes.ptr);

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

//Phase retrieval using c++ random library
py::array_t<double, py::array::c_style> fienup_phase_retrieval_c_random(py::array_t<complex<double>, py::array::c_style> mag, py::array_t<double, py::array::c_style> masks, int steps, bool verbose, string mode, double beta)
{
    //asserting inputs
    assert(beta > 0);
    assert(steps > 0);
    assert(mode == "input-output" || mode == "output-output" || mode == "hybrid");

    py::buffer_info bufMag = mag.request();
    py::buffer_info bufMask = masks.request();

    int int_mode;
    if(mode.compare("hybrid") == 0) int_mode = 1;
    else if(mode.compare("input-output") == 0) int_mode = 2;
    else if(mode.compare("output-output") == 0) int_mode = 3;

    complex<double> *ptrMag = static_cast<complex<double>*>(bufMag.ptr); //magnitude 1D
    size_t X = bufMag.shape[0]; //width of magnitude
    size_t Y = bufMag.shape[1]; //height of magnitude
    double *mask = static_cast<double*>(bufMask.ptr); //mask array, same size as magnitude
    
    //alternative fot saving mag size, prevent warning while using CUFFT 
    //( warning C4267: 'argument': conversion from 'size_t' to 'int', possible loss of data)"
    //get int version of size instead of size_t, then create dimension (mag size)
    int size_x = static_cast<int>(X); 
    int size_y = static_cast<int>(Y);
    int dimension = size_x*size_y;

    //initialize arrays for GPU
    double *mask_dev, *image_x_device, *image_x_p_device, *random_value_dev;
    cufftDoubleComplex *y_hat_dev, *mag_dev , *image_x_dev_comp;
    CUDA_CHECK(cudaMalloc(&y_hat_dev, dimension * sizeof(cufftDoubleComplex))); //sample random phase
    CUDA_CHECK(cudaMalloc(&mag_dev, dimension * sizeof(cufftDoubleComplex))); //device magnitudes
    CUDA_CHECK(cudaMalloc(&mask_dev, dimension * sizeof(double))); //device mask
    CUDA_CHECK(cudaMalloc(&image_x_device, dimension * sizeof(double))); //image x in device
    CUDA_CHECK(cudaMalloc(&image_x_p_device, dimension * sizeof(double))); //image x_p in device
    CUDA_CHECK(cudaMalloc(&random_value_dev, dimension * sizeof(double))); //array of random using c++ random library
    CUDA_CHECK(cudaMalloc(&image_x_dev_comp, dimension * sizeof(cufftDoubleComplex))); //complex version if image x

    //allocating inital values to device
    cudaMemset(image_x_device,  0, dimension * sizeof(double));
    cudaMemset(image_x_p_device, 0, dimension * sizeof(double));

    int devId, numSMs;
    cudaGetDevice(&devId);
    cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, devId);

    //array of random using c++ random library
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 1);
    
    vector<double> random_value;
    random_value.resize(dimension);
    generate(random_value.begin(), random_value.end(), [&] { return dis(gen);});

    //copy input magnitudes and mask to gpu
    CUDA_CHECK(cudaMemcpy(mag_dev, ptrMag, dimension * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mask_dev, mask, dimension * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(random_value_dev, random_value.data(), dimension * sizeof(double), cudaMemcpyHostToDevice));

    random_phase_v2<<<8*numSMs, 256>>>(random_value_dev, y_hat_dev, mag_dev, dimension);

    //print image result every iteration if verbose is True, this will create result_c_rand folder automatically and put the images there
    string workdir = get_current_dir() + "/result_c_rand";
    vector<double> image;
    image.resize(dimension);
    if(verbose == true) createDir(workdir);

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

        if(verbose == true)
        {
            CUDA_CHECK(cudaMemcpy(image.data(), image_x_device, dimension * sizeof(double), cudaMemcpyDeviceToHost));
            print_image(workdir, image, iter, size_x, size_y); 
        }
        
    }

    py::array_t<double, py::array::c_style> result = py::array_t<double, py::array::c_style>(bufMag.size);
    py::buffer_info bufRes = result.request();
    double *ptrRes = static_cast<double*>(bufRes.ptr);
    CUDA_CHECK(cudaMemcpy(ptrRes, image_x_device, dimension * sizeof(double), cudaMemcpyDeviceToHost));

    //free CUDA usages
    cudaFree(random_value_dev);
    cudaFree(y_hat_dev);
    cudaFree(mag_dev);
    cudaFree(mask_dev);
    cudaFree(image_x_device); 
    cudaFree(image_x_p_device);
    cudaFree(image_x_dev_comp);

    result.resize({X, Y});
    return result;
}

//Phase retrieval using numpy random
py::array_t<double, py::array::c_style> fienup_phase_retrieval_numpy_random(py::array_t<complex<double>, py::array::c_style> mag, py::array_t<double, py::array::c_style> masks, py::array_t<double, py::array::c_style> randoms, int steps, bool verbose, string mode, double beta)
{
    //asserting inputs
    assert(beta > 0);
    assert(steps > 0);
    assert(mode == "input-output" || mode == "output-output" || mode == "hybrid");


    py::buffer_info bufMag = mag.request();
    py::buffer_info bufMask = masks.request();
    py::buffer_info bufRand = randoms.request();

    int int_mode;
    if(mode.compare("hybrid") == 0) int_mode = 1;
    else if(mode.compare("input-output") == 0) int_mode = 2;
    else if(mode.compare("output-output") == 0) int_mode = 3;

    complex<double> *ptrMag = static_cast<complex<double>*>(bufMag.ptr); //magnitude 1D
    size_t X = bufMag.shape[0]; //width of magnitude
    size_t Y = bufMag.shape[1]; //height of magnitude
    double *mask = static_cast<double*>(bufMask.ptr); //mask array, same size as magnitude
    double *random_value = static_cast<double*>(bufRand.ptr); //numpy random
    
    //alternative fot saving mag size, prevent warning while using CUFFT 
    //( warning C4267: 'argument': conversion from 'size_t' to 'int', possible loss of data)"
    //get int version of size instead of size_t, then create dimension (mag size)
    int size_x = static_cast<int>(X); 
    int size_y = static_cast<int>(Y);
    int dimension = size_x*size_y;

    //initialize arrays for GPU
    double *mask_dev, *image_x_device, *image_x_p_device, *random_value_dev;
    cufftDoubleComplex *y_hat_dev, *mag_dev , *image_x_dev_comp;
    CUDA_CHECK(cudaMalloc(&y_hat_dev, dimension * sizeof(cufftDoubleComplex))); //sample random phase
    CUDA_CHECK(cudaMalloc(&mag_dev, dimension * sizeof(cufftDoubleComplex))); //device magnitudes
    CUDA_CHECK(cudaMalloc(&mask_dev, dimension * sizeof(double))); //device mask
    CUDA_CHECK(cudaMalloc(&image_x_device, dimension * sizeof(double))); //image x in device
    CUDA_CHECK(cudaMalloc(&image_x_p_device, dimension * sizeof(double))); //image x_p in device
    CUDA_CHECK(cudaMalloc(&random_value_dev, dimension * sizeof(double))); //array of random using c++ random library
    CUDA_CHECK(cudaMalloc(&image_x_dev_comp, dimension * sizeof(cufftDoubleComplex))); //complex version if image x

    //allocating inital values to device
    cudaMemset(image_x_device,  0, dimension * sizeof(double));
    cudaMemset(image_x_p_device, 0, dimension * sizeof(double));

    int devId, numSMs;
    cudaGetDevice(&devId);
    cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, devId);
   
    //copy input magnitudes and mask to gpu
    CUDA_CHECK(cudaMemcpy(mag_dev, ptrMag, dimension * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mask_dev, mask, dimension * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(random_value_dev, random_value, dimension * sizeof(double), cudaMemcpyHostToDevice));

    random_phase_v2<<<8*numSMs, 256>>>(random_value_dev, y_hat_dev, mag_dev, dimension);

    //print image result every iteration if verbose is True, this will create result_numpy_rand folder automatically and put the images there
    string workdir = get_current_dir() + "/result_numpy_rand";
    vector<double> image;
    image.resize(dimension);
    if(verbose == true) createDir(workdir);

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

        if(verbose == true)
        {
            CUDA_CHECK(cudaMemcpy(image.data(), image_x_device, dimension * sizeof(double), cudaMemcpyDeviceToHost));
            print_image(workdir, image, iter, size_x, size_y);
        }

    }

    py::array_t<double, py::array::c_style> result = py::array_t<double, py::array::c_style>(bufMag.size);
    py::buffer_info bufRes = result.request();
    double *ptrRes = static_cast<double*>(bufRes.ptr);
    CUDA_CHECK(cudaMemcpy(ptrRes, image_x_device, dimension * sizeof(double), cudaMemcpyDeviceToHost));

    //free CUDA usages
    cudaFree(random_value_dev);
    cudaFree(y_hat_dev);
    cudaFree(mag_dev);
    cudaFree(mask_dev);
    cudaFree(image_x_device); 
    cudaFree(image_x_p_device);
    cudaFree(image_x_dev_comp);

    result.resize({X, Y});
    return result;
}

//Create a folder
void createDir(string dir)
{
    #if defined _MSC_VER
        _mkdir(dir.data());
    #else
        mkdir(dir.data(), 0777);
    #endif
}

//Get current location (same as the executable file)
string get_current_dir() 
{
   char buff[FILENAME_MAX]; //create string buffer to hold path
   GetCurrentDir( buff, FILENAME_MAX );
   string current_working_dir(buff);
   return current_working_dir;
}

//Create grayscale image file in a folder
void print_image(string workdir, vector<double> image, int iter, int size_x, int size_y)
{
    string fulldir = workdir + "/image_" + to_string(iter+1) + ".png";
    const char *filename = fulldir.c_str();
    cv::Mat img = cv::Mat(size_x, size_y, CV_64FC1, image.data()).clone();
    cv::imwrite(filename, img);
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

//Random phase using array of random as random value
__global__ void random_phase_v2(double *random, cufftDoubleComplex *y_hat, cufftDoubleComplex *ptrMag, int dimension ) 
{
    cufftDoubleComplex complex1i, exp_target;
    complex1i.x = 0; complex1i.y = 1;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dimension; idx += blockDim.x * gridDim.x)  
    {
        exp_target.x = PI*2.0*random[idx];
        exp_target.y = 0;
        y_hat[idx] = cuCmul(ptrMag[idx], gpu_exp(cuCmul(complex1i, exp_target)));
    }
}