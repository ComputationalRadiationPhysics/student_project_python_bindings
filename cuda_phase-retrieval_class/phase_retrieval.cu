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

/**
 * \brief CUDA Phase Retrieval primary function with random array as an additional parameter
 * \param image Input image
 * \param masks Input mask (default 1)
 * \param steps Number of iteration
 * \param mode Input mode ("input-ouput", "output-output", or "hybrid")
 * \param beta Input beta (default 0.8)
 * \param randoms Input array of random
 * \return Image result
 */

class Phase_Algo
{
  private:
    //CPU variables
    py::buffer_info bufImg, bufMask, bufRand, bufRes;
    size_t X, Y, 
           mask_X, mask_Y, 
           rand_X, rand_Y;    
    int mode, size_x, size_y, dimension, devId, numSMs;;
    double beta;
    double *ptrImg, *mask, *random_value, *ptrRes;
    py::array_t<double, py::array::c_style> result;

    //GPU Arrays
    double *src_img_dev,        //Source image in GPU 
           *mag_dev,            //Magnitudes in GPU
           *mask_dev,           //Mask in GPU
           *image_x_device,     //Image output in GPU 
           *image_x_p_device,   //Save previous image output in GPU for iteration
           *random_value_dev;   //Array of random in GPU

    cufftDoubleComplex *y_hat_dev,          //Sample random phase in GPU
                       *y_hat_dev_init,     //Store initial GPU random phase before iterations
                       *src_img_dev_comp,   //complex number version of source image in GPU
                       *image_x_dev_comp;   //Complex number version of image output in GPU

  public:
    Phase_Algo(py::array_t<double, py::array::c_style> image, py::array_t<double, py::array::c_style> masks, string mode, double beta, py::array_t<double, py::array::c_style> randoms)
    {
      /**
    * \b Process
    * 1. Asserting inputs
    */
      assert(beta > 0);
      assert(mode == "input-output" || mode == "output-output" || mode == "hybrid");

      set_mode(mode); //use integer instead of string for mode 
      bufImg = image.request();       
      bufMask = masks.request();      
      bufRand = randoms.request();
      X = bufImg.shape[0];               //Width of image
      Y = bufImg.shape[1]; 
      this->beta = beta;

      /**
      * 2. Asserting array size, make sure all arrays are using the same size
      */
      mask_X = bufMask.shape[0];               //Width of mask
      mask_Y = bufMask.shape[1];               //Height of mask
      rand_X = bufRand.shape[0];               //Width of random array
      rand_Y = bufRand.shape[1];               //Height of random array
      assert(mask_X == X && rand_X == X);
      assert(mask_Y == Y && rand_Y == Y);

      ptrImg = static_cast<double*>(bufImg.ptr);          //Get 1D image array
      mask = static_cast<double*>(bufMask.ptr);           //Mask array, same size as image 
      random_value = static_cast<double*>(bufRand.ptr);   //Array of uniform random number, same size as image

      size_x = static_cast<int>(X); //Convert X to integer to prevent getting warning from CUFFT
      size_y = static_cast<int>(Y); //Convert Y to integer to prevent getting warning from CUFFT
      dimension = size_x*size_y;    //Area or dimension of image, mask, and array of random

      /**
      * 3. Allocating memories in GPU
      */                   
      CUDA_CHECK(cudaMalloc(&y_hat_dev, dimension * sizeof(cufftDoubleComplex)));
      CUDA_CHECK(cudaMalloc(&y_hat_dev_init, dimension * sizeof(cufftDoubleComplex)));
      CUDA_CHECK(cudaMalloc(&src_img_dev_comp, dimension * sizeof(cufftDoubleComplex)));
      CUDA_CHECK(cudaMalloc(&mag_dev, dimension * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&src_img_dev, dimension * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&mask_dev, dimension * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&image_x_device, dimension * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&image_x_p_device, dimension * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&image_x_dev_comp, dimension * sizeof(cufftDoubleComplex)));
      CUDA_CHECK(cudaMalloc(&random_value_dev, dimension * sizeof(double)));

      /**
      * 4. Allocating inital values of output image to 0 in GPU
      */
      cudaMemset(image_x_device,  0, dimension * sizeof(double));
      cudaMemset(image_x_p_device, 0, dimension * sizeof(double));

      /**
      * 5. Set number of SM for CUDA Kernel
      */
      cudaGetDevice(&devId);
      cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, devId);

      /**
      * 6. Get magntitudes
      */   
      CUDA_CHECK(cudaMemcpy(src_img_dev, ptrImg, dimension * sizeof(double), cudaMemcpyHostToDevice));

      //Convert the source image array into array of complex number
      get_complex_array<<<8*numSMs, 256>>>(src_img_dev, src_img_dev_comp, dimension);

      //After that, do CUFFT first time to the complex source image,
      cufftHandle plan;
      CUFFT_CHECK(cufftPlan2d(&plan, size_x, size_y, CUFFT_Z2Z));
      CUFFT_CHECK(cufftExecZ2Z(plan, src_img_dev_comp, src_img_dev_comp, CUFFT_FORWARD));
      cufftDestroy(plan);

      //Then get the absolute value of the result, the absolute result is called magnitude
      get_absolute_array<<<8*numSMs, 256>>>(src_img_dev_comp, mag_dev, dimension);
      
      /**
      * 7. Copy mask and array of random to GPU
      */
      CUDA_CHECK(cudaMemcpy(mask_dev, mask, dimension * sizeof(double), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(random_value_dev, random_value, dimension * sizeof(double), cudaMemcpyHostToDevice));

      /**
      * 8. Get initial random phase, and create a copy for reset
      */
      random_phase<<<8*numSMs, 256>>>(random_value_dev, y_hat_dev, mag_dev, dimension);
      CUDA_CHECK(cudaMemcpy(y_hat_dev_init, y_hat_dev, dimension * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice));
      }

    void iterate_random_phase(int steps)
    {
       /**
      * For every iteration :\n
      * a. Create 2D CUFFT plan using complex double to complex double\n
      * b. Do CUFFT Inverse to the random phase array\n
      * c. Processing the arrays. This will generate 2 version of result, complex version and real version\n
      * d. The real version of result image array will be used as output after the final iteration\n
      * e. Do normal FFT to the complex version of result image array\n
      * f. Combine the FFT'ed result with the random phase array\n
      * g. The combined array is used as random phase array for the next iterarion\n
      */
      for(int iter = 0; iter < steps; iter++)
      {   
        cufftHandle plan; //create cufft plan
        CUFFT_CHECK(cufftPlan2d(&plan, size_x, size_y, CUFFT_Z2Z));
        CUFFT_CHECK(cufftExecZ2Z(plan, y_hat_dev, y_hat_dev, CUFFT_INVERSE));
        process_arrays<<<8*numSMs, 256>>>(mask_dev, y_hat_dev, image_x_device, image_x_p_device, image_x_dev_comp, beta, mode, iter, dimension);
        CUFFT_CHECK(cufftExecZ2Z(plan, image_x_dev_comp, image_x_dev_comp, CUFFT_FORWARD));
        satisfy_fourier<<<8*numSMs, 256>>>(y_hat_dev, image_x_dev_comp, mag_dev, dimension);
        cufftDestroy(plan);
      }
    }

    void reset_random_phase()
    {
      CUDA_CHECK(cudaMemcpy(y_hat_dev, y_hat_dev_init, dimension * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice));
    }

    py::array_t<double, py::array::c_style> get_result()
    {
      /**
      * 1. Create a pybind array to store the final image result
      */
      result = py::array_t<double, py::array::c_style>(bufImg.size);
      bufRes = result.request();
      ptrRes = static_cast<double*>(bufRes.ptr);
      CUDA_CHECK(cudaMemcpy(ptrRes, image_x_device, dimension * sizeof(double), cudaMemcpyDeviceToHost));

      /**
      * 2. Return the final result image
      */
      result.resize({X, Y});
      return result;
    }

    void set_mode(string mode)
    {
      if(mode.compare("hybrid") == 0) this->mode = 1;
      else if(mode.compare("input-output") == 0) this->mode = 2;
      else if(mode.compare("output-output") == 0) this->mode = 3;
    }

    int get_mode()
    {
      return mode;
    }

    double get_beta()
    {
      return beta;
    }

    ~Phase_Algo()
    {
      cudaFree(y_hat_dev);
      cudaFree(y_hat_dev_init);
      cudaFree(src_img_dev_comp);
      cudaFree(mag_dev);
      cudaFree(src_img_dev);
      cudaFree(mask_dev);
      cudaFree(image_x_device); 
      cudaFree(image_x_p_device);
      cudaFree(image_x_dev_comp);
      cudaFree(random_value_dev);
    }
};

PYBIND11_MODULE(cuPhaseRet, m) 
{
  //main phase retrieval
  // m.def("fienup_phase_retrieval", py::overload_cast<py::array_t<double, py::array::c_style>, py::array_t<double, py::array::c_style>, int, string, double, py::array_t<double, py::array::c_style>>(&fienup_phase_retrieval));
  // m.def("fienup_phase_retrieval", py::overload_cast<py::array_t<double, py::array::c_style>, py::array_t<double, py::array::c_style>, int, string, double>(&fienup_phase_retrieval));

  py::class_<Phase_Algo>(m, "Phase_Algo", py::module_local())
      .def(py::init<py::array_t<double, py::array::c_style>, py::array_t<double, py::array::c_style>, string, double, py::array_t<double, py::array::c_style>>())
	    .def("get_mode", &Phase_Algo::get_mode)
      .def("get_beta", &Phase_Algo::get_beta)
      .def("iterate_random_phase", &Phase_Algo::iterate_random_phase)
      .def("reset_random_phase", &Phase_Algo::reset_random_phase)
      .def("get_result", &Phase_Algo::get_result);
}

/**
 * \brief Calculate exponential of a double complex in GPU
 * \param arg A single double complex number, implemented using CUFFT library
 * \return Exponential value of the complex number
 */
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
 
 /**
  * \brief Normalize every result elements of after doing CUFFT_INVERSE
  * \param comp_data A single double complex number, implemented using CUFFT library
  * \param dimension Size of all arrays
  * \return normalized complex number
  */
 __device__ cufftDoubleComplex normalize(cufftDoubleComplex comp_data, int dimension)
 {
     cufftDoubleComplex norm_data;
     norm_data.x = comp_data.x / static_cast<double>(dimension);
     norm_data.y = comp_data.y / static_cast<double>(dimension);
     
     return norm_data;
 }
 
 /**
  * \brief Convert real number to CUFFT complex number, using 0 as imaginary part
  * \param real_data A single double value
  * \return Complex version of the double value, implemented using CUFFT
  */
 __device__ cufftDoubleComplex get_complex(double real_data)
 {
     cufftDoubleComplex comp_data;
     comp_data.x = real_data;
     comp_data.y = 0;
 
     return comp_data;
 }
 
 /**
  * \brief Get real number part of a CUFFT complex number
  * \param comp_data A single double complex number, implemented using CUFFT library
  * \return real number part of the complex number
  */
 __device__ double get_real(cufftDoubleComplex comp_data)
 {
     return comp_data.x;
 }
 
 /**
  * \brief Convert array of real number into array of complex number
  * \param real_array Array of double real number
  * \param complex_array Array of double complex number, implemented using CUFFT library
  * \param dimension Size of all arrays
  */
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
 
 /**
  * \brief Get array of absolute value from array of complex number
  * \param complex_array Array of double complex number, implemented using CUFFT library
  * \param real_array Array of absolute value of the complex number
  * \param dimension Size of all arrays
  */
 __global__ void get_absolute_array(cufftDoubleComplex *complex_array, double *real_array , int dimension)
 {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dimension; idx += blockDim.x * gridDim.x)  
     {
         real_array[idx] = cuCabs(complex_array[idx]);
     }
 }
 
 /**
  * \brief Create states for random values
  * \param seed Current time in seconds (seed for randomizer)
  * \param states Random states generated by for CURAND
  * \param dimension Size of all arrays
  */
 __global__ void init_random(double seed, curandState_t *states, int dimension)
 {
     for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dimension; idx += blockDim.x * gridDim.x) 
     { 
         curand_init(seed, idx, 0, &states[idx]);
     }
 }
 
 /**
  * \brief Sample random phase using array of random from input
  * \param random Array of random numbers with the size of dimension
  * \param y_hat Random phase data with size of dimension
  * \param ptrMag Array of magnitudes, which is the absoulte value of the initial FFT result of input image
  * \param dimension Size of all arrays
  */
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
 
 /**
  * \brief Sample random phase using curandState_t as random value
  * \param states Random states generated by for CURAND
  * \param y_hat Random phase data with size of dimension
  * \param ptrMag Array of magnitudes, which is the absoulte value of the initial FFT result of input image
  * \param dimension Size of all arrays
  */
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
 
 /**
  * \brief Satisfy fourier domain constraints
  * \param y_hat Random phase data with size of dimension
  * \param x_hat Complex version of image result in an iteration
  * \param ptrMag Array of magnitudes, which is the absoulte value of the initial FFT result of input image
  * \param dimension Size of all arrays
  */
 __global__ void satisfy_fourier(cufftDoubleComplex *y_hat, cufftDoubleComplex *x_hat, double *ptrMag, int dimension ) 
 {
     cufftDoubleComplex complex1i, exp_target, mag_comp;
     complex1i.x = 0; complex1i.y = 1;
     for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dimension; idx += blockDim.x * gridDim.x)  
     {
         mag_comp.x = ptrMag[idx];
         mag_comp.y = 0;
         exp_target.x = atan2(x_hat[idx].y, x_hat[idx].x); //arg = atan2(imag, real)
         exp_target.y = 0;
         y_hat[idx] = cuCmul(mag_comp, gpu_exp(cuCmul(complex1i, exp_target)));
     }
 }
 
 /**
  * \brief Processing random phase array with mask, generating image result
  * \param mask Input mask
  * \param y_hat Random phase data with size of dimension
  * \param image_x Image result
  * \param image_x_p Image result from previous iteration
  * \param image_x_comp Complex version of image result in an iteration
  * \param beta Input beta
  * \param mode Integer version of Input mode (hybrid = 1, input-output = 2, output-output = 3)
  * \param iter current iteration
  * \param dimension Size of all arrays
  */
 __global__ void process_arrays(double *mask, cufftDoubleComplex *y_hat, double *image_x, double *image_x_p, cufftDoubleComplex *image_x_comp, double beta, int mode, int iter, int dimension)
 {
     for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dimension; idx += blockDim.x * gridDim.x)  
     {
         bool logical_not_mask;
         bool y_less_than_zero;
         bool logical_and;
         bool indices;
 
         double y = get_real(normalize(y_hat[idx], dimension)); //Get real version of normalized complex number
 
         /**
         * \b Process
         * 1. Get previous image based on current iteration
         */
         if(iter == 0) image_x_p[idx] = y;
         else image_x_p[idx] = image_x[idx];
 
         /**
          * 2. Updates for elements that satisfy object domain constraint
          */
         if(mode == 3 || mode == 1) image_x[idx] = y;
 
         /**
          * 3. Find elements that violate object domain constraints or are not masked 
          */
         if(mask[idx] <= 0) logical_not_mask = true;
         else if(mask[idx] >= 1) logical_not_mask = false;
 
         /**
          * 4. Check if any element y is less than zero 
          */
         if(y < 0) y_less_than_zero = true;
         else if(y >= 0) y_less_than_zero = false;
 
         /**
          * 5. Use "and" logical to check the "less than zero y" and the mask
          */   
         if(y_less_than_zero == true && mask[idx] >= 1) logical_and = true;
         else logical_and = false;
 
         /**
          * 6. Determine indices value
          */  
         if(logical_and == false && logical_not_mask == false) indices = false;
         else indices = true;
 
         /**
          * 7. Updates for elements that violate object domain constraints
          */ 
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
 
 /**
  * \brief CUFFT error checking
  * \param cufft_process Result of a CUFFT operation
  */ 
 void CUFFT_CHECK(cufftResult cufft_process)
 {
     if(cufft_process != CUFFT_SUCCESS) cout<<cufft_process<<endl;
 }


