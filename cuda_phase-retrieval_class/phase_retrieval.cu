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

//store C++ and CUDA version of phase retrieval
// #include "phase_algo.hpp"

#define PI 3.1415926535897932384626433
#define CUDA_CHECK(call) {cudaError_t error = call; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}
using namespace std;
using namespace std::literals::complex_literals;
namespace py = pybind11;

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
    //CPU Arrays
    py::buffer_info bufImg, bufMask, bufRand;
    size_t X, Y, 
           mask_X, mask_Y, 
           rand_X, rand_Y;    
    int mode, steps, size_x, size_y, dimension;
    double beta;
    double *ptrImg, *mask, *random_value;

    //GPU Arrays
    double *src_img_dev,        //Source image in GPU 
           *mag_dev,            //Magnitudes in GPU
           *mask_dev,           //Mask in GPU
           *image_x_device,     //Image output in GPU 
           *image_x_p_device,   //Save previous image output in GPU for iteration
           *random_value_dev;   //Array of random in GPU

    cufftDoubleComplex *y_hat_dev,          //Sample random phase in GPU
                       *src_img_dev_comp,   //complex number version of source image in GPU
                       *image_x_dev_comp;   //Complex number version of image output in GPU

  public:
    Phase_Algo(py::array_t<double, py::array::c_style> image, py::array_t<double, py::array::c_style> masks, int steps, string mode, double beta, py::array_t<double, py::array::c_style> randoms)
    {
      /**
    * \b Process
    * 1. Asserting inputs
    */
      assert(beta > 0);
      assert(steps > 0);
      assert(mode == "input-output" || mode == "output-output" || mode == "hybrid");

      setMode(mode); //use integer instead of string for mode 
      this->bufImg = image.request();       
      this->bufMask = masks.request();      
      this->bufRand = randoms.request();
      this->X = bufImg.shape[0];               //Width of image
      this->Y = bufImg.shape[1]; 
      this->steps = steps;
      this->beta = beta;

      /**
      * 2. Asserting array size, make sure all arrays are using the same size
      */
      this->mask_X = bufMask.shape[0];               //Width of mask
      this->mask_Y = bufMask.shape[1];               //Height of mask
      this->rand_X = bufRand.shape[0];               //Width of random array
      this->rand_Y = bufRand.shape[1];               //Height of random array
      assert(mask_X == X && rand_X == X);
      assert(mask_Y == Y && rand_Y == Y);

      this->ptrImg = static_cast<double*>(bufImg.ptr);          //Get 1D image array
      this->mask = static_cast<double*>(bufMask.ptr);           //Mask array, same size as image 
      this->random_value = static_cast<double*>(bufRand.ptr);   //Array of uniform random number, same size as image

     this->size_x = static_cast<int>(X); //Convert X to integer to prevent getting warning from CUFFT
     this->size_y = static_cast<int>(Y); //Convert Y to integer to prevent getting warning from CUFFT
     this->dimension = size_x*size_y;    //Area or dimension of image, mask, and array of random

      /**
      * 3. Allocating memories in GPU
      */                   
      CUDA_CHECK(cudaMalloc(&y_hat_dev, dimension * sizeof(cufftDoubleComplex)));
      CUDA_CHECK(cudaMalloc(&src_img_dev_comp, dimension * sizeof(cufftDoubleComplex)));
      CUDA_CHECK(cudaMalloc(&mag_dev, dimension * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&src_img_dev, dimension * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&mask_dev, dimension * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&image_x_device, dimension * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&image_x_p_device, dimension * sizeof(double)));
      CUDA_CHECK(cudaMalloc(&image_x_dev_comp, dimension * sizeof(cufftDoubleComplex)));
      CUDA_CHECK(cudaMalloc(&random_value_dev, dimension * sizeof(double)));
          
      }

    void setMode(string mode)
    {
      if(mode.compare("hybrid") == 0) this->mode = 1;
      else if(mode.compare("input-output") == 0) this->mode = 2;
      else if(mode.compare("output-output") == 0) this->mode = 3;
    }

    int getMode()
    {
      return this->mode;
    }

    int getSteps()
    {
      return this->steps;
    }

    double getBeta()
    {
      return this->beta;
    }

    ~Phase_Algo()
    {
      cudaFree(y_hat_dev);
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
      .def(py::init<py::array_t<double, py::array::c_style>, py::array_t<double, py::array::c_style>, int, string, double, py::array_t<double, py::array::c_style>>())
	    .def("getMode", &Phase_Algo::getMode)
      .def("getSteps", &Phase_Algo::getSteps)
      .def("getBeta", &Phase_Algo::getBeta);

}


