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

class Phase_Algo
{
  private:
    py::buffer_info bufImg, bufMask, bufRand;
    int mode;

  public:
    Phase_Algo(int mode)
    {
      this.mode = mode
    }

    int getMode()
    {
      return mode;
    }

}

PYBIND11_MODULE(cuPhaseRet, m) 
{
  //main phase retrieval
  // m.def("fienup_phase_retrieval", py::overload_cast<py::array_t<double, py::array::c_style>, py::array_t<double, py::array::c_style>, int, string, double, py::array_t<double, py::array::c_style>>(&fienup_phase_retrieval));
  // m.def("fienup_phase_retrieval", py::overload_cast<py::array_t<double, py::array::c_style>, py::array_t<double, py::array::c_style>, int, string, double>(&fienup_phase_retrieval));

  py::class_<Phase_Algo>(m, "Phase_Algo", py::module_local())
	    .def("getMode", &getMode);

}


