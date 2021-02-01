#include <stdio.h>
#include <string>
#include <cufft.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

//store C++ and CUDA version of phase retrieval
#include "phase_algo.hpp"

namespace py = pybind11;

//if a python code calling the "fienup_phase_retrieval" function,
//it will run the c++ version of "fienup_phase_retrieval" stored in "phase_algo"
PYBIND11_MODULE(cuPhaseRet, m) 
{
  m.def("fienup_phase_retrieval", &fienup_phase_retrieval);
  // m.def("test_fft", &test_fft);
}


