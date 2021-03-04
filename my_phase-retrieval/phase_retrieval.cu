#include <stdio.h>
#include <string>
#include <cufft.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

//store C++ and CUDA version of phase retrieval
#include "phase_algo.hpp"
#include "test_algo.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cuPhaseRet, m) 
{
  m.def("fienup_phase_retrieval", &fienup_phase_retrieval);
  m.def("fienup_phase_retrieval_c_random", &fienup_phase_retrieval_c_random);
  m.def("array_check", &array_check);
  m.def("array_check_cuda", &array_check_cuda);
  m.def("array_check_complex", &array_check_complex);
  m.def("array_check_complex_cuda", &array_check_complex_cuda);
  m.def("cufft_inverse_forward", &cufft_inverse_forward);
}


