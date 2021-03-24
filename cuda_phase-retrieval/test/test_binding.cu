#include <stdio.h>
#include <string>
#include <cufft.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "test_algo.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cuPhaseRet_Test, m) 
{
  m.def("array_check", &array_check);
  m.def("array_check_cuda", &array_check_cuda);
  m.def("array_check_complex", &array_check_complex);
  m.def("array_check_complex_cuda", &array_check_complex_cuda);
  m.def("cufft_inverse_forward", &cufft_inverse_forward);
  m.def("abs_cufft_forward", &abs_cufft_forward);

  //phase retrieval with verbose
  m.def("fienup_phase_retrieval", py::overload_cast<py::array_t<double, py::array::c_style>, py::array_t<double, py::array::c_style>, int, bool, string, double, py::array_t<double, py::array::c_style>>(&fienup_phase_retrieval));
}


