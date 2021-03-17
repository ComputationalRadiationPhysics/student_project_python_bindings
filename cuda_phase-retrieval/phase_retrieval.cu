#include <stdio.h>
#include <string>
#include <cufft.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/imgcodecs.hpp>

//store C++ and CUDA version of phase retrieval
#include "phase_algo.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cuPhaseRet, m) 
{
  //main phase retrieval
  m.def("fienup_phase_retrieval", py::overload_cast<py::array_t<complex<double>, py::array::c_style>, py::array_t<double, py::array::c_style>, int, bool, string, double, py::array_t<double, py::array::c_style>>(&fienup_phase_retrieval));
  m.def("fienup_phase_retrieval", py::overload_cast<py::array_t<complex<double>, py::array::c_style>, py::array_t<double, py::array::c_style>, int, bool, string, double>(&fienup_phase_retrieval));
}


