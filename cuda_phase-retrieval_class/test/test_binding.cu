#include <cstdio>
#include <string>
#include <cufft.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "cupy_ref.hpp"
#include "cupy_caster.hpp"
#include "test_algo.hpp"
#include "test_cupy.hpp"

namespace py = pybind11;
using namespace std;

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

  //test_cupy
  m.def("test_generating_cupy_of_complex_double_from_c", &test_generating_cupy_of_complex_double_from_c);
  m.def("test_send_cupy_complex_to_c_and_send_it_back", &test_send_cupy_complex_to_c_and_send_it_back);
  m.def("test_cupy_cufft_inverse_forward", &test_cupy_cufft_inverse_forward);
  m.def("test_cupy_cufft_inverse_forward_with_caster", &test_cupy_cufft_inverse_forward_with_caster<complex<double>>);
  m.def("test_send_cupy_caster_to_c_and_get_it_back", &test_send_cupy_caster_to_c_and_get_it_back<complex<double>>);
  m.def("test_cupy_from_c_memory", &test_cupy_from_c_memory);
}


