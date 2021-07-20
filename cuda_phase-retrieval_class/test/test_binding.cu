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

PYBIND11_MODULE(cuPhaseRet_Test, m) 
{
  m.def("array_check", &array_check);
  m.def("array_check_cuda", &array_check_cuda);
  m.def("array_check_complex", &array_check_complex);
  m.def("array_check_complex_cuda", &array_check_complex_cuda);
  m.def("cufft_inverse_forward", &cufft_inverse_forward);
  m.def("abs_cufft_forward", &abs_cufft_forward);

  //phase retrieval with verbose
  m.def("fienup_phase_retrieval", pybind11::overload_cast<pybind11::array_t<double, pybind11::array::c_style>, pybind11::array_t<double, pybind11::array::c_style>, int, bool, std::string, double, pybind11::array_t<double, pybind11::array::c_style>>(&fienup_phase_retrieval));

  //test_cupy
  /*1*/ m.def("test_generating_cupy_of_complex_double_from_c", &test_generating_cupy_of_complex_double_from_c);
  /*2*/ m.def("test_send_cupy_complex_to_c_and_send_it_back", &test_send_cupy_complex_to_c_and_send_it_back);
  /*3*/ m.def("test_cupy_cufft_inverse_forward", &test_cupy_cufft_inverse_forward);
  /*4*/ m.def("test_create_a_custom_cupy_from_a_non_cupy_object_in_c", &test_create_a_custom_cupy_from_a_non_cupy_object_in_c<std::complex<double>>);
  /*5*/ m.def("test_create_a_custom_cupy_with_flexible_dimension", &test_create_a_custom_cupy_with_flexible_dimension<std::complex<double>>);
  /*6*/ m.def("test_create_a_custom_cupy_with_fixed_dimension_success", &test_create_a_custom_cupy_with_fixed_dimension_success<std::complex<double>>);
  /*7*/ m.def("test_create_a_custom_cupy_with_fixed_dimension_fail", &test_create_a_custom_cupy_with_fixed_dimension_fail<std::complex<double>>);
  /*8*/ m.def("test_cupy_cufft_inverse_forward_with_caster", &test_cupy_cufft_inverse_forward_with_caster<std::complex<double>>);
  /*9*/ m.def("test_send_cupy_caster_to_c_and_get_it_back", &test_send_cupy_caster_to_c_and_get_it_back<std::complex<double>>);
  /*10*/ m.def("test_cupy_from_c_memory", &test_cupy_from_c_memory);
}


