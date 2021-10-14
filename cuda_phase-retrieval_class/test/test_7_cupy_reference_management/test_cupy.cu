#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "test_cupy.hpp"

PYBIND11_MODULE(Test_Cupy, m) 
{
  //---test_cupy----------
  pybind11::enum_<Mode>(m, "Mode")
        .value("Hybrid", Hybrid)
        .value("InputOutput", InputOutput)
        .value("OutputOutput", OutputOutput)
        .export_values();

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
  /*11*/ m.def("test_enum", &test_enum);
  /*12*/ m.def("test_custom_cupy_object_creator_1d", &test_custom_cupy_object_creator_1d);
  /*13*/ m.def("test_custom_cupy_object_creator_2d", &test_custom_cupy_object_creator_2d);
  /*14*/ m.def("test_custom_cupy_object_creator_3d", &test_custom_cupy_object_creator_3d);
}


