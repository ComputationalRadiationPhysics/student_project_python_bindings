#include <pybind11/numpy.h>
#include <pybind11/stl.h>

//store C++ and CUDA version of phase retrieval
#include "test_custom_cupy.hpp"

PYBIND11_MODULE(Test_Custom_Cupy, m) 
{
  m.def("test_if_reinterpret_ptr_is_the_same", &test_if_reinterpret_ptr_is_the_same);
  m.def("test_if_real_cupy_reinterpret_ptr_is_a_gpu_array", &test_if_real_cupy_reinterpret_ptr_is_a_gpu_array);
  m.def("test_if_custom_cupy_reinterpret_ptr_is_a_gpu_array", &test_if_custom_cupy_reinterpret_ptr_is_a_gpu_array);
  m.def("test_copy_custom_cupy_to_cpu", &test_copy_custom_cupy_to_cpu);
  m.def("test_copy_real_cupy_to_cpu", &test_copy_real_cupy_to_cpu);
  m.def("real_cupy_increment_all_data_by_1", &real_cupy_increment_all_data_by_1);
  m.def("custom_cupy_increment_all_data_by_1", &custom_cupy_increment_all_data_by_1);
  m.def("test_create_real_cupy_from_c", &test_create_real_cupy_from_c, pybind11::return_value_policy::move);
  m.def("test_copy_custom_cupy_to_custom_cupy", &test_copy_custom_cupy_to_custom_cupy);
  m.def("test_wrong_dtype_float", &test_wrong_dtype_float);
  m.def("test_wrong_dtype_int", &test_wrong_dtype_int);
  m.def("test_wrong_dtype_complex", &test_wrong_dtype_complex);

  //missmatch error always appeared based on the order of types
  //ex : if cupy with double type is used, them there will be 3 missmatch error for uint16, uint32, and float
  //ex : if cupy with complex128 type is used, them there will be 5 missmatch error for uint16, uint32, float, double, and complex64
  //this error appeard with "pytest -s"
  m.def("test_custom_cupy_template_function", &test_custom_cupy_template_function<uint16_t>);
  m.def("test_custom_cupy_template_function", &test_custom_cupy_template_function<uint32_t>);
  m.def("test_custom_cupy_template_function", &test_custom_cupy_template_function<float>);
  m.def("test_custom_cupy_template_function", &test_custom_cupy_template_function<double>);
  m.def("test_custom_cupy_template_function", &test_custom_cupy_template_function<std::complex<float>>);
  m.def("test_custom_cupy_template_function", &test_custom_cupy_template_function<std::complex<double>>);

  //overloading produce the same errors
  // m.def("test_custom_cupy_template_function", pybind11::overload_cast<size_t, Custom_Cupy_Ref<uint16_t>>(&test_custom_cupy_template_function<uint16_t>));
  // m.def("test_custom_cupy_template_function", pybind11::overload_cast<size_t, Custom_Cupy_Ref<uint32_t>>(&test_custom_cupy_template_function<uint32_t>));
  // m.def("test_custom_cupy_template_function", pybind11::overload_cast<size_t, Custom_Cupy_Ref<float>>(&test_custom_cupy_template_function<float>));
  // m.def("test_custom_cupy_template_function", pybind11::overload_cast<size_t, Custom_Cupy_Ref<double>>(&test_custom_cupy_template_function<double>));
  // m.def("test_custom_cupy_template_function", pybind11::overload_cast<size_t, Custom_Cupy_Ref<complex<float>>>(&test_custom_cupy_template_function<complex<float>>));
  // m.def("test_custom_cupy_template_function", pybind11::overload_cast<size_t, Custom_Cupy_Ref<complex<double>>>(&test_custom_cupy_template_function<complex<double>>));

}



