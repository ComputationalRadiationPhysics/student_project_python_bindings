#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "test_algo.hpp"

PYBIND11_MODULE(Test_Algorithm, m) 
{
  //---test_algo----------
  m.def("array_check", &array_check);
  m.def("array_check_cuda", &array_check_cuda);
  m.def("array_check_complex", &array_check_complex);
  m.def("array_check_complex_cuda", &array_check_complex_cuda);
  m.def("cufft_inverse_forward", &cufft_inverse_forward);
  m.def("abs_cufft_forward", &abs_cufft_forward);
}


