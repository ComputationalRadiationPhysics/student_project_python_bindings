#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "test_interface.hpp"

PYBIND11_MODULE(Test_Interface, m) 
{
  m.def("get", &get);
}


