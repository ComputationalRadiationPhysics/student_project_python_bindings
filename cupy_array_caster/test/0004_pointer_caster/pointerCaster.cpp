#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "ptr_container.hpp"
#include "bidirectCaster.hpp"

#include <iostream>
#include <sstream>

namespace py = pybind11;

// /* Set the first element of a numpy array to 42. */
// void change_first_element_numpy_array(py::array_t<int> &input){
//   py::buffer_info buf = input.request();
//   static_cast<int*>(buf.ptr)[0] = 42;
// }

// /* Returns the memory address of the numpy array data as string. */
// std::string return_numpy_ptr_as_string(py::array_t<int> &input){
//   py::buffer_info buf = input.request();

//   std::ostringstream oss;
//   oss << buf.ptr;

//   return oss.str();
// }

/* Set the first element of a numpy array to 42. */
void change_first_element_numpy_array(simple_ptr input){
  static_cast<int*>(input.ptr)[0] = 42;
}

/* Returns the memory address of the numpy array data as string. */
std::string return_numpy_ptr_as_string(simple_ptr input){
  std::ostringstream oss;
  oss << input.ptr;

  return oss.str();
}

PYBIND11_MODULE(pointerCaster, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("change_first_element_numpy_array", &change_first_element_numpy_array, "Set the first element of a numpy array to 42.");
    m.def("return_numpy_ptr_as_string", &return_numpy_ptr_as_string, "Returns the memory address of the numpy array data as string.");
}
