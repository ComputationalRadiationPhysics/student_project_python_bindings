#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "numpy_ref.hpp"
#include "numpy_ref_caster.hpp"

#include <iostream>
#include <sstream>
#include <cstdint>

/* Set the first element of a numpy array to 42. */
void change_first_element_numpy_array(pybind11::array_t<double> &input){
  pybind11::buffer_info buf = input.request();
  static_cast<double*>(buf.ptr)[0] = 42.0;
}

/* Returns the memory address of the numpy array data as string. */
std::string return_numpy_ptr_as_string(pybind11::array_t<double> &input){
  pybind11::buffer_info buf = input.request();

  std::ostringstream oss;
  oss << buf.ptr;

  return oss.str();
}

/* get the pointer of single numpy value of type uint32, deference
the pointer and return the value */
std::uint32_t dereference_pointer_numpy_uint32(std::size_t py_ptr){
  std::ostringstream oss;
  oss << "[cpp] memory address: ";
  oss << py_ptr;

  pybind11::print(oss.str());
  std::uint32_t * ptr = reinterpret_cast<std::uint32_t *>(py_ptr);
  pybind11::print("[cpp] value ptr: ", *ptr);
  return *ptr;
}

/* Get a pointer of a numpy array with the data type uint32. The functions add an offset
to each element of the array. */
void add_to_numpy_array_uint32(std::size_t py_ptr, std::size_t range, std::uint32_t offset){
  std::uint32_t * ptr = reinterpret_cast<std::uint32_t *>(py_ptr);

  for(size_t i = 0; i < range; ++i){
    ptr[i] += offset;
  }
}

/* Get a reference to a numpy array with the data type uint32. The functions add an offset
to each element of the array. */
void add_to_simple_numpy_ref(Simple_Numpy_Ref ref, std::size_t range, std::uint32_t offset){
  for(size_t i = 0; i < range; ++i){
    ref.ptr[i] += offset;
  }
}

/* Get a reference to a numpy array with the data type uint32. Initialize each value with
it's index position. */
void init_advanced_numpy_ref(Advanced_Numpy_Ref ref){
  for(size_t i = 0; i < ref.size; ++i){
    ref[i] = i;
  }
}

/* Get a reference to a numpy array with the data type uint32. The functions add an offset
to each element of the array. */
void add_to_advanced_numpy_ref(Advanced_Numpy_Ref ref, std::uint32_t offset){
  for(auto & r : ref){
    r += offset;
  }
}


PYBIND11_MODULE(pointerCaster, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("change_first_element_numpy_array", &change_first_element_numpy_array, "Set the first element of a numpy array to 42.");
    m.def("return_numpy_ptr_as_string", &return_numpy_ptr_as_string, "Returns the memory address of the numpy array data as string.");
    m.def("dereference_pointer_numpy_uint32", &dereference_pointer_numpy_uint32);
    m.def("add_to_numpy_array_uint32", &add_to_numpy_array_uint32);
    m.def("add_to_simple_numpy_ref", &add_to_simple_numpy_ref);
    m.def("init_advanced_numpy_ref", &init_advanced_numpy_ref);
    m.def("add_to_advanced_numpy_ref", &add_to_advanced_numpy_ref);
}
