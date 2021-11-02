#include <pybind11/pybind11.h>

#include "bidirectCaster.hpp"
#include "bidirect.hpp"

#include <vector>

data_container test_simple_return(){
  std::vector<double> v = {1.0, 2.0, 3.0, 4.0};
  data_container d;
  d.data = v;
  d.scalar = 1.5;

  return d;
}

data_container copy_data(data_container input){
  data_container result;
  // during development, there was an bug, that the data vector was correct copied
  // only the length of data was overtook and all values was setup with the initial value 0.0
  // to detect this behavior, data is initial setup with other values, than 0.0
  result.data = {-1.0, -1.0};

  result.data = input.data;
  result.scalar = input.scalar;
  return result;
}

data_container add_scale(data_container input1, data_container input2){
  data_container result;
  if (input1.data.size() != input2.data.size()){
    throw std::length_error("input1 and input2 has not the same size");
  }

  result.data.resize(input1.data.size());

  for(std::vector<double>::size_type i = 0; i < input1.data.size(); ++i){
    result.data[i] = input1.scalar * input1.data[i] + input2.scalar * input2.data[i];
  }
  result.scalar = 1.5;

  return result;
}

PYBIND11_MODULE(bidirectCaster, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("test_simple_return", &test_simple_return, "Function returns a bidirect.data_container object.");
    m.def("copy_data", &copy_data, "Create a copy of the input.");
    m.def("add_scale", &add_scale, "Do the mathematic function: result.data[i] = input1.scalar * input1.data[i] + input2.scalar * input2.data[i];");
}
