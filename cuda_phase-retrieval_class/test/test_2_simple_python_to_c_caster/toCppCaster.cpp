#include <pybind11/pybind11.h>

#include <string>
#include <sstream>
#include "toCpp.hpp"
#include "toCppCaster.hpp"

std::string test_toCpp(toCpp tc){
  std::stringstream ss;
  ss << tc.text << " " << std::to_string(tc.version);
  return ss.str();
}

PYBIND11_MODULE(toCppCaster, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("test_toCpp", &test_toCpp, "Function concatenate the text version member of the toCpp object.");
}
