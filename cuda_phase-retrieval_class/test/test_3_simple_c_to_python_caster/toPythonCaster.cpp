#include <pybind11/pybind11.h>

#include "toPythonCaster.hpp"
#include "toPython.hpp"

#include <vector>

toPython test_toPython(){
  std::vector<int> v = {1, 2, 3, 4};

  toPython tp;
  tp.number = 3.5f;
  tp.data = v;

  return tp;
}

PYBIND11_MODULE(toPythonCaster, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("test_toPython", &test_toPython, "Function returns a toPython.toPython object.");
}
