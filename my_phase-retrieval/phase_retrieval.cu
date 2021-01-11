#include <stdio.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>

#include "phase_algo.hpp"

namespace py = pybind11;

// template <typename ALGO_CLASS_T>
// void decare_algo_binding(py::module &m, const std::string &class_name) 
// {
//   py::class_<ALGO_CLASS_T>(m, class_name.c_str(), py::module_local())
//     //   .def(py::init<size_t>())
//     //   .def("init", &ALGO_CLASS_T::init)
//     //   .def("deinit", &ALGO_CLASS_T::deinit)
//     //   .def("get_size", &ALGO_CLASS_T::get_size)
//     //   .def("get_input_view", &ALGO_CLASS_T::get_input_view,
//     //        py::return_value_policy::reference_internal)
//     //   .def("get_output_view", &ALGO_CLASS_T::get_output_view,
//     //        py::return_value_policy::reference_internal)
//     //   .def("compute", &ALGO_CLASS_T::compute);
//     .def("fienup_phase_retrieval", &ALGO_CLASS_T::fienup_phase_retrieval)
//     .def("setMag", &ALGO_CLASS_T::setMag);
// }

PYBIND11_MODULE(cuPhaseRet, m) {
  m.def("fienup_phase_retrieval", &fienup_phase_retrieval);
}


