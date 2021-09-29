#include <pybind11/numpy.h>
#include "phase_algo.hpp"

PYBIND11_MODULE(cuPhaseRet, m) 
{
  pybind11::enum_<Mode>(m, "Mode")
        .value("Hybrid", Hybrid)
        .value("InputOutput", InputOutput)
        .value("OutputOutput", OutputOutput)
        .export_values();

  pybind11::class_<Phase_Algo<double>>(m, "Phase_Algo", pybind11::module_local())
      .def(pybind11::init<pybind11::array_t<double, pybind11::array::c_style>, pybind11::array_t<double, pybind11::array::c_style>, Mode, double>())
      .def(pybind11::init<pybind11::array_t<double, pybind11::array::c_style>, pybind11::array_t<double, pybind11::array::c_style>, Mode, double, pybind11::array_t<double, pybind11::array::c_style>>())
      .def("iterate_random_phase", &Phase_Algo<double>::iterate_random_phase)
      .def("reset_random_phase", &Phase_Algo<double>::reset_random_phase)
      .def("do_cufft_inverse", &Phase_Algo<double>::do_cufft_inverse)
      .def("do_process_arrays", &Phase_Algo<double>::do_process_arrays)
      .def("do_cufft_forward", &Phase_Algo<double>::do_cufft_forward)
      .def("do_satisfy_fourier", &Phase_Algo<double>::do_satisfy_fourier)
      .def("get_random_phase_custom_cupy", &Phase_Algo<double>::get_random_phase_custom_cupy)
      .def("get_result", &Phase_Algo<double>::get_result);
}