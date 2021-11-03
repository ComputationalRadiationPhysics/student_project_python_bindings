#include "tags.hpp"
#include "algo.hpp"
#include "mem_ref.hpp"
#include "mem_ref_caster.hpp"

PYBIND11_MODULE(binding, m) 
{
  pybind11::class_<Algo<CPU>>(m, "AlgoCPU", pybind11::module_local())
      .def(pybind11::init())
      .def("initialize_array", &Algo<CPU>::initialize_array)
      .def("get_input_memory", &Algo<CPU>::get_input_memory)
      .def("get_output_memory", &Algo<CPU>::get_output_memory)
      .def("compute", &Algo<CPU>::compute);
  
  pybind11::class_<Mem_Ref<CPU>>(m, "Mem_RefCPU", pybind11::module_local())
      .def(pybind11::init())
      .def("get_data", &Mem_Ref<CPU>::get_data);
}