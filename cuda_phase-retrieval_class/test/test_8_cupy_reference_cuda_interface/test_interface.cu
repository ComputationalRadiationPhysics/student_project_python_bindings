#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include "test_interface.hpp"
#include <iostream>
#include "gpu_memory_holder.hpp"

PYBIND11_MODULE(Test_Interface, m) 
{
    pybind11::class_<GPU_memory_holder<double>>(m, "GPU_memory_holder", pybind11::module_local())
    .def(pybind11::init<std::vector<int>>())
    .def("get_memory_reference", &GPU_memory_holder<double>::get_memory_reference);
}

