#include <pybind11/numpy.h>
#include <pybind11/stl.h>

//store C++ and CUDA version of phase retrieval
#include "test_mem_ref.hpp"

PYBIND11_MODULE(Test_Mem_Ref, m) 
{
    m.def("get_available_device", &get_available_device);
    m.def("get_cuda", &get_cuda);

    pybind11::class_<Algo<CPU>>(m, "AlgoCPU", pybind11::module_local())
        .def(pybind11::init())
        .def("whoami", &Algo<CPU>::whoami)
        .def("initialize_array", &Algo<CPU>::initialize_array)
        .def("get_input_memory", &Algo<CPU>::get_input_memory)
        .def("get_output_memory", &Algo<CPU>::get_output_memory)
        .def("compute", &Algo<CPU>::compute);

    #ifdef ENABLED_CUDA
    pybind11::class_<Algo<CUDAGPU>>(m, "AlgoCUDA", pybind11::module_local())
        .def(pybind11::init())
        .def("whoami", &Algo<CUDAGPU>::whoami)
        .def("initialize_array", &Algo<CUDAGPU>::initialize_array)
        .def("get_input_memory", &Algo<CUDAGPU>::get_input_memory)
        .def("get_output_memory", &Algo<CUDAGPU>::get_output_memory)
        .def("compute", &Algo<CUDAGPU>::compute);
    #endif 
}



