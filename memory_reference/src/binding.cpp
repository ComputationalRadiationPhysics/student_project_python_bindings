#include "tags.hpp"
#include "algo.hpp"

PYBIND11_MODULE(binding, m) 
{
    m.def("is_cuda_available", &is_cuda_available);
    
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