#include "tags.hpp"
#include "algo.hpp"

PYBIND11_MODULE(binding, m) 
{
    m.def("is_cuda_available", &is_cuda_available);
    m.def("is_hip_available", &is_hip_available);
    m.def("get_available_device", &get_available_device);
    
    pybind11::class_<Algo<CPU>>(m, "AlgoCPU", pybind11::module_local())
        .def(pybind11::init())
        .def("whoami", &Algo<CPU>::whoami)
        .def("is_synced_mem", &Algo<CPU>::is_synced_mem)
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

    #ifdef ENABLED_HIP
    //initialize_array, get_input and get_output will be handled by class AlgoGPUHIP in include\algogpu.py
    pybind11::class_<Algo<HIPGPU>>(m, "AlgoHIP", pybind11::module_local())
        .def(pybind11::init())
        .def("whoami", &Algo<HIPGPU>::whoami)
        .def("compute", &Algo<HIPGPU>::compute);

    pybind11::class_<Hip_Mem_Impl>(m, "Hip_Mem_Impl", pybind11::module_local())
        .def(pybind11::init())
        .def("read", &Hip_Mem_Impl::read)
        .def("write", &Hip_Mem_Impl::write)
        .def("get_hip_array", &Hip_Mem_Impl::get_hip_array);   

    #endif
}