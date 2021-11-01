#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include "test_interface.hpp"
#include <iostream>
#include "cupy_ref.hpp"
#include "gpu_memory_holder.hpp"

template<typename TData>
__global__ void incOneKernel(TData * const data, int const size){
  for(int i = 0; i < size; ++i){
    data[i] = data[i] + static_cast<TData>(1);
  }
}

template<typename TData>
void incOne(Cupy_Ref<TData> data, int const size){
  incOneKernel<TData><<<1,1>>>(data.ptr, size);
}

PYBIND11_MODULE(Test_Interface, m)
{
    pybind11::class_<GPU_memory_holder<double>>(m, "GPU_memory_holder", pybind11::module_local())
    .def(pybind11::init<std::vector<int>>())
    .def("get_memory_reference", &GPU_memory_holder<double>::get_memory_reference);

    m.def("incOne", &incOne<double>);
}
