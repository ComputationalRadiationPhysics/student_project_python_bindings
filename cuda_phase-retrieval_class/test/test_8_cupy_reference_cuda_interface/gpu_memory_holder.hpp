#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdio>
#include <iostream>
#include "cupy_ref.hpp"
#include "cupy_caster.hpp"
#include "cupy_allocate.hpp"

template<typename T>
class GPU_memory_holder 
{
    public:

    std::vector<int> shape;

    private:
        pybind11::object gpu_memory;
        Cupy_Ref<T> gpu_memory_reference;

    public:
        GPU_memory_holder(std::vector<int> shape)
        {
            this->shape = shape;
            gpu_memory = cupy_allocate<T>(shape);
            gpu_memory_reference = Cupy_Ref<T>::getCupyRef(gpu_memory);
        }


        Cupy_Ref<T> get_memory_reference()
        {
            return gpu_memory_reference;
        }
    };