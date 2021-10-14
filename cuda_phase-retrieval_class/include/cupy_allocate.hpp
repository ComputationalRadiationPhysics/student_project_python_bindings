#include <pybind11/numpy.h>
#include <pybind11/stl.h>

template<typename T> 
pybind11::object cupy_allocate(std::vector<int> shape)
{
    int linear_size = 1;
    for(int const &s : shape) linear_size *= s;
    pybind11::object cp = pybind11::module::import("cupy").attr("zeros")(linear_size, "dtype"_a=cupy_ref_get_dtype<T>()).attr("reshape")(shape);
    return cp;
}