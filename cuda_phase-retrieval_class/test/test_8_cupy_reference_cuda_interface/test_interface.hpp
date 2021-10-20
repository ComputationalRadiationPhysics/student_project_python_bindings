#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cufft.h>
#include <cstdio>
#include <iostream>
#include <complex>

#include "cupy_ref.hpp"
#include "cupy_caster.hpp"
#include "cuda_algo.hpp"
#include "cupy_allocate.hpp"

using namespace std::literals::complex_literals;
using namespace pybind11::literals;

pybind11::object get(size_t address, std::vector<int> shape)
{
    int linear_size = 1;
    for(int const &s : shape) linear_size *= s;

    int *gpu_data = reinterpret_cast<int *>(address);

    pybind11::object result = cupy_allocate<int>(shape);
    int *result_data = reinterpret_cast<int *>(result.attr("data").attr("ptr").cast<std::size_t>());

    CUDA_CHECK(cudaMemcpy(result_data, gpu_data, linear_size * sizeof(int), cudaMemcpyDeviceToHost));

    return result;
}