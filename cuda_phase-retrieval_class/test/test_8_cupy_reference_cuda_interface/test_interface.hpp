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