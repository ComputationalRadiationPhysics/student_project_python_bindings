#include <stdio.h>
#include <string>
#include <cufft.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

//store C++ and CUDA version of phase retrieval
#include "gpu_algo.hpp"

namespace py = pybind11;

PYBIND11_MODULE(gpuMemManagement, m) 
{
  m.def("getDeviceNumber", &getDeviceNumber);
  m.def("update_images", &update_images);
}


