#include <stdio.h>
#include <string>
#include <cufft.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


//store C++ and CUDA version of phase retrieval
#include "gpu_algo.hpp"

namespace py = pybind11;
using namespace std;

PYBIND11_MODULE(gpuMemManagement, m) 
{
  //cuda
  m.def("getDeviceNumber", &getDeviceNumber);
  m.def("getNumberofSM", &getNumberofSM);

  m.def("copy_to_device", &copy_to_device);
  m.def("update_images", &update_images);
  m.def("free_gpu_memory", &free_gpu_memory);

  //cuda stream test
  m.def("update_images_stream", &update_images_stream);

  
}



