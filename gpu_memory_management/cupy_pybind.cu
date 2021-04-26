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
  
  //1st try
  m.def("update_images", &update_images);

  //2nd try
  m.def("copy_to_device", &copy_to_device, py::return_value_policy::reference);

  //3rd try
  m.def("copy_parted_image_to_device", &copy_parted_image_to_device);
  m.def("update_images_v3", &update_images_v3);

  //4th try
  m.def("update_images_stream", &update_images_stream);
}


