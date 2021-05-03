#include <stdio.h>
#include <string>
#include <cufft.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


//store C++ and CUDA version of phase retrieval
#include "gpu_algo.hpp"

namespace py = pybind11;

PYBIND11_MODULE(gpuMemManagement, m) 
{
  //cuda
  m.def("getDeviceNumber", &getDeviceNumber);
  m.def("getNumberofSM", &getNumberofSM);
  
  //1st try
  m.def("update_images", &update_images);

  //2nd try
  m.def("copy_parted_image_to_device", &copy_parted_image_to_device);
  m.def("update_images_v2", &update_images_v2);
  m.def("allocate_device", &allocate_device);

  //3rd try
  m.def("update_images_stream", &update_images_stream);

  //4th try, 
  //first part still return a float, not an array
  // m.def("copy_to_device", &copy_to_device, py::return_value_policy::copy);
  m.def("copy_to_device", [](py::array_t<double, py::array::c_style> image, int size)
    {
      py::buffer_info bufImg = image.request();
      double *ptrImg = static_cast<double*>(bufImg.ptr);

      double *image_dev;
      CUDA_CHECK(cudaMallocManaged(&image_dev, size * sizeof(double))); //unified memory
      // CUDA_CHECK(cudaMemcpy(image_dev, ptrImg, size * sizeof(double), cudaMemcpyHostToDevice));
      image_dev = ptrImg;
      return ptrImg;
    } 
  );

  //second part
  m.def("update_images_v4", &update_images_v4);
}


