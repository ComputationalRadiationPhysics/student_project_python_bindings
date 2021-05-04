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
  m.def("copy_to_device", [=](py::array_t<double, py::array::c_style> image, int size)
    {
      py::buffer_info bufImg = image.request();
      double *ptrImg = static_cast<double*>(bufImg.ptr);

      vector<double> img_vec(ptrImg, ptrImg+size);

      double *image_dev;
      CUDA_CHECK(cudaMallocManaged(&image_dev, size * sizeof(double))); //unified memory
    
      image_dev = img_vec.data();
      
      return image_dev;
    } 
  );
  
  //4th try, second part, somehow without any return, it is working
  m.def("update_images_v4", [&](size_t image, size_t partial_update, double update, int size) 
    {
      //get the value of the address
      double *ptrImg = reinterpret_cast<double*>(image);
      double *ptrUpd = reinterpret_cast<double*>(partial_update);

      int devId, numSMs;
      cudaGetDevice(&devId);
      cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, devId);
      
      partial_image_update<<<8*numSMs, 256>>>(ptrImg, ptrUpd, update, size);

      cudaDeviceSynchronize();
    }
  );
}



