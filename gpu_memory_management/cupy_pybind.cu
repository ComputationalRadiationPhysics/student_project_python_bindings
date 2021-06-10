#include <stdio.h>
#include <string>
#include <cufft.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


//store C++ and CUDA version of phase retrieval
#include "cupy_ref.hpp"
#include "cupy_caster.hpp"
#include "gpu_algo.hpp"
#include "test/test.hpp"

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


  //---------------------TEST-------------------------//
  m.def("test_if_reinterpret_ptr_is_the_same", &test_if_reinterpret_ptr_is_the_same);
  m.def("test_if_real_cupy_reinterpret_ptr_is_a_gpu_array", &test_if_real_cupy_reinterpret_ptr_is_a_gpu_array);
  m.def("test_if_custom_cupy_reinterpret_ptr_is_a_gpu_array", &test_if_custom_cupy_reinterpret_ptr_is_a_gpu_array);
  m.def("test_copy_cupy_of_ones_to_cpu", &test_copy_cupy_of_ones_to_cpu);
  m.def("test_copy_custom_cupy_of_ones_to_cpu", &test_copy_custom_cupy_of_ones_to_cpu);
  m.def("test_copy_custom_cupy_of_float_to_cpu", &test_copy_custom_cupy_of_float_to_cpu);
  m.def("test_2_different_reiterpret_cast", &test_2_different_reiterpret_cast);
}



