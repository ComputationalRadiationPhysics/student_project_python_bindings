#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "tags.hpp"

#ifdef ENABLED_CUDA
#include "cupy_ref.hpp"
#endif

#ifdef ENABLED_HIP
#include "hip_ref.hpp"
#endif

template<typename T>
struct Mem_Ref_Detail;

template<>
struct Mem_Ref_Detail<CPU>{
  using type = pybind11::array_t<double, pybind11::array::c_style>;
};

#ifdef ENABLED_CUDA
template<>
struct Mem_Ref_Detail<CUDAGPU>{
  using type = Cupy_Ref<double>;
};
#endif

#ifdef ENABLED_HIP
template<>
struct Mem_Ref_Detail<HIPGPU>{
  using type = Hip_Ref<double>;
};
#endif 

template <class T>
using Mem_Ref = typename Mem_Ref_Detail<T>::type;

std::vector<std::tuple<std::string, bool>> get_available_device()
{
  std::vector<std::tuple<std::string, bool>> device_list;
  device_list.push_back({"CPU", true});

  #ifdef ENABLED_CUDA
  device_list.push_back({"GPU-CUDA", true});
  #else
  device_list.push_back({"GPU-CUDA", false});
  #endif

  #ifdef ENABLED_HIP
  device_list.push_back({"GPU-HIP", true});
  #else
  device_list.push_back({"GPU-HIP", false});
  #endif

  return device_list;
}
bool is_cuda_available()
{
  #ifdef ENABLED_CUDA
  return true;
  #else
  return false;
  #endif
}

bool is_hip_available()
{
  #ifdef ENABLED_HIP
  return true;
  #else
  return false;
  #endif
}