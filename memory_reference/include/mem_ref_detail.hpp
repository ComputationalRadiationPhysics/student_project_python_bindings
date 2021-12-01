#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "tags.hpp"
#include "cupy_ref.hpp"

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

template <class T>
using Mem_Ref = typename Mem_Ref_Detail<T>::type;