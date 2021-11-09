#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "tags.hpp"
#include "cupy_ref.hpp"

template<typename T>
struct Mem_Ref_Detail;

template<>
struct Mem_Ref_Detail<CPU>{
  using type = typename pybind11::array_t<double, pybind11::array::c_style>;
};

template<>
struct Mem_Ref_Detail<CUDAGPU>{
  using type = typename Cupy_Ref<double>;
};

template <typename T>
using Mem_Ref = Mem_Ref_Detail<T>::type;