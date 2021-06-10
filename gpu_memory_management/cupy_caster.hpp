#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <cmath>
#include <cstdio>
#include <time.h>
#include <iostream>
#include <complex>
#include <string>
#include <random>

#define CUDA_CHECK(call) {cudaError_t error = call; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}
using namespace std;
using namespace std::literals::complex_literals;
namespace py = pybind11;


namespace pybind11 { namespace detail {
    template <> struct type_caster<Custom_Cupy_Ref> 
    {
    public:
        PYBIND11_TYPE_CASTER(Custom_Cupy_Ref, _("cupy_ref.Custom_Cupy_Ref"));
      
        // python -> C++
        bool load(handle src, bool)
        {
            if(!hasattr(src, "ptr") && !hasattr(src, "size"))
            {
                 return false;
            }
            
            value.ptr = reinterpret_cast<double *>(src.attr("ptr").cast<size_t>());
            value.size = src.attr("size").cast<size_t>();
           
            return true;
        }
    };
}}