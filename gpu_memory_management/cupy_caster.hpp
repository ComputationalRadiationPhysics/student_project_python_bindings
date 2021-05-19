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


struct cupy_array
{
    // std::string typestr;
    // int version;
    size_t size;
    size_t address;
    int device_id;
};

namespace pybind11 { namespace detail {
    template <> struct type_caster<cupy_array> 
    {
      // cupy_array value;
    public:
        PYBIND11_TYPE_CASTER(cupy_array, _("cupy.core.core.ndarray"));
      
        // python -> C++
        bool load(handle src, bool)
        {
            if(hasattr(src, "size"))
            {
                value.size = src.attr("size").cast<size_t>();
            } 
            else 
            {
                return false;
            }

            if(hasattr(src, "data"))
            {
                value.address = src.attr("data").attr("ptr").cast<size_t>();
                value.device_id = src.attr("data").attr("device_id").cast<int>();
            } 
            else 
            {
                return false;
            }
            
            return true;
        }

        // static handle cast(const cupy_array src, return_value_policy, handle) {
        //     py::object cp = py::module_::import("cupy");
        //     py::object ones = cp.attr("ones")(3);
        //     return ones.release();
        // }
    };
}}