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
            // try 4 copying src to local cupy

            py::object cp = py::module_::import("cupy");
            py::object a = cp.attr("asarray")(src);
            
            double *cpu_data = new double[a.attr("size").cast<size_t>()];

            double *device_data = reinterpret_cast<double*>(a.attr("data").attr("ptr").cast<size_t>()); //device data is not recognized by both cpu and gpu

            CUDA_CHECK(cudaMemcpy(cpu_data, device_data, a.attr("size").cast<size_t>() * sizeof(double), cudaMemcpyDeviceToHost));


            //try 3 using reinterpret_borrow from pybind11 handle, fail
            // py::object address = reinterpret_borrow<py::object>(src.attr("data").attr("ptr"));
            // py::object size = reinterpret_borrow<py::object>(src.attr("size"));

            // double *cpu_data = (double*)malloc(size.cast<size_t>()*sizeof(double));

            // double *device_data = reinterpret_cast<double*>(address.cast<size_t>()); //device data is not recognized by both cpu and gpu

            // CUDA_CHECK(cudaMemcpy(cpu_data, device_data, size.cast<size_t>() * sizeof(double), cudaMemcpyDeviceToHost));

            // try 2 fail
            // PyObject *source = src.ptr();
            
            // py::object cp = reinterpret_borrow<py::object>(source);


            // double *cpu_data = (double*)malloc(cp.attr("size").cast<size_t>()*sizeof(double));

            // double *device_data = reinterpret_cast<double*>(cp.attr("data").attr("ptr").cast<size_t>()); //device data is not recognized by both cpu and gpu


            // CUDA_CHECK(cudaMemcpy(cpu_data, device_data, cp.attr("size").cast<size_t>() * sizeof(double), cudaMemcpyDeviceToHost));

            //try 1 fail
            
            // if(hasattr(src, "size"))
            // {
            //     value.size = src.attr("size").cast<size_t>();
            // } 
            // else 
            // {
            //     return false;
            // }

            // if(hasattr(src, "data"))
            // {
            //     value.address = src.attr("data").attr("ptr").cast<size_t>();
            //     value.device_id = src.attr("data").attr("device_id").cast<int>();
            // } 
            // else 
            // {
            //     return false;
            // }
            
            return true;
        }

        // static handle cast(const cupy_array src, return_value_policy, handle) {
        //     py::object cp = py::module_::import("cupy");
        //     py::object ones = cp.attr("ones")(3);
        //     return ones.release();
        // }
    };
}}