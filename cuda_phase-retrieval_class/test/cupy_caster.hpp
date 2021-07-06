#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <cstdio>
#include <iostream>
#include <complex>
#include <string>

using namespace std;
using namespace std::literals::complex_literals;
namespace py = pybind11;

template<typename TData> std::string cupy_ref_get_dtype(){ return "C++ type not implemented";}

template<> std::string cupy_ref_get_dtype<float>(){ return "float32";}
template<> std::string cupy_ref_get_dtype<double>(){ return "float64";}
template<> std::string cupy_ref_get_dtype<std::uint16_t>(){ return "uint16";}
template<> std::string cupy_ref_get_dtype<std::uint32_t>(){ return "uint32";}
template<> std::string cupy_ref_get_dtype<std::complex<float>>(){ return "complex64";}
template<> std::string cupy_ref_get_dtype<std::complex<double>>(){ return "complex128";}

namespace pybind11 { namespace detail {
    template <typename T> struct type_caster<Custom_Cupy_Ref<T>> 
    {
    public:
        PYBIND11_TYPE_CASTER(Custom_Cupy_Ref<T>, _("cupy_ref.Custom_Cupy_Ref"));
      
        // python -> C++
        bool load(handle src, bool)
        {
            if(!hasattr(src, "ptr") && !hasattr(src, "size") && !hasattr(src, "dtype") && !hasattr(src, "shape_x") && !hasattr(src, "shape_y"))
            {
                 return false;
            }

            if(src.attr("dtype").cast<std::string>() !=
				   cupy_ref_get_dtype<T>()){
					std::ostringstream oss;
					oss << "Cupy Ref type missmatch\n";
					oss << "  Python type: " << src.attr("dtype").cast<std::string>() << "\n";
					oss << "  Expected Python type: " << cupy_ref_get_dtype<T>() << "\n";
					std::cerr << oss.str();
					return false;
            }

            
            value.ptr = reinterpret_cast<T *>(src.attr("ptr").cast<size_t>());
            value.size = src.attr("size").cast<size_t>();
            value.dtype = src.attr("dtype").cast<string>();
            value.shape_x = src.attr("shape_x").cast<size_t>();
            value.shape_y = src.attr("shape_y").cast<size_t>();
           
            return true;
        }

        static handle cast(Custom_Cupy_Ref<T> src, return_value_policy /* policy */, handle /* parent */) {

            //in the previous function "load" to convert python to c++, I need to reinterpret cast a size_t of the ptr to a double*.
            //in this function, because I have to convert it from c++ to python, then I need to convert the ptr from double* to size_t 
            size_t python_pointer = reinterpret_cast<size_t>(src.ptr);
            auto custom_cupy = module::import("cupy_ref").attr("Custom_Cupy_Ref")(python_pointer, src.size, src.dtype, src.shape_x, src.shape_y);
            return custom_cupy.release();
        }
    };
}}