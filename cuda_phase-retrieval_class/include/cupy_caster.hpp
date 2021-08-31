#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <cstdio>
#include <iostream>
#include <complex>
#include <string>

using namespace std::literals::complex_literals;
using namespace pybind11::literals;

template<typename TData> std::string cupy_ref_get_dtype(){ return "C++ type not implemented";}
template<> std::string cupy_ref_get_dtype<short int>(){ return "int16";}
template<> std::string cupy_ref_get_dtype<int>(){ return "int32";}
template<> std::string cupy_ref_get_dtype<long long int>(){ return "int64";}
template<> std::string cupy_ref_get_dtype<std::uint16_t>(){ return "uint16";}
template<> std::string cupy_ref_get_dtype<std::uint32_t>(){ return "uint32";}
template<> std::string cupy_ref_get_dtype<std::uint64_t>(){ return "uint64";}
template<> std::string cupy_ref_get_dtype<float>(){ return "float32";}
template<> std::string cupy_ref_get_dtype<double>(){ return "float64";}
template<> std::string cupy_ref_get_dtype<std::complex<float>>(){ return "complex64";}
template<> std::string cupy_ref_get_dtype<std::complex<double>>(){ return "complex128";}


namespace pybind11 { namespace detail {
    template <typename T, int TDim> struct type_caster<Custom_Cupy_Ref<T, TDim>> 
    {
        using Custom_Cupy_Ref_t = Custom_Cupy_Ref<T, TDim>;

        public:
            PYBIND11_TYPE_CASTER(Custom_Cupy_Ref_t, _("cupy_ref.Custom_Cupy_Ref"));
      
        // python -> C++
        bool load(handle src, bool)
        {
            if(!hasattr(src, "ptr") && !hasattr(src, "size") && !hasattr(src, "dtype") && !hasattr(src, "shape"))
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

            if(TDim != 0 && src.attr("shape").cast<std::vector<std::size_t>>().size() != TDim)
            {
                std::ostringstream oss;
                oss << "Wrong cupy dimension\n";
                oss << "Current dimension : " << src.attr("shape").cast<std::vector<std::size_t>>().size() << "\n";
                oss << "Expected dimension : " << TDim << "\n"; 
                std::cerr << oss.str();
				return false;  
            }
            
            value.ptr = reinterpret_cast<T *>(src.attr("ptr").cast<std::size_t>());
            value.size = src.attr("size").cast<std::size_t>();
            value.dtype = src.attr("dtype").cast<std::string>();
            value.shape = src.attr("shape").cast<std::vector<std::size_t>>();
           
            return true;
        }

        static handle cast(Custom_Cupy_Ref<T> src, return_value_policy /* policy */, handle /* parent */) {

            //in the previous function "load" to convert python to c++, I need to reinterpret cast a size_t of the ptr to a double*.
            //in this function, because I have to convert it from c++ to python, then I need to convert the ptr from double* to size_t 
            std::size_t python_pointer = reinterpret_cast<std::size_t>(src.ptr);
            auto custom_cupy = module::import("cupy_ref").attr("Custom_Cupy_Ref")(python_pointer, src.size, src.dtype, src.shape);
            return custom_cupy.release();
        }
    };
}}