#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdio>
#include <iostream>
#include <complex>
#include "cupy_ref.hpp"
#include "dtype_getter.hpp"

using namespace std::literals::complex_literals;
using namespace pybind11::literals;

namespace pybind11 { namespace detail {
    template <typename T, int TDim> struct type_caster<Cupy_Ref<T, TDim>> 
    {
        using Cupy_Ref_t = Cupy_Ref<T, TDim>;

        public:
            PYBIND11_TYPE_CASTER(Cupy_Ref_t, _("cupy_ref.Cupy_Ref"));
      
        // python -> C++
        bool load(handle src, bool)
        {
            if(!hasattr(src, "ptr") || !hasattr(src, "typestr") || !hasattr(src, "dtype") || !hasattr(src, "shape"))
            {
                 return false;
            }

            if(src.attr("dtype").cast<std::string>() != get_dtype<T>()){
					std::ostringstream oss;
					oss << "Cupy Ref type missmatch\n";
					oss << "  Python type: " << src.attr("dtype").cast<std::string>() << "\n";
					oss << "  Expected Python type: " << get_dtype<T>() << "\n";
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
            value.dtype = src.attr("dtype").cast<std::string>();
            value.shape = src.attr("shape").cast<std::vector<unsigned int>>();
            value.typestr = src.attr("typestr").cast<std::string>();
           
            return true;
        }

        static handle cast(Cupy_Ref<T> src, return_value_policy /* policy */, handle /* parent */) {

            //in the previous function "load" to convert python to c++, I need to reinterpret cast a size_t of the ptr to a double*.
            //in this function, because I have to convert it from c++ to python, then I need to convert the ptr from double* to size_t 
            std::size_t python_pointer = reinterpret_cast<std::size_t>(src.ptr);
            auto cupy = module::import("cupy_ref").attr("Cupy_Ref")(python_pointer, src.dtype, src.shape, src.typestr);
            return cupy.release();
        }
    };
}}