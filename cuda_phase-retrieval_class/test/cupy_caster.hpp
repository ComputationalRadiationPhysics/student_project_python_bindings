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

template<> std::string cupy_ref_get_dtype<float>(){ return "float32";}
template<> std::string cupy_ref_get_dtype<double>(){ return "float64";}
template<> std::string cupy_ref_get_dtype<std::uint16_t>(){ return "uint16";}
template<> std::string cupy_ref_get_dtype<std::uint32_t>(){ return "uint32";}
template<> std::string cupy_ref_get_dtype<std::complex<float>>(){ return "complex64";}
template<> std::string cupy_ref_get_dtype<std::complex<double>>(){ return "complex128";}

namespace pybind11 { namespace detail {
    template <typename T, int TDim> struct type_caster<Custom_Cupy_Ref<T, TDim>> 
    {
        // "PYBIND11_TYPE_CASTER" is not working with 2 types template (line 31) because there is a bug in the  macro
        // https://github.com/pybind/pybind11/blob/b5357d1fa8e91ddbfbc2ad057b9a1ccb74becbba/include/pybind11/cast.h#L88
        // pybind11 version : 2.6.1
        // public:
        //     PYBIND11_TYPE_CASTER(Custom_Cupy_Ref<T, TDim>, _("cupy_ref.Custom_Cupy_Ref"));
        
        protected:                                                                                        
            Custom_Cupy_Ref<T, TDim> value;                                                                                   
                                                                                                        
        public:                                                                                          
            static constexpr auto name = _("cupy_ref.Custom_Cupy_Ref");                                                         
            template <typename T_, enable_if_t<std::is_same<Custom_Cupy_Ref<T, TDim>, remove_cv_t<T_>>::value, int> = 0>      
            static handle cast(T_ *src, return_value_policy policy, handle parent) {                      
                if (!src)                                                                                 
                    return none().release();                                                              
                if (policy == return_value_policy::take_ownership) {                                      
                    auto h = cast(std::move(*src), policy, parent);                                       
                    delete src;                                                                           
                    return h;                                                                             
                }                                                                                         
                return cast(*src, policy, parent);                                                        
            }                                                                                             
            operator Custom_Cupy_Ref<T, TDim> *() { return &value; }                                                          
            operator Custom_Cupy_Ref<T, TDim> &() { return value; }                                                           
            operator Custom_Cupy_Ref<T, TDim> &&() && { return std::move(value); } 
                                                    
        template <typename T_>                                                                        
        using cast_op_type = pybind11::detail::movable_cast_op_type<T_>;
      
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