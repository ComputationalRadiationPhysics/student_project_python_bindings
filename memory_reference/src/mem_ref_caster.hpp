#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdio>
#include <iostream>
#include "tags.hpp"

using namespace std::literals::complex_literals;
using namespace pybind11::literals;

namespace pybind11 { namespace detail {
    template <> struct type_caster<Mem_Ref<CPU>> 
    {
        // using Mem_Ref_t = Mem_Ref<CPU>;

        public:
            PYBIND11_TYPE_CASTER(Mem_Ref<CPU>, _("mem_ref.Mem_RefCPU"));
      
        // python -> C++
        bool load(handle src, bool)
        {
            if(!hasattr(src, "data"))
            {
                 return false;
            }
            
            pybind11::array_t<float, pybind11::array::c_style> numpy_data = (pybind11::array_t<float, pybind11::array::c_style>)src.attr("data");
            int size = src.attr("data").attr("size").cast<int>();

            pybind11::buffer_info numpy_data_buffer = numpy_data.request();
            float *c_data = static_cast<float*>(numpy_data_buffer.ptr);

            value.numpy_data = numpy_data;
            value.size = size;
            value.c_data = c_data;

            return true;
        }

        static handle cast(Mem_Ref<CPU> src, return_value_policy /* policy */, handle /* parent */) {

            auto data = module::import("mem_ref").attr("Mem_RefCPU")(src.numpy_data);
            return data.release();
        }
    };
}}