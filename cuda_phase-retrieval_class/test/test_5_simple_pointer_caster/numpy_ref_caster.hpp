#include <pybind11/pybind11.h>
#include "numpy_ref.hpp"

namespace pybind11 { namespace detail {
    template <> struct type_caster<Simple_Numpy_Ref> {
    public:
        /**
         * This macro establishes the name 'Simple_Numpy_Ref' in
         * function signatures and declares a local variable
         * 'value' of type Simple_Numpy_Ref
         */
        PYBIND11_TYPE_CASTER(Simple_Numpy_Ref, _("numpy_ref.Simple_Numpy_Ref"));

        /**
         * Conversion part 1 (Python->C++): convert a PyObject into a toCpp
         * instance or return false upon failure. The second argument
         * indicates whether implicit conversions should be applied.
         */
        bool load(handle src, bool) {
	    // TODO: implement

           if(!hasattr(src, "ptr"))
            {
                 return false;
            }
            
            value.ptr = reinterpret_cast<std::uint32_t *>(src.attr("ptr").cast<size_t>());
           
            return true;
        }

    };
}} // namespace pybind11::detail

namespace pybind11 { namespace detail {
    template <> struct type_caster<Advanced_Numpy_Ref> {
    public:
        /**
         * This macro establishes the name 'Simple_Numpy_Ref' in
         * function signatures and declares a local variable
         * 'value' of type Simple_Numpy_Ref
         */
        PYBIND11_TYPE_CASTER(Advanced_Numpy_Ref, _("numpy_ref.Advanced_Numpy_Ref"));

        /**
         * Conversion part 1 (Python->C++): convert a PyObject into a toCpp
         * instance or return false upon failure. The second argument
         * indicates whether implicit conversions should be applied.
         */
        bool load(handle src, bool) {
	    // TODO: implement

           if(!hasattr(src, "ptr") && !hasattr(src, "size"))
            {
                 return false;
            }
            
            value.ptr = reinterpret_cast<std::uint32_t *>(src.attr("ptr").cast<size_t>());
            value.size = src.attr("size").cast<size_t>();
           
            return true;
        }

    };
}} // namespace pybind11::detail
