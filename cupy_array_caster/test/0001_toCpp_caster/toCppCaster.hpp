#include <pybind11/pybind11.h>
#include "toCpp.hpp"

namespace pybind11 { namespace detail {
    template <> struct type_caster<toCpp> {
    public:
        /**
         * This macro establishes the name 'toCpp' in
         * function signatures and declares a local variable
         * 'value' of type toCpp
         */
        PYBIND11_TYPE_CASTER(toCpp, _("toCpp.toCpp"));

        /**
         * Conversion part 1 (Python->C++): convert a PyObject into a toCpp
         * instance or return false upon failure. The second argument
         * indicates whether implicit conversions should be applied.
         */
        bool load(handle src, bool) {
	    // TODO: implement cast function

            return false;
        }

    };
}} // namespace pybind11::detail