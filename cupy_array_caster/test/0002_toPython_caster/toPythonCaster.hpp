#include <pybind11/pybind11.h>
// the header is necessary, that the std::vector can be casted to a python list
#include <pybind11/stl.h>

#include "toPython.hpp"


namespace pybind11 { namespace detail {
    template <> struct type_caster<toPython> {
    public:
        /**
         * This macro establishes the name 'toPython' in
         * function signatures and declares a local variable
         * 'value' of type toPython
         */
        PYBIND11_TYPE_CASTER(toPython, _("toPython.toPython"));

        /**
         * Conversion part 2 (C++ -> Python): convert an toPython instance into
         * a Python toPython.toPython obeject. The second and third arguments are
	 * used to indicate the return value policy and parent object (for
         * ``return_value_policy::reference_internal``) and are generally
         * ignored by implicit casters.
         */
        static handle cast(toPython src, return_value_policy /* policy */, handle /* parent */) {
	  // TODO: implement the correct cast
	  // The return value PyLong_FromLong is only used, that it compiles
	  return PyLong_FromLong(1);
        }

    };
}} // namespace pybind11::detail
