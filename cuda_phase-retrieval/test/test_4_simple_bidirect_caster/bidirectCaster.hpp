#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "bidirect.hpp"

namespace pybind11 { namespace detail {
    template <> struct type_caster<data_container> {
    public:
        /**
         * This macro establishes the name 'data_container' in
         * function signatures and declares a local variable
         * 'value' of type data_container
         */
        PYBIND11_TYPE_CASTER(data_container, _("bidirect.data_container"));

        /**
         * Conversion part 1 (Python->C++): convert a PyObject into a data_container
         * instance or return false upon failure. The second argument
         * indicates whether implicit conversions should be applied.
         */
        bool load(handle src, bool) {
	        // TODO: implement
            if(!hasattr(src, "data") && !hasattr(src, "scalar"))
            {
                 return false;
            }
            
            value.data = src.attr("data").cast<std::vector<double>>();
            value.scalar = src.attr("scalar").cast<double>();
           
            return true;
        }

        /**
         * Conversion part 2 (C++ -> Python): convert an data_container instance into
         * a Python bidirect.data_container obeject. The second and third arguments are
	 * used to indicate the return value policy and parent object (for
         * ``return_value_policy::reference_internal``) and are generally
         * ignored by implicit casters.
         */
        static handle cast(data_container src, return_value_policy /* policy */, handle /* parent */) {
            // TODO: implement the correct cast
            // The return value PyLong_FromLong is only used, that it compiles

            auto data_container = module::import("bidirect").attr("data_container")(src.data, src.scalar);
            return data_container.release();

            //return PyLong_FromLong(1);

        }
    };
}} // namespace pybind11::detail
