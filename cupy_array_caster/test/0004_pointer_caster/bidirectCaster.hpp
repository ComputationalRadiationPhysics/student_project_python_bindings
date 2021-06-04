#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;

namespace pybind11 { namespace detail {
    template <> struct type_caster<simple_ptr> {
    public:
        /**
         * This macro establishes the name 'data_container' in
         * function signatures and declares a local variable
         * 'value' of type data_container
         */
        PYBIND11_TYPE_CASTER(simple_ptr, _("numpy.ndarray"));

        /**
         * Conversion part 1 (Python->C++): convert a PyObject into a data_container
         * instance or return false upon failure. The second argument
         * indicates whether implicit conversions should be applied.
         */
        bool load(handle src, bool) {
	        // TODO: implement
            if(!hasattr(src, "__array_interface__"))
            {
                 return false;
            }
            
            // auto address = src.attr("__array_interface__").operator[]("data").operator[](0);
            auto dict = src.attr("__array_interface__");
            auto address = dict.operator[]("data").operator[](0);
            auto hex = pybind11::module::import("builtins").attr("hex")(address);
            *value.ptr = hex.cast<int>();
           
            return true;
        }

        /**
         * Conversion part 2 (C++ -> Python): convert an data_container instance into
         * a Python bidirect.data_container obeject. The second and third arguments are
	 * used to indicate the return value policy and parent object (for
         * ``return_value_policy::reference_internal``) and are generally
         * ignored by implicit casters.
         */
        // static handle cast(simple_ptr src, return_value_policy /* policy */, handle /* parent */) {
        //     // TODO: implement the correct cast
        //     // The return value PyLong_FromLong is only used, that it compiles

        //     // auto ndarray = module::import("numpy").attr("arrange")(1);
        //     // ndarray.attr("__array_interface__['data']")[0] = src.ptr;
        //     // return ndarray.release();

        //     return PyLong_FromLong(1);

        // }
    };
}} // namespace pybind11::detail
