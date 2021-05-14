#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <cstdio>
#include <time.h>
#include <iostream>
#include <complex>
#include <string>
#include <random>

#define CUDA_CHECK(call) {cudaError_t error = call; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}
using namespace std;
using namespace std::literals::complex_literals;
namespace py = pybind11;



// class cupy_array
// {
//     public:

//     size_t gpu_adress;
//     size_t size;
//     double* casted_value;

//     cupy_array(){};
//     cupy_array(size_t gpu_adress){
//         this->gpu_adress = gpu_adress;
//         cast_double();
//     }
//     cupy_array(size_t gpu_adress, size_t size){
//         this->gpu_adress = gpu_adress;
//         this->size = size;
//     }
    

//     void cast_double()
//     {
//         this->casted_value = reinterpret_cast<double*>(this->gpu_adress);
//     }

//     int getSize()
//     {
//         return this->size;
//     }

//     size_t getAdress()
//     {
//         return this->gpu_adress;
//     }


// };

// namespace pybind11 { namespace detail {
//     template <> struct type_caster<cupy_array> {
//     public:
//         /**
//          * This macro establishes the name 'inty' in
//          * function signatures and declares a local variable
//          * 'value' of type inty
//          */
//         PYBIND11_TYPE_CASTER(cupy_array, _("cupy_array"));

//         /**
//          * Conversion part 1 (Python->C++): convert a PyObject into a inty
//          * instance or return false upon failure. The second argument
//          * indicates whether implicit conversions should be applied.
//          */
//         bool load(handle src, bool) {
//             /* Extract PyObject from handle */
//             PyObject *source = src.ptr();
//             /* Try converting into a Python integer value */
//             PyObject *tmp = PyNumber_Long(source);
//             if (!tmp)
//                 return false;
//             /* Now try to convert into a C++ int */
//             value.gpu_adress = PyLong_AsLong(tmp);
//             value.cast_double();

//             Py_DECREF(tmp);
//             /* Ensure return code was OK (to avoid out-of-range errors etc) */
//             return !(value.gpu_adress == -1 && !PyErr_Occurred());
//         }

//         /**
//          * Conversion part 2 (C++ -> Python): convert an inty instance into
//          * a Python object. The second and third arguments are used to
//          * indicate the return value policy and parent object (for
//          * ``return_value_policy::reference_internal``) and are generally
//          * ignored by implicit casters.
//          */
//         static handle cast(cupy_array src, return_value_policy /* policy */, handle /* parent */) {
//             return PyLong_FromLong(src.gpu_adress);
//         }
//     };
// }} // namespace py