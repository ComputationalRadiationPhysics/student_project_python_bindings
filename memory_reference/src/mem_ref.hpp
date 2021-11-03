#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "tags.hpp"

template<typename TAcc>
class Mem_Ref;

template<>
class Mem_Ref<CPU>
{
    public:
        using type = pybind11::array_t<float, pybind11::array::c_style>;
        type numpy_data;
        float *c_data;
        int size;

        Mem_Ref(int size, float value)
        {
            this->size = size;
            numpy_data = type(size);
            pybind11::buffer_info numpy_data_buffer = numpy_data.request();
            c_data = static_cast<float*>(numpy_data_buffer.ptr);

            for(int i = 0; i < size; i++)
            {
                c_data[i] = value;
            }
        }

        float *get_c_data()
        {
            return c_data;
        }

        type get_data()
        {
            return numpy_data;
        }
    
    // ...
};


// template<>
// class Mem_Ref<CUDAGPU>{
//   using type = Cupy_Ref;
  
//   // ...
// };