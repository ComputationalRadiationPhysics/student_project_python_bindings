#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "tags.hpp"
#include "mem_ref_detail.hpp"
#include "cupy_ref.hpp"
#include "cupy_caster.hpp"
#include "cupy_allocate.hpp"

template<typename TDevice>
class Algo {
    public:
        void whoami();

        Algo();
        void initialize_array(int size);
        Mem_Ref<TDevice> get_input_memory();
        void compute(Mem_Ref<TDevice> input, Mem_Ref<TDevice> output);
        Mem_Ref<TDevice> get_ouput_memory();
};

template<>
class Algo<CPU> {
public:
    pybind11::array_t<double, pybind11::array::c_style> input, output;
    int size;

    void whoami()
    {
        std::cout << "I'm the CPU version\n";
    }

    void initialize_array(int size)
    {
        this->size = size;

        input = Mem_Ref<CPU>(size);
        pybind11::buffer_info bufInput = input.request();
        double *ptrInput = static_cast<double*>(bufInput.ptr);
        
        output = Mem_Ref<CPU>(size);
        pybind11::buffer_info bufOutput = output.request();
        double *ptrOutput = static_cast<double*>(bufOutput.ptr);

        for(int i = 0; i < size; i++)
        {
            ptrOutput[i] =  0.0;
            ptrInput[i] = 0.0;
        }
    }

    Mem_Ref<CPU> get_input_memory()
    {
        return input;
        
    }

    Mem_Ref<CPU> get_output_memory()
    {
        return output;
        
    }
    void compute(Mem_Ref<CPU> input, Mem_Ref<CPU> output)
    {
        pybind11::buffer_info bufInput = input.request();
        double *ptrInput = static_cast<double*>(bufInput.ptr);

        pybind11::buffer_info bufOutput = output.request();
        double *ptrOutput = static_cast<double*>(bufOutput.ptr);

        for(int i = 0; i < size; i++)
        {
            ptrOutput[i] =  ptrInput[i];
        }
    }
};

template<>
class Algo<CUDAGPU> {
public:

    pybind11::object input_allocate, output_allocate;

    int size;

    void whoami()
    {
        std::cout << "I'm the GPU version\n";
    }

    void initialize_array(int size)
    {
        this->size = size;
        input_allocate = cupy_allocate<double>({size});
        output_allocate = cupy_allocate<double>({size});
    }

    Mem_Ref<CUDAGPU> get_input_memory() 
    {      
        return Mem_Ref<CUDAGPU>::getCupyRef(input_allocate);       
    }

    Mem_Ref<CUDAGPU> get_output_memory()
    {
        return Mem_Ref<CUDAGPU>::getCupyRef(output_allocate);
    }
    void compute(Mem_Ref<CUDAGPU> input, Mem_Ref<CUDAGPU> output)
    {
        cudaMemcpy(output.ptr, input.ptr, size * sizeof(double), cudaMemcpyDeviceToDevice);
    }
};