#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "tags.hpp"
#include "mem_ref_detail.hpp"

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
    Mem_Ref<CPU> input;
    Mem_Ref<CPU> output;

    int size;

    void whoami()
    {
        std::cout << "I'm the CPU version\n";
    }

    void initialize_array(int size)
    {
        this->size = size;
        input = Mem_Ref<CPU>(size);
        output = Mem_Ref<CPU>(size);
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

// template<>
// class Algo<CUDAGPU> {
// using TAcc = CUDAGPU;
// public:
//   void whoami(){
//     std::cout << "I'm the CUDA GPU version\n";
//   }

//   Mem_Ref<TAcc> get_input_memory(){
//     return ; // input as cupy_ref
//   }

//   void compute(Ref<TAcc> input, Ref<TAcc> output, int size){
//     // execute cuda kernel
//   }

//   Mem_Ref<TAcc> get_output_memory(){
//     return; // output as cupy_ref
//   }
// };