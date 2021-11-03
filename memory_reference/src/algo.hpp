#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "tags.hpp"
#include "mem_ref.hpp"
#include "mem_ref_caster.hpp"

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
    using TAcc = CPU;

    // float * input;
    // float * output;

    Mem_Ref<TAcc> input;
    Mem_Ref<TAcc> output;
    int size;

    void whoami(){
    std::cout << "I'm the CPU version\n";
    }

    Algo(){}

    void initialize_array(int size)
    {
        input =  Mem_Ref<TAcc>(size, 0.0);
        output = Mem_Ref<TAcc>(size, 0.0);
    }

    Mem_Ref<TAcc> get_input_memory()
    {
        // input as numpy_array or similar
        return input;
    }

    Mem_Ref<TAcc> get_output_memory(int size)
    {
        // output as numpy_array or similar
        return output;
    }
    void compute(Mem_Ref<TAcc> input, Mem_Ref<TAcc> output)
    {
        for(int i = 0; i < size; ++i)
        {
            output.get_c_data()[i] = 2 * input.get_c_data()[i];
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