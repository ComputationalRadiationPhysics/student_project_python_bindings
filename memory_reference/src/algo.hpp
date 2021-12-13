#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "tags.hpp"
#include "mem_ref_detail.hpp"
#include "dtype_getter.hpp"

#ifdef ENABLED_CUDA
#include "cupy_ref.hpp"
#include "cupy_caster.hpp"
#include "cupy_allocate.hpp"
#endif

#ifdef ENABLED_HIP
#include "hip_ref.hpp"
#include "hip_caster.hpp"
#include "hip/hip_runtime.h"
#endif

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

    bool is_synced_mem()
    {
        return false;
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

#ifdef ENABLED_CUDA
template<>
class Algo<CUDAGPU> {
public:

    pybind11::object input_allocate, output_allocate;

    int size;

    void whoami()
    {
        std::cout << "I'm the CUDA GPU version\n";
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
#endif

#ifdef ENABLED_HIP
template<>
class Algo<HIPGPU> {
public:

    void whoami()
    {
        std::cout << "I'm the HIP GPU version\n";
    }

    void compute(Mem_Ref<HIPGPU> input, Mem_Ref<HIPGPU> output, int size)
    {
        hipMemcpy(output.ptr, input.ptr, size * sizeof(double), hipMemcpyDeviceToDevice);
    }
};

class Hip_Mem_Impl {
public:

    int size;
    Hip_Mem_Impl(){}

    Mem_Ref<HIPGPU> get_hip_array(int size) 
    {
        this->size = size;
        Mem_Ref<HIPGPU> hip_array;

        std::vector<double> cpu_array;
        for(int i = 0; i < size; i++) cpu_array.push_back(0.0);

        double *device_array;
        hipMalloc((void**)&device_array, size*sizeof(double));

        hipMemcpy(device_array, cpu_array.data(), size * sizeof(double), hipMemcpyHostToDevice);

        hip_array.ptr = device_array;
        hip_array.dtype = get_dtype<double>();
        hip_array.shape.push_back(size);

        return hip_array;   
    }

    void read(pybind11::array_t<double, pybind11::array::c_style> numpy_array, Mem_Ref<HIPGPU> hip_array)
    {
        pybind11::buffer_info numpy_buffer = numpy_array.request();
        double *cpu_array = static_cast<double*>(numpy_buffer.ptr);

        hipMemcpy(cpu_array, hip_array.ptr, size * sizeof(double), hipMemcpyDeviceToHost);
    }

    void write(pybind11::array_t<double, pybind11::array::c_style> numpy_array, Mem_Ref<HIPGPU> hip_array)
    {
        pybind11::buffer_info numpy_buffer = numpy_array.request();
        double *cpu_array = static_cast<double*>(numpy_buffer.ptr);

        hipMemcpy(hip_array.ptr, cpu_array, size * sizeof(double), hipMemcpyHostToDevice);
    }
};

#endif