#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cufft.h>
#include <cuComplex.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cstdio>
#include <iostream>
#include <complex>

#include "cupy_ref.hpp"
#include "cupy_caster.hpp"
#include "cuda_algo.hpp"
#include "cupy_allocate.hpp"

using namespace std::literals::complex_literals;
using namespace pybind11::literals;

enum Mode { Hybrid = 1, InputOutput = 2, OutputOutput = 3};

__global__ void get_initial_random_phase(double *random_array, cufftDoubleComplex *random_phase, double *magnitude, int dimension); 
__global__ void get_initial_random_phase_curand(curandState_t *states, cufftDoubleComplex *random_phase, double *magnitude, int dimension);
__global__ void satisfy_fourier_gpu(cufftDoubleComplex *random_phase, double *magnitude, int dimension);
__global__ void process_arrays_gpu(cufftDoubleComplex *random_phase, double *mask, double *image_output, double beta, int mode, int iter, int dimension);


template<typename T>
class Phase_Algo
{
    private:
        //CPU class variables
        pybind11::buffer_info bufImg;  
        Mode phase_mode;
        int size_x, size_y, dimension, numSMs;
        std::size_t X, Y;
        T beta;

        //Pybind11 objects for GPU
        pybind11::object magnitude_gpu,
                         random_phase, 
                         random_phase_init,
                         mask_gpu,
                         image_output_gpu;

        //Custom GPU class variables
        Cupy_Ref<T> magnitude_gpu_cp,     //Magnitudes in GPU
                    mask_gpu_cp,          //Mask in GPU
                    image_output_gpu_cp;  //Image output in GPU

        Cupy_Ref<std::complex<T>> random_phase_cp,       //Sample random phase in GPU
                                  random_phase_init_cp;  //Store initial GPU random phase before iterations 

        cufftHandle plan; //handling FFT on gpu with CUFFT

        /**
        * \brief Allocating memories in GPU with cupy
        */    
        void allocate_memory()
        {
            random_phase = cupy_allocate<std::complex<T>>({size_x,size_y});
            random_phase_cp = Cupy_Ref<std::complex<T>>::getCupyRef(random_phase);
            
            random_phase_init = cupy_allocate<std::complex<T>>({size_x,size_y});
            random_phase_init_cp = Cupy_Ref<std::complex<T>>::getCupyRef(random_phase_init);

            mask_gpu = cupy_allocate<T>({size_x,size_y});
            mask_gpu_cp = Cupy_Ref<T>::getCupyRef(mask_gpu);

            image_output_gpu = cupy_allocate<T>({size_x,size_y});
            image_output_gpu_cp = Cupy_Ref<T>::getCupyRef(image_output_gpu);
        }

        /**
        * \brief Get magntitudes
        * \param images source image
        */
        void set_magnitudes(T *images)
        {
            //Magnitude result
            magnitude_gpu = cupy_allocate<T>({size_x,size_y});
            magnitude_gpu_cp = Cupy_Ref<T>::getCupyRef(magnitude_gpu);

            //Source image in GPU 
            pybind11::object source_image_gpu = cupy_allocate<T>({size_x,size_y});
            Cupy_Ref<T> source_image_gpu_cp = Cupy_Ref<T>::getCupyRef(source_image_gpu);

            //complex number version of source image in GPU
            pybind11::object source_image_complex_gpu = cupy_allocate<std::complex<T>>({size_x,size_y});
            Cupy_Ref<std::complex<T>> source_image_complex_gpu_cp = Cupy_Ref<std::complex<T>>::getCupyRef(source_image_complex_gpu);
            cufftDoubleComplex *source_image_gpu_cufft_cp = convertToCUFFT<std::complex<T>, cufftDoubleComplex>(source_image_complex_gpu_cp.ptr);

            CUDA_CHECK(cudaMemcpy(source_image_gpu_cp.ptr, images, dimension * sizeof(T), cudaMemcpyHostToDevice));

            //Convert the source image array into array of complex number
            get_complex_array<<<8*numSMs, 256>>>(source_image_gpu_cp.ptr, source_image_gpu_cufft_cp, dimension);

            //After that, do CUFFT first time to the complex source image,
            CUFFT_CHECK(cufftPlan2d(&plan, size_x, size_y, CUFFT_Z2Z));
            do_cufft_forward(source_image_complex_gpu_cp);

            //Then get the absolute value of the result, the absolute result is called magnitude
            get_absolute_array<<<8*numSMs, 256>>>(source_image_gpu_cufft_cp, magnitude_gpu_cp.ptr, dimension);
        }

        /**
        * \brief Get initial random phase, and create a copy for resetting random phase
        * \param randoms Array of random, can be an empty array
        */
        void set_random_phase(pybind11::array_t<T, pybind11::array::c_style> randoms)
        {
            cufftDoubleComplex *random_phase_cufft_cp = convertToCUFFT<std::complex<T>, cufftDoubleComplex>(random_phase_cp.ptr);
            cufftDoubleComplex *random_phase_init_cufft_cp = convertToCUFFT<std::complex<T>, cufftDoubleComplex>(random_phase_init_cp.ptr);

            pybind11::buffer_info bufRand = randoms.request();

            if(bufRand.size != 0)
            {
                std::size_t rand_X = bufRand.shape[0];            //Width of random array
                std::size_t rand_Y = bufRand.shape[1];            //Height of random array
                assert(rand_X == X && rand_Y == Y);
                T *random_value = static_cast<T*>(bufRand.ptr);   //Array of uniform random number, same size as image

                //Array of random in GPU
                pybind11::object random_value_gpu = cupy_allocate<T>({size_x,size_y});
                Cupy_Ref<T> random_value_gpu_cp = Cupy_Ref<T>::getCupyRef(random_value_gpu);
                CUDA_CHECK(cudaMemcpy(random_value_gpu_cp.ptr, random_value, dimension * sizeof(T), cudaMemcpyHostToDevice));
                get_initial_random_phase<<<8*numSMs, 256>>>(random_value_gpu_cp.ptr, random_phase_cufft_cp, magnitude_gpu_cp.ptr, dimension);
            }
            else
            {
                srand((unsigned)time( NULL ) );
                curandState_t* states;
                cudaMalloc(&states, dimension * sizeof(curandState_t));
                init_random<<<8*numSMs, 256>>>(static_cast<double>(time(0)), states, dimension);
                get_initial_random_phase_curand<<<8*numSMs, 256>>>(states, random_phase_cufft_cp, magnitude_gpu_cp.ptr, dimension);
                cudaFree(states);
            }

            CUDA_CHECK(cudaMemcpy(random_phase_init_cufft_cp, random_phase_cufft_cp, dimension * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice));
        }

    public:

        /**
        * \brief CUDA Phase Retrieval primary function
        * \param image source image
        * \param masks mask array
        * \param phase_mode mode for iteration (hybrid, input-output, or output-output)
        * \param beta beta
        */
        Phase_Algo(pybind11::array_t<T, pybind11::array::c_style> image, pybind11::array_t<T, pybind11::array::c_style> masks, Mode phase_mode, T beta)
        {
            pybind11::array_t<T, pybind11::array::c_style> randoms = pybind11::array_t<T, pybind11::array::c_style>(0);
            init(image, masks, phase_mode, beta, randoms);
        }
        
        /**
        * \brief CUDA Phase Retrieval primary function with additional random array as parameter
        * \param image source image
        * \param masks mask array
        * \param phase_mode mode for iteration (hybrid, input-output, or output-output)
        * \param beta beta
        * \param randoms Array of random, can be an empty array
        */
        Phase_Algo(pybind11::array_t<T, pybind11::array::c_style> image, pybind11::array_t<T, pybind11::array::c_style> masks, Mode phase_mode, T beta, pybind11::array_t<T, pybind11::array::c_style> randoms)
        {
            init(image, masks, phase_mode, beta, randoms);
        }

        /**
        * \brief Initialization of the phase retrieval object, called by constructor
        * \param image source image
        * \param masks mask array
        * \param phase_mode mode for iteration (hybrid, input-output, or output-output)
        * \param beta beta
        * \param randoms Array of random, can be an empty array
        */
        void init(pybind11::array_t<T, pybind11::array::c_style> image, pybind11::array_t<T, pybind11::array::c_style> masks, Mode phase_mode, T beta, pybind11::array_t<T, pybind11::array::c_style> randoms)
        {
            /**
            * \b Process
            */
            assert(beta > 0);  //Make sure beta is bigger than 0
            this->phase_mode = phase_mode;
            bufImg = image.request();       
            pybind11::buffer_info bufMask = masks.request();      
           
            X = bufImg.shape[0];     //Width of image
            Y = bufImg.shape[1];     //Height of image
            this->beta = beta;

            /**
            * Asserting array size, make sure mask from python has the same size as input image
            */
            std::size_t mask_X = bufMask.shape[0];            //Width of mask
            std::size_t mask_Y = bufMask.shape[1];            //Height of mask
            assert(mask_X == X && mask_Y == Y);

            T *source_image = static_cast<T*>(bufImg.ptr);    //Get 1D image array
            T *mask = static_cast<T*>(bufMask.ptr);           //Mask array, same size as image 
            
            size_x = static_cast<int>(X); //Convert X to integer to prevent getting warning from CUFFT
            size_y = static_cast<int>(Y); //Convert Y to integer to prevent getting warning from CUFFT
            dimension = size_x*size_y;    //Area or dimension of image, mask, and array of random
               
            allocate_memory(); //Allocate several GPU array using cupy
                   
            numSMs = get_number_of_cuda_sm(); //Get the number of GPU Streaming Multiprocessor
         
            set_magnitudes(source_image); //Set magnitudes using input image
            
            /**
            * Copy mask and array of random to GPU
            */
            CUDA_CHECK(cudaMemcpy(mask_gpu_cp.ptr, mask, dimension * sizeof(T), cudaMemcpyHostToDevice));
            
            set_random_phase(randoms);      
        }

        /**
        * \brief Get initial random phase, and create a copy for resetting random phase
        * \param steps number of iterations
        */
        void iterate_random_phase(int steps)
        {
            /**
            * For every iteration :\n
            * a. Do CUFFT Inverse to the random phase array\n
            * b. Processing the arrays. This will produce an image output for this iteration\n
            * c. Do normal FFT to the random phase array\n
            * d. Satisfy fourier to the random phase array\n
            */
            
            for(int iter = 0; iter < steps; iter++)
            {   
                do_cufft_inverse(random_phase_cp);
                do_process_arrays(random_phase_cp, iter);
                do_cufft_forward(random_phase_cp);
                do_satisfy_fourier(random_phase_cp);
            }
        }

        /**
        * \brief Reset random phase to its initial value
        */
        void reset_random_phase()
        {
            cufftDoubleComplex *random_phase_cufft_cp = convertToCUFFT<std::complex<T>, cufftDoubleComplex>(random_phase_cp.ptr);
            cufftDoubleComplex *random_phase_init_cufft_cp = convertToCUFFT<std::complex<T>, cufftDoubleComplex>(random_phase_init_cp.ptr);
            CUDA_CHECK(cudaMemcpy(random_phase_cufft_cp, random_phase_init_cufft_cp, dimension * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice));
        }

        /**
        * \brief Do an FFT Inverse to a CUFFT array
        * \param data A CUFFT array inplemented in a custom GPU array
        */
        void do_cufft_inverse(Cupy_Ref<std::complex<T>> data)
        {
            cufftDoubleComplex *data_cufft = convertToCUFFT<std::complex<T>, cufftDoubleComplex>(data.ptr); 
            CUFFT_CHECK(cufftExecZ2Z(plan, data_cufft, data_cufft, CUFFT_INVERSE));
        }

        /**
        * \brief Do an FFT to a CUFFT array
        * \param data A CUFFT array inplemented in a custom GPU array
        */
        void do_cufft_forward(Cupy_Ref<std::complex<T>> data)
        {
            cufftDoubleComplex *data_cufft = convertToCUFFT<std::complex<T>, cufftDoubleComplex>(data.ptr); 
            CUFFT_CHECK(cufftExecZ2Z(plan, data_cufft, data_cufft, CUFFT_FORWARD));
        }

        /**
        * \brief Modify random phase array based on mask and mode
        * \param data A CUFFT array inplemented in a custom GPU array
        * \param iter Iteration index
        */
        void do_process_arrays(Cupy_Ref<std::complex<T>> data, int iter)
        {
            cufftDoubleComplex *data_cufft = convertToCUFFT<std::complex<T>, cufftDoubleComplex>(data.ptr); 
            process_arrays_gpu<<<8*numSMs, 256>>>(data_cufft, mask_gpu_cp.ptr, image_output_gpu_cp.ptr, beta, phase_mode, iter, dimension);
        }

        /**
        * \brief Satisfy fourier of the random phase array
        * \param data A CUFFT array inplemented in a custom GPU array
        */
        void do_satisfy_fourier(Cupy_Ref<std::complex<T>> data)
        {
            cufftDoubleComplex *data_cufft = convertToCUFFT<std::complex<T>, cufftDoubleComplex>(data.ptr); 
            satisfy_fourier_gpu<<<8*numSMs, 256>>>(data_cufft, magnitude_gpu_cp.ptr, dimension);
        }

        /**
        * \brief Get the random phase
        * \return Random phase array in GPU implemented with a custom cupy array
        */
        Cupy_Ref<std::complex<T>> get_random_phase_custom_cupy()
        {
            return random_phase_cp;
        }

        /**
        * \brief Get the image result
        * \return Image result as a python numpy object
        */
        pybind11::array_t<T, pybind11::array::c_style> get_result()
        {
            /**
            * 1. Create a pybind array to store the final image result
            */
            pybind11::array_t<T, pybind11::array::c_style> result = pybind11::array_t<T, pybind11::array::c_style>(bufImg.size);
            pybind11::buffer_info bufRes = result.request();
            T *ptrRes = static_cast<T*>(bufRes.ptr);
            CUDA_CHECK(cudaMemcpy(ptrRes, image_output_gpu_cp.ptr, dimension * sizeof(T), cudaMemcpyDeviceToHost));

            /**
            * 2. Return the final result image
            */
            result.resize({X, Y});
            return result;
        }

        Cupy_Ref<T> get_custom_cupy_result()
        {
            return image_output_gpu_cp;
        }


        ~Phase_Algo()
        {
            cufftDestroy(plan);
        }
};

/**
* \brief Sample random phase using array of random from input
* \param random_array Array of random numbers with the size of dimension
* \param random_phase Random phase data with size of dimension
* \param magnitude Array of magnitudes, which is the absoulte value of the initial FFT result of input image
* \param dimension Size of all arrays
*/
__global__ void get_initial_random_phase(double *random_array, cufftDoubleComplex *random_phase, double *magnitude, int dimension) 
{
    cufftDoubleComplex complex1i, exp_target, magnitude_complex;
    complex1i.x = 0; complex1i.y = 1;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dimension; idx += blockDim.x * gridDim.x)  
    {
        magnitude_complex.x = magnitude[idx];
        magnitude_complex.y = 0;
        exp_target.x = PI*2.0*random_array[idx];
        exp_target.y = 0;
        random_phase[idx] = cuCmul(magnitude_complex, gpu_exp(cuCmul(complex1i, exp_target)));
    }
}

/**
* \brief Sample random phase using curandState_t as random value
* \param states Random states generated by for CURAND
* \param random_phase Random phase data with size of dimension
* \param magnitude Array of magnitudes, which is the absoulte value of the initial FFT result of input image
* \param dimension Size of all arrays
*/
__global__ void get_initial_random_phase_curand(curandState_t *states, cufftDoubleComplex *random_phase, double *magnitude, int dimension) 
{
    cufftDoubleComplex complex1i, exp_target, magnitude_complex;
    complex1i.x = 0; complex1i.y = 1;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dimension; idx += blockDim.x * gridDim.x)  
    {
        magnitude_complex.x = magnitude[idx];
        magnitude_complex.y = 0;
        exp_target.x = PI*2.0*curand_uniform(&states[idx]);
        exp_target.y = 0;
        random_phase[idx] = cuCmul(magnitude_complex, gpu_exp(cuCmul(complex1i, exp_target)));
    }
}

/**
* \brief Satisfy fourier domain constraints
* \param random_phase Random phase data with size of dimension
* \param magnitude Array of magnitudes, which is the absoulte value of the initial FFT result of input image
* \param dimension Size of all arrays
*/
__global__ void satisfy_fourier_gpu(cufftDoubleComplex *random_phase, double *magnitude, int dimension) 
{
    cufftDoubleComplex complex1i, exp_target, magnitude_complex;
    complex1i.x = 0; complex1i.y = 1;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dimension; idx += blockDim.x * gridDim.x)  
    {
        magnitude_complex.x = magnitude[idx];
        magnitude_complex.y = 0;
        exp_target.x = atan2(random_phase[idx].y, random_phase[idx].x); //arg = atan2(imag, real)
        exp_target.y = 0;
        random_phase[idx] = cuCmul(magnitude_complex, gpu_exp(cuCmul(complex1i, exp_target)));
    }
}

/**
* \brief Processing random phase array with mask, generating image result
* \param mask Input mask
* \param random_phase Random phase data with size of dimension
* \param image_output Image result
* \param beta Input beta
* \param phase_mode Mode (hybrid = 1, input-output = 2, output-output = 3)
* \param iter current iteration
* \param dimension Size of all arrays
*/
__global__ void process_arrays_gpu(cufftDoubleComplex *random_phase, double *mask, double *image_output, double beta, Mode phase_mode, int iter, int dimension)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dimension; idx += blockDim.x * gridDim.x)  
    {
        bool logical_not_mask;
        bool y_less_than_zero;
        bool logical_and;
        bool indices;
        double image_output_temp;

        double y = get_real(normalize(random_phase[idx], dimension)); //Get real version of normalized complex number

        /**
       * \b Process
       * 1. Get previous image based on current iteration
       */
        if(iter == 0) image_output_temp = y;
        else image_output_temp = image_output[idx];

        /**
        * 2. Updates for elements that satisfy object domain constraint
        */
        if(phase_mode == OutputOutput || phase_mode == Hybrid) image_output[idx] = y;

        /**
        * 3. Find elements that violate object domain constraints or are not masked 
        */
        if(mask[idx] <= 0) logical_not_mask = true;
        else if(mask[idx] >= 1) logical_not_mask = false;

        /**
        * 4. Check if any element y is less than zero 
        */
        if(y < 0) y_less_than_zero = true;
        else if(y >= 0) y_less_than_zero = false;

        /**
        * 5. Use "and" logical to check the "less than zero y" and the mask
        */   
        if(y_less_than_zero == true && mask[idx] >= 1) logical_and = true;
        else logical_and = false;

        /**
        * 6. Determine indices value
        */  
        if(logical_and == false && logical_not_mask == false) indices = false;
        else indices = true;

        /**
        * 7. Updates for elements that violate object domain constraints
        */ 
        if(indices == true)
        {
            if(phase_mode == Hybrid || phase_mode == InputOutput)
            {
                image_output[idx] = image_output_temp-beta*y;
            }
            else if(phase_mode == OutputOutput)
            {
                image_output[idx] = y-beta*y;
            }
        }

        random_phase[idx] = get_complex(image_output[idx]);
    }
}



