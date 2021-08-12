#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cufft.h>
#include <cuComplex.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>
#include <cstdio>
#include <time.h>
#include <iostream>
#include <complex>
#include <string>
#include <random>

#define PI 3.1415926535897932384626433
#define CUDA_CHECK(call) {cudaError_t error = call; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}
using namespace std::literals::complex_literals;
using namespace pybind11::literals;

__device__ cufftDoubleComplex gpu_exp(cufftDoubleComplex complex_number);
__device__ cufftDoubleComplex normalize(cufftDoubleComplex complex_number, int size);
__device__ cufftDoubleComplex get_complex(double real_number);
__device__ double get_real(cufftDoubleComplex complex_number);
__global__ void get_complex_array(double *real_array, cufftDoubleComplex *complex_array, int dimension);
__global__ void get_absolute_array(cufftDoubleComplex *complex_array, double *real_array , int dimension);
__global__ void init_random(double seed, curandState_t *states, int dimension);
__global__ void get_initial_random_phase(double *random_array, cufftDoubleComplex *random_phase, double *magnitude, int dimension); 
__global__ void get_initial_random_phase_curand(curandState_t *states, cufftDoubleComplex *random_phase, double *magnitude, int dimension);
__global__ void satisfy_fourier_gpu(cufftDoubleComplex *random_phase, double *magnitude, int dimension);
__global__ void process_arrays_gpu(cufftDoubleComplex *random_phase, double *mask, double *image_output, double beta, int mode, int iter, int dimension);
void CUFFT_CHECK(cufftResult cufft_process);
int get_number_of_cuda_sm();
template<typename TInputData, typename TOutputData> TOutputData * convertToCUFFT(TInputData * ptr);
template<> cufftDoubleComplex *convertToCUFFT(std::complex<double> * ptr);

/**
 * \brief CUDA Phase Retrieval primary function with random array as an additional parameter
 * \param image Input image
 * \param masks Input mask (default 1)
 * \param steps Number of iteration
 * \param mode Input mode ("input-ouput", "output-output", or "hybrid")
 * \param beta Input beta (default 0.8)
 * \param randoms Input array of random
 * \return Image result
 */

template<typename T>
class Phase_Algo
{
    private:
        //CPU class variables
        pybind11::buffer_info bufImg;  
        int mode, size_x, size_y, dimension, numSMs;
        std::size_t X, Y;
        T beta;

        //Pybind11 objects for GPU
        pybind11::object magnitude_gpu,
                         random_phase, 
                         random_phase_init,
                         mask_gpu,
                         image_output_gpu;

        //Custom GPU class variables
        Custom_Cupy_Ref<T> magnitude_gpu_cp,   //Magnitudes in GPU
                        mask_gpu_cp,           //Mask in GPU
                        image_output_gpu_cp;   //Image output in GPU

        Custom_Cupy_Ref<std::complex<T>> random_phase_cp,      //Sample random phase in GPU
                                        random_phase_init_cp;  //Store initial GPU random phase before iterations 

        cufftHandle plan; //handling FFT on gpu with CUFFT

        /**
        * \brief Change mode from string to integer
        * \param mode string version of mode
        */ 
        void set_mode(std::string mode)
        {
            if(mode.compare("hybrid") == 0) this->mode = 1;
            else if(mode.compare("input-output") == 0) this->mode = 2;
            else if(mode.compare("output-output") == 0) this->mode = 3;
        }

        /**
        * \brief Allocating memories in GPU with cupy
        */    
        void allocate_memory()
        {
            random_phase = pybind11::module::import("cupy").attr("zeros")(dimension, "dtype"_a="complex128").attr("reshape")(size_x, size_y);
            random_phase_cp = Custom_Cupy_Ref<std::complex<T>>::getCustomCupyRef(random_phase);
            
            random_phase_init = pybind11::module::import("cupy").attr("zeros")(dimension, "dtype"_a="complex128").attr("reshape")(size_x, size_y);
            random_phase_init_cp = Custom_Cupy_Ref<std::complex<T>>::getCustomCupyRef(random_phase_init);

            mask_gpu = pybind11::module::import("cupy").attr("zeros")(dimension, "dtype"_a="float64").attr("reshape")(size_x, size_y);
            mask_gpu_cp = Custom_Cupy_Ref<T>::getCustomCupyRef(mask_gpu);

            image_output_gpu = pybind11::module::import("cupy").attr("zeros")(dimension, "dtype"_a="float64").attr("reshape")(size_x, size_y);
            image_output_gpu_cp = Custom_Cupy_Ref<T>::getCustomCupyRef(image_output_gpu);
        }

        /**
        * \brief Get magntitudes
        * \param images source image
        */
        void set_magnitudes(T *images)
        {
            //Magnitude result
            magnitude_gpu = pybind11::module::import("cupy").attr("zeros")(dimension, "dtype"_a="float64").attr("reshape")(size_x, size_y);
            magnitude_gpu_cp = Custom_Cupy_Ref<T>::getCustomCupyRef(magnitude_gpu);

            //Source image in GPU 
            pybind11::object source_image_gpu = pybind11::module::import("cupy").attr("zeros")(dimension, "dtype"_a="float64").attr("reshape")(size_x, size_y);
            Custom_Cupy_Ref<T> source_image_gpu_cp = Custom_Cupy_Ref<T>::getCustomCupyRef(source_image_gpu);

            //complex number version of source image in GPU
            pybind11::object source_image_complex_gpu = pybind11::module::import("cupy").attr("zeros")(dimension, "dtype"_a="complex128").attr("reshape")(size_x, size_y);
            Custom_Cupy_Ref<std::complex<T>> source_image_complex_gpu_cp = Custom_Cupy_Ref<std::complex<T>>::getCustomCupyRef(source_image_complex_gpu);
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
                pybind11::object random_value_gpu = pybind11::module::import("cupy").attr("zeros")(dimension, "dtype"_a="float64").attr("reshape")(size_x, size_y);
                Custom_Cupy_Ref<T> random_value_gpu_cp = Custom_Cupy_Ref<T>::getCustomCupyRef(random_value_gpu);
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
        Phase_Algo(pybind11::array_t<T, pybind11::array::c_style> image, pybind11::array_t<T, pybind11::array::c_style> masks, std::string mode, T beta)
        {
            pybind11::array_t<T, pybind11::array::c_style> randoms = pybind11::array_t<T, pybind11::array::c_style>(0);
            init(image, masks, mode, beta, randoms);
        }
        
        Phase_Algo(pybind11::array_t<T, pybind11::array::c_style> image, pybind11::array_t<T, pybind11::array::c_style> masks, std::string mode, T beta, pybind11::array_t<T, pybind11::array::c_style> randoms)
        {
            init(image, masks, mode, beta, randoms);
        }

        /**
        * \brief Initialization of the phase retrieval object, called by constructor
        * \param image source image
        * \param masks mask array
        * \param mode mode for iteration (hybrid, input-output, or output-output)
        * \param beta beta
        * \param randoms Array of random, can be an empty array
        */
        void init(pybind11::array_t<T, pybind11::array::c_style> image, pybind11::array_t<T, pybind11::array::c_style> masks, std::string mode, T beta, pybind11::array_t<T, pybind11::array::c_style> randoms)
        {
            /**
            * \b Process
            * Asserting inputs
            */
            assert(beta > 0);
            assert(mode == "input-output" || mode == "output-output" || mode == "hybrid");

            set_mode(mode); //use integer instead of string for mode 
            bufImg = image.request();       
            pybind11::buffer_info bufMask = masks.request();      
           
            X = bufImg.shape[0];     //Width of image
            Y = bufImg.shape[1];     //Height of image
            this->beta = beta;

            /**
            * Asserting array size, make sure all arrays are using the same size
            */
            std::size_t mask_X = bufMask.shape[0];            //Width of mask
            std::size_t mask_Y = bufMask.shape[1];            //Height of mask
            assert(mask_X == X && mask_Y == Y);

            T *source_image = static_cast<T*>(bufImg.ptr);    //Get 1D image array
            T *mask = static_cast<T*>(bufMask.ptr);           //Mask array, same size as image 
            
            size_x = static_cast<int>(X); //Convert X to integer to prevent getting warning from CUFFT
            size_y = static_cast<int>(Y); //Convert Y to integer to prevent getting warning from CUFFT
            dimension = size_x*size_y;    //Area or dimension of image, mask, and array of random
               
            allocate_memory();
                   
            numSMs = get_number_of_cuda_sm();
         
            set_magnitudes(source_image);
            
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
        void do_cufft_inverse(Custom_Cupy_Ref<std::complex<T>> data)
        {
            cufftDoubleComplex *data_cufft = convertToCUFFT<std::complex<T>, cufftDoubleComplex>(data.ptr); 
            CUFFT_CHECK(cufftExecZ2Z(plan, data_cufft, data_cufft, CUFFT_INVERSE));
        }

        /**
        * \brief Do an FFT to a CUFFT array
        * \param data A CUFFT array inplemented in a custom GPU array
        */
        void do_cufft_forward(Custom_Cupy_Ref<std::complex<T>> data)
        {
            cufftDoubleComplex *data_cufft = convertToCUFFT<std::complex<T>, cufftDoubleComplex>(data.ptr); 
            CUFFT_CHECK(cufftExecZ2Z(plan, data_cufft, data_cufft, CUFFT_FORWARD));
        }

        /**
        * \brief Modify random phase array based on mask and mode
        * \param data A CUFFT array inplemented in a custom GPU array
        * \param iter Iteration index
        */
        void do_process_arrays(Custom_Cupy_Ref<std::complex<T>> data, int iter)
        {
            cufftDoubleComplex *data_cufft = convertToCUFFT<std::complex<T>, cufftDoubleComplex>(data.ptr); 
            process_arrays_gpu<<<8*numSMs, 256>>>(data_cufft, mask_gpu_cp.ptr, image_output_gpu_cp.ptr, beta, mode, iter, dimension);
        }

        /**
        * \brief Satisfy fourier of the random phase array
        * \param data A CUFFT array inplemented in a custom GPU array
        */
        void do_satisfy_fourier(Custom_Cupy_Ref<std::complex<T>> data)
        {
            cufftDoubleComplex *data_cufft = convertToCUFFT<std::complex<T>, cufftDoubleComplex>(data.ptr); 
            satisfy_fourier_gpu<<<8*numSMs, 256>>>(data_cufft, magnitude_gpu_cp.ptr, dimension);
        }

        /**
        * \brief Get the random phase
        * \return Random phase array in GPU implemented with a custom cupy array
        */
        Custom_Cupy_Ref<std::complex<T>> get_random_phase_custom_cupy()
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


        ~Phase_Algo()
        {
            cufftDestroy(plan);
        }
};

/**
 * \brief Calculate exponential of a double complex in GPU
 * \param complex_number A single double complex number, implemented using CUFFT library
 * \return Exponential value of the complex number
 */
__device__ cufftDoubleComplex gpu_exp(cufftDoubleComplex complex_number)
{
    cufftDoubleComplex result;
    float s, c;
    float e = expf(complex_number.x);
    sincosf(complex_number.y, &s, &c);
    result.x = c * e;
    result.y = s * e;
    return result;
}

/**
* \brief Normalize every result elements of after doing CUFFT_INVERSE
* \param complex_number A single double complex number, implemented using CUFFT library
* \param dimension Size of all arrays
* \return normalized complex number
*/
__device__ cufftDoubleComplex normalize(cufftDoubleComplex complex_number, int dimension)
{
    cufftDoubleComplex normalized_complex_number;
    normalized_complex_number.x = complex_number.x / static_cast<double>(dimension);
    normalized_complex_number.y = complex_number.y / static_cast<double>(dimension);
    
    return normalized_complex_number;
}

/**
* \brief Convert real number to CUFFT complex number, using 0 as imaginary part
* \param real_number A single double value
* \return Complex version of the double value, implemented using CUFFT
*/
__device__ cufftDoubleComplex get_complex(double real_number)
{
    cufftDoubleComplex complex_number;
    complex_number.x = real_number;
    complex_number.y = 0;

    return complex_number;
}

/**
* \brief Get real number part of a CUFFT complex number
* \param complex_number A single double complex number, implemented using CUFFT library
* \return real number part of the complex number
*/
__device__ double get_real(cufftDoubleComplex complex_number)
{
    return complex_number.x;
}

/**
* \brief Convert array of real number into array of complex number
* \param real_array Array of double real number
* \param complex_array Array of double complex number, implemented using CUFFT library
* \param dimension Size of all arrays
*/
__global__ void get_complex_array(double *real_array, cufftDoubleComplex *complex_array, int dimension)
{
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dimension; idx += blockDim.x * gridDim.x)  
    {
        cufftDoubleComplex complex_number;
        complex_number.x = real_array[idx];
        complex_number.y = 0;
        complex_array[idx] = complex_number;
    }
}

/**
* \brief Get array of absolute value from array of complex number
* \param complex_array Array of double complex number, implemented using CUFFT library
* \param real_array Array of absolute value of the complex number
* \param dimension Size of all arrays
*/
__global__ void get_absolute_array(cufftDoubleComplex *complex_array, double *real_array , int dimension)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dimension; idx += blockDim.x * gridDim.x)  
    {
        real_array[idx] = cuCabs(complex_array[idx]);
    }
}

/**
* \brief Create states for random values
* \param seed Current time in seconds (seed for randomizer)
* \param states Random states generated by for CURAND
* \param dimension Size of all arrays
*/
__global__ void init_random(double seed, curandState_t *states, int dimension)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < dimension; idx += blockDim.x * gridDim.x) 
    { 
        curand_init(seed, idx, 0, &states[idx]);
    }
}

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
* \param mode Integer version of Input mode (hybrid = 1, input-output = 2, output-output = 3)
* \param iter current iteration
* \param dimension Size of all arrays
*/
__global__ void process_arrays_gpu(cufftDoubleComplex *random_phase, double *mask, double *image_output, double beta, int mode, int iter, int dimension)
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
        if(mode == 3 || mode == 1) image_output[idx] = y;

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
            if(mode == 1 || mode == 2)
            {
                image_output[idx] = image_output_temp-beta*y;
            }
            else if(mode == 3)
            {
                image_output[idx] = y-beta*y;
            }
        }

        random_phase[idx] = get_complex(image_output[idx]);
    }
}

/**
* \brief CUFFT error checking
* \param cufft_process Result of a CUFFT operation
*/ 
void CUFFT_CHECK(cufftResult cufft_process)
{
    if(cufft_process != CUFFT_SUCCESS) std::cout<<cufft_process<<std::endl;
}

/**
* \brief Set number of Streaming Multiprocessor for CUDA Kernel
*/ 
int get_number_of_cuda_sm()
{
    int devId, numSMs;
    cudaGetDevice(&devId);
    cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, devId);
    return numSMs;
}

/**
* \brief Reinterpret a complex pointer from standard complex to CUDA FFT
* \param ptr a standard complex number
* \return CUDA FFT version of the standard complex number
*/ 
template<typename TInputData, typename TOutputData>
TOutputData * convertToCUFFT(TInputData * ptr){}

template<>
cufftDoubleComplex *convertToCUFFT(std::complex<double> * ptr)
{  
    return reinterpret_cast<cufftDoubleComplex *>(ptr);
}


