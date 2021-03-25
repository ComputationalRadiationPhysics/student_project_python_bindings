import cuPhaseRet_Test #c++ pybind phase retrieval test version
import numpy as np
import imageio
import matplotlib.pyplot as plt

#still cannot find a way to use pytest with numpy, so I do it manually
image = imageio.imread('../example_images/a.png', as_gray=True)
magnitudes = np.abs(np.fft.fft2(image))
mask = np.ones(magnitudes.shape) #default mask
array_random = np.random.rand(*magnitudes.shape)

#test 1. Send array of double to c++, copy the value, receive it again (without CUDA)
array_double = cuPhaseRet_Test.array_check(image)
print("Test 1 Array of double received from C++ code is the same as initial python image value : ", np.array_equal(image, array_double))
print("Test 1 Error : ", np.testing.assert_array_equal(image, array_double, "some elements are not equal", verbose=True))

#test 2. Send array of double to c++, copy the value with CUDA, receive it again
array_double = cuPhaseRet_Test.array_check_cuda(image)
print("Test 2 Array of double received from CUDA code is the same as initial python image value : ", np.array_equal(image, array_double))
print("Test 2 Error : ", np.testing.assert_array_equal(image, array_double, "some elements are not equal", verbose=True))

#test 3. Send array of complex number to c++, copy the value, receive it again (without CUDA)
array_complex = cuPhaseRet_Test.array_check_complex(magnitudes)
print("Test 3 Array of complex received from C++ code is the same as initial python magnitudes value : ", np.array_equal(magnitudes, array_complex))
print("Test 3 Error : ", np.testing.assert_array_equal(magnitudes, array_complex, "some elements are not equal", verbose=True))

#test 4. Send array of complex number to c++, copy the value with CUDA, receive it again
array_complex = cuPhaseRet_Test.array_check_complex_cuda(magnitudes)
print("Test 4 Array of complex received from CUDA code is the same as initial python magnitudes value : ", np.array_equal(magnitudes, array_complex))
print("Test 4 Error : ", np.testing.assert_array_equal(magnitudes, array_complex, "some elements are not equal", verbose=True))

#test 5. Test to see if doing CUDA IFFT and then doing the CUDA FFT in the magnitunes value will return very close the original magnitunes
#still searching a way to get ifft then fft CUDA return the exact same value, if possible
cuda_ifft_fft_magnitudes = cuPhaseRet_Test.cufft_inverse_forward(magnitudes)
print("Test 5 Array of complex received from CUDA IFFT then FFT is very close to the initial python magnitudes value : ", np.allclose(magnitudes, cuda_ifft_fft_magnitudes, rtol=1e-5, atol=0))
print("Test 5 Error : ", np.testing.assert_allclose(magnitudes, cuda_ifft_fft_magnitudes, rtol=1e-5, atol=0, err_msg="some elements are not close", verbose=True))

#test 5. Test to see if absolute value of complex number that comes from CUDA FFT of the image is very close to the initial python magnitudes value
abs_cufft_image = cuPhaseRet_Test.abs_cufft_forward(image)
print("Test 6 Absolute value of complex number with CUDA FFT is very close to the initial python magnitudes value : ", np.allclose(magnitudes, abs_cufft_image, rtol=1e-5, atol=0))
print("Test 6 Error : ", np.testing.assert_allclose(magnitudes, abs_cufft_image, rtol=1e-5, atol=0, err_msg="some elements are not close", verbose=True))
