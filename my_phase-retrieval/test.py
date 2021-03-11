import cuPhaseRet #c++ pybind phase retrieval
import numpy as np
import imageio
import matplotlib.pyplot as plt

#still cannot find a way to use pytest with numpy, so I do it manually
image = imageio.imread('a.png', as_gray=True)
magnitudes = np.abs(np.fft.fft2(image))
mask = np.ones(magnitudes.shape) #default mask
array_random = np.random.rand(*magnitudes.shape)

#test 1. Send array of double to c++, copy the value, receive it again (without CUDA)
array_double = cuPhaseRet.array_check(image)
print("Test 1 Array of double received from C++ code is the same as initial python value : ", np.array_equal(image, array_double))
print("Test 1 Error : ", np.testing.assert_array_equal(image, array_double, "some elements are not equal", verbose=True))

#test 2. Send array of double to c++, copy the value with CUDA, receive it again
array_double = cuPhaseRet.array_check_cuda(image)
print("Test 2 Array of double received from CUDA code is the same as initial python value : ", np.array_equal(image, array_double))
print("Test 2 Error : ", np.testing.assert_array_equal(image, array_double, "some elements are not equal", verbose=True))

#test 3. Send array of complex number to c++, copy the value, receive it again (without CUDA)
array_complex = cuPhaseRet.array_check_complex(magnitudes)
print("Test 3 Array of complex received from C++ code is the same as initial python value : ", np.array_equal(magnitudes, array_complex))
print("Test 3 Error : ", np.testing.assert_array_equal(magnitudes, array_complex, "some elements are not equal", verbose=True))

#test 4. Send array of complex number to c++, copy the value with CUDA, receive it again
array_complex = cuPhaseRet.array_check_complex_cuda(magnitudes)
print("Test 4 Array of complex received from CUDA code is the same as initial python value : ", np.array_equal(magnitudes, array_complex))
print("Test 4 Error : ", np.testing.assert_array_equal(magnitudes, array_complex, "some elements are not equal", verbose=True))

#test 5. Test to see if doing CUDA IFFT and then doing the CUDA FFT in the magnitunes value will return very close the original magnitunes
#still searching a way to get ifft then fft CUDA return the exact same value, if possible
cuda_ifft_fft_magnitudes = cuPhaseRet.cufft_inverse_forward(magnitudes)
print("Test 5 Array of complex received from CUDA IFFT then FFT is very close to the initial python value : ", np.allclose(magnitudes, cuda_ifft_fft_magnitudes, rtol=1e-5, atol=0))
print("Test 5 Error : ", np.testing.assert_allclose(magnitudes, cuda_ifft_fft_magnitudes, rtol=1e-5, atol=0, err_msg="some elements are not close", verbose=True))

#test 6. phase retrieval using c++ random library
# result_cuda_random =  cuPhaseRet.fienup_phase_retrieval(magnitudes, mask, 10, False, "hybrid", 0.8)
result_c_random =  cuPhaseRet.fienup_phase_retrieval_c_random(magnitudes, mask, 20, True, "hybrid", 0.8)
result_numpy_random =  cuPhaseRet.fienup_phase_retrieval_numpy_random(magnitudes, mask, array_random, 20, True, "hybrid", 0.8)

plt.show()
# plt.subplot(221)
# plt.imshow(image, cmap='gray')
# plt.title('Image')
# plt.subplot(222)
# plt.imshow(result_cuda_random, cmap='gray')
# plt.title('Curand')
plt.subplot(221)
plt.imshow(result_c_random, cmap='gray')
plt.title('C++ Random')
plt.subplot(222)
plt.imshow(result_numpy_random, cmap='gray')
plt.title('Numpy Random')
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()
