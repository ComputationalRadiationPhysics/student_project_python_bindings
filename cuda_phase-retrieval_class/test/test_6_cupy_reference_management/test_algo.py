import cuPhaseRet_Test #c++ pybind phase retrieval test version
import numpy as np
import imageio
import matplotlib.pyplot as plt

#still cannot find a way to use pytest with numpy, so I do it manually
image = imageio.imread('../../example_images/a.png', as_gray=True)
magnitudes = np.abs(np.fft.fft2(image))
mask = np.ones(magnitudes.shape) #default mask
array_random = np.random.rand(*magnitudes.shape)

#test 1. Send array of double to c++, copy the value, receive it again (without CUDA)
def test_copy_array_c_double_to_python():
    array_double = cuPhaseRet_Test.array_check(image)
    np.testing.assert_array_equal(image, array_double, "some elements are not equal", verbose=True)

#test 2. Send array of double to c++, copy the value with CUDA, receive it again
def test_copy_array_c_double_to_python_with_cuda():
    array_double = cuPhaseRet_Test.array_check_cuda(image)
    np.testing.assert_array_equal(image, array_double, "some elements are not equal", verbose=True)

#test 3. Send array of complex number to c++, copy the value, receive it again (without CUDA)
def test_copy_array_c_complex_double_to_python():
    array_complex = cuPhaseRet_Test.array_check_complex(magnitudes)
    np.testing.assert_array_equal(magnitudes, array_complex, "some elements are not equal", verbose=True)

#test 4. Send array of complex number to c++, copy the value with CUDA, receive it again
def test_copy_array_c_complex_double_to_python_with_cuda():
    array_complex = cuPhaseRet_Test.array_check_complex_cuda(magnitudes)
    np.testing.assert_array_equal(magnitudes, array_complex, "some elements are not equal", verbose=True)

#test 5. Test to see if doing CUDA IFFT and then doing the CUDA FFT in the magnitunes value will return very close the original magnitunes
def test_cuda_fft_ifft_magnitudes():
    cuda_ifft_fft_magnitudes = cuPhaseRet_Test.cufft_inverse_forward(magnitudes)
    np.testing.assert_allclose(magnitudes, cuda_ifft_fft_magnitudes, rtol=1e-5, atol=0, err_msg="some elements are not close", verbose=True)

#test 6. Test to see if absolute value of complex number that comes from CUDA FFT of the image is very close to the initial python magnitudes value
def test_cuda_fft_images_to_magnitudes():
    abs_cufft_image = cuPhaseRet_Test.abs_cufft_forward(image)
    np.testing.assert_allclose(magnitudes, abs_cufft_image, rtol=1e-5, atol=0, err_msg="some elements are not close", verbose=True)


