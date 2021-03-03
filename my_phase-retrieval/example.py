import cuPhaseRet #c++ pybind phase retrieval
import numpy as np
import imageio
import matplotlib.pyplot as plt
# from phase_retrieval import fienup_phase_retrieval
from time import perf_counter 

np.random.seed(1)
image = imageio.imread('b.png', as_gray=True)
magnitudes = np.abs(np.fft.fft2(image))

#still cannot find a way to use pytest with numpy, so I do it manually
#test 1. send array of double to c++, copy the value, receive it again (without CUDA)
array_double = cuPhaseRet.array_check(image)
print("Test 1 Array of double received from C++ code is the same as initial python value : ", np.array_equal(image, array_double))
print("Test 1 Error : ", np.testing.assert_array_equal(image, array_double, "some elements are not equal", verbose=True))

#test 2. send array of double to c++, copy the value with CUDA, receive it again
array_double = cuPhaseRet.array_check_cuda(image)
print("Test 2 Array of double received from CUDA code is the same as initial python value : ", np.array_equal(image, array_double))
print("Test 2 Error : ", np.testing.assert_array_equal(image, array_double, "some elements are not equal", verbose=True))

#test 3. send array of complex number to c++, copy the value, receive it again (without CUDA)
array_complex = cuPhaseRet.array_check_complex(magnitudes)
print("Test 3 Array of complex received from C++ code is the same as initial python value : ", np.array_equal(magnitudes, array_complex))
print("Test 3 Error : ", np.testing.assert_array_equal(magnitudes, array_complex, "some elements are not equal", verbose=True))

#test 4. send array of complex number to c++, copy the value with CUDA, receive it again
array_complex = cuPhaseRet.array_check_complex_cuda(magnitudes)
print("Test 4 Array of complex received from CUDA code is the same as initial python value : ", np.array_equal(magnitudes, array_complex))
print("Test 4 Error : ", np.testing.assert_array_equal(magnitudes, array_complex, "some elements are not equal", verbose=True))

print("Running phase retrieval...")

mask = np.ones(magnitudes.shape) #default mask

t1_start = perf_counter()

result =  cuPhaseRet.fienup_phase_retrieval(magnitudes, mask, 500, False, "hybrid", 0.8)

t1_stop = perf_counter() 
print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)

plt.show()
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Image')
plt.subplot(122)
plt.imshow(result, cmap='gray')
plt.title('Reconstruction')
plt.show()
