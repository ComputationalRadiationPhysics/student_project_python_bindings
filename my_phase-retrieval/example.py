import cuPhaseRet #c++ pybind phase retrieval
import numpy as np
import imageio
import matplotlib.pyplot as plt
from phase_retrieval import fienup_phase_retrieval
from time import perf_counter 


np.random.seed(1)
image = imageio.imread('c.png', as_gray=True)
magnitudes = np.abs(np.fft.fft2(image))

result =  np.reshape(cuPhaseRet.fienup_phase_retrieval(magnitudes, 500, False, "hybrid", 0.8) , magnitudes.shape, "F")

# result =  cuPhaseRet.fienup_phase_retrieval(magnitudes, 500, False, "hybrid", 0.8)


# result = fienup_phase_retrieval(magnitudes, steps=500, verbose=False)

# t1_start = perf_counter()
# t1_stop = perf_counter() 
# print("Elapsed time during the whole program in seconds:", t1_stop-t1_start) 


# cuPhaseRet.test_fft(magnitudes)


plt.show()
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Image')
plt.subplot(122)
plt.imshow(result, cmap='gray')
plt.title('Reconstruction')
plt.show()
