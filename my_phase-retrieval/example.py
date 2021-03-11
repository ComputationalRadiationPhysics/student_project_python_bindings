import cuPhaseRet #c++ pybind phase retrieval
import numpy as np
import imageio
import matplotlib.pyplot as plt
# from phase_retrieval_original_edited import fienup_phase_retrieval
from time import perf_counter 

#np.random.seed(1)
image = imageio.imread('b.png', as_gray=True)
magnitudes = np.abs(np.fft.fft2(image))

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
