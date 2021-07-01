import cuPhaseRet #c++ pybind phase retrieval
import numpy as np
import imageio
import matplotlib.pyplot as plt
import phase_retrieval_python
from time import perf_counter

#np.random.seed(1)
image = imageio.imread('example_images/a.png', as_gray=True)
array_random = np.random.rand(*image.shape) #uniform random
mask = np.ones(image.shape) #default mask, must have the same size as input image

print("Running phase retrieval...")

# t1_start = perf_counter()

result_original = phase_retrieval_python.fienup_phase_retrieval(image, mask, 20, "hybrid", 0.8, array_random)
# result_cuda =  cuPhaseRet.fienup_phase_retrieval(image, mask, 20, "hybrid", 0.8, array_random)

phase_retrieval_pybind = cuPhaseRet.Phase_Algo(image, mask, "hybrid", 0.8, array_random)

phase_retrieval_pybind.iterate_random_phase(20)

result_cuda = phase_retrieval_pybind.get_result()

phase_retrieval_pybind.reset_random_phase() #if there is a need to iterate again

# t1_stop = perf_counter()
# print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)

plt.show()
plt.subplot(221)
plt.imshow(image, cmap='gray')
plt.title('Image')
plt.subplot(222)
plt.imshow(result_original, cmap='gray')
plt.title('Original Phase Retrieval')
plt.subplot(223)
plt.imshow(result_cuda, cmap='gray')
plt.title('CUDA Phase Retrieval')
# on headless systems, maximizing the window could be a problem
try:
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
except:
    # simply ignore it, if maximizing is not possible
    pass
plt.show()
