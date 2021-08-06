import cuPhaseRet #c++ pybind phase retrieval
import numpy as np
import imageio
import matplotlib.pyplot as plt
import phase_retrieval_python
import cupy_ref
from time import perf_counter

#np.random.seed(1)
image = imageio.imread('../example_images/a.png', as_gray=True)
array_random = np.random.rand(*image.shape) #uniform random
mask = np.ones(image.shape) #default mask, must have the same size as input image
iteration = 100

print("Running phase retrieval with " + str(iteration) + " iterations")

phase_retrieval_pybind = cuPhaseRet.Phase_Algo(image, mask, "hybrid", 0.8)

phase_retrieval_pybind.iterate_random_phase(1) #run phase retrieval once first
phase_retrieval_pybind.reset_random_phase() #reset random phase to its initial values

#start measuring time
t0_start = perf_counter()

phase_retrieval_pybind.iterate_random_phase(iteration)

#stop measuring time
t0_stop = perf_counter()
t0_elapsed = t0_stop-t0_start

result_cuda = phase_retrieval_pybind.get_result()

plt.show()
plt.subplot(221)
plt.imshow(image, cmap='gray')
plt.title('Image')
plt.subplot(222)
plt.imshow(result_cuda, cmap='gray')
plt.title('CUDA Phase Retrieval, runtime (s) : ' + str(t0_elapsed))
# on headless systems, maximizing the window could be a problem
try:
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
except:
    # simply ignore it, if maximizing is not possible
    pass
plt.show()
