import cuPhaseRet #c++ pybind phase retrieval
import numpy as np
import imageio
import matplotlib.pyplot as plt
import phase_retrieval_python
import cupy_ref
from time import perf_counter

#np.random.seed(1)
image = imageio.imread('example_images/a.png', as_gray=True)
array_random = np.random.rand(*image.shape) #uniform random
mask = np.ones(image.shape) #default mask, must have the same size as input image
iteration = 100

print("Running phase retrieval with " + str(iteration) + " iterations")

t0_start = perf_counter()

phase_retrieval_pybind_v2 = cuPhaseRet.Phase_Algo(image, mask, "hybrid", 0.8, array_random)

for x in range(iteration):
    phase_retrieval_pybind_v2.do_cufft_inverse(phase_retrieval_pybind_v2.get_random_phase_custom_cupy())
    phase_retrieval_pybind_v2.do_process_arrays(phase_retrieval_pybind_v2.get_random_phase_custom_cupy(), x)
    phase_retrieval_pybind_v2.do_cufft_forward(phase_retrieval_pybind_v2.get_random_phase_custom_cupy())
    phase_retrieval_pybind_v2.do_satisfy_fourier(phase_retrieval_pybind_v2.get_random_phase_custom_cupy())

result_cuda_v2 = phase_retrieval_pybind_v2.get_result()

phase_retrieval_pybind_v2.reset_random_phase() #if there is a need to iterate again

t0_stop = perf_counter()
t0_elapsed = t0_stop-t0_start

plt.show()
plt.subplot(221)
plt.imshow(image, cmap='gray')
plt.title('Image')
plt.subplot(222)
plt.imshow(result_cuda_v2, cmap='gray')
plt.title('CUDA Phase Retrieval V2, runtime (s) : ' + str(t0_elapsed))
# on headless systems, maximizing the window could be a problem
try:
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
except:
    # simply ignore it, if maximizing is not possible
    pass
plt.show()
