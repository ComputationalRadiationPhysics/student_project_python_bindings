import cuPhaseRet #c++ pybind phase retrieval
import numpy as np
import imageio
import matplotlib.pyplot as plt
from phase_retrieval import fienup_phase_retrieval
from time import perf_counter 


np.random.seed(1)
image = imageio.imread('test.png', as_gray=True)
magnitudes = np.abs(np.fft.fft2(image))

result = cuPhaseRet.fienup_phase_retrieval(magnitudes, 2, False, "hybrid", 0.8)

# cuPhaseRet.test_fft(magnitudes)



# t1_start = perf_counter()

result2 = fienup_phase_retrieval(magnitudes, steps=2, verbose=False)

# t1_stop = perf_counter() 
# print("Elapsed time during the whole program in seconds:", t1_stop-t1_start) 

# print(magnitudes.shape)
# print(result.shape)

for i in range(2):
    for j in range(2):
        print(result[i][j]," ", result2[i][j])


# plt.show()
# plt.subplot(121)
# plt.imshow(magnitudes, cmap='gray')
# plt.title('Image')
# plt.subplot(122)
# plt.imshow(result, cmap='gray')
# plt.title('Reconstruction')
# plt.show()
