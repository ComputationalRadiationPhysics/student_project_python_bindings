import cuPhaseRet
import numpy as np
import imageio
import matplotlib.pyplot as plt
# from phase_retrieval import fienup_phase_retrieval


np.random.seed(1)
image = imageio.imread('test.png', as_gray=True)
magnitudes = np.abs(np.fft.fft2(image))

cuPhaseRet.fienup_phase_retrieval(magnitudes, 200, False, "hybrid", 0.8)

# result = fienup_phase_retrieval(magnitudes, steps=500,
#                                 verbose=False)

# plt.show()
# plt.subplot(121)
# plt.imshow(magnitudes, cmap='gray')
# plt.title('Image')
# plt.subplot(122)
# plt.imshow(mags, cmap='gray')
# plt.title('Reconstruction')
# plt.show()
