import cuPhaseRet #c++ pybind phase retrieval
import numpy as np
import imageio
import matplotlib.pyplot as plt
import phase_retrieval_python
import string
from time import perf_counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image", help="Image path")
parser.add_argument("--step", help="number of iterations (default 20)", type=int, default=20)
parser.add_argument("--beta", help="beta (default 0.8)", type=float, default=0.8)
parser.add_argument("--type", help="Hybrid, Input-Ouput, or Output-Output", default="hybrid")
args = parser.parse_args()

if(args.image is None):
    print("Please input image path using --image")
    print("Use -h for help")
    exit()
    

#np.random.seed(1)
image = imageio.imread(args.image, as_gray=True)
array_random = np.random.rand(*image.shape) #uniform random
mask = np.ones(image.shape) #default mask
step = args.step
beta = args.beta
type = args.type.lower()

print("Running phase retrieval...")

t1_start = perf_counter()

result_original = phase_retrieval_python.fienup_phase_retrieval(image, mask, 20, "hybrid", 0.8, array_random)
result_cuda =  cuPhaseRet.fienup_phase_retrieval(image, mask, 20, "hybrid", 0.8, array_random)

t1_stop = perf_counter() 
print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)

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