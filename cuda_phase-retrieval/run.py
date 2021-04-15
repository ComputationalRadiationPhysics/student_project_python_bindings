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
parser.add_argument("--mask",  help="Initial mask value (default 1). The mask is created automatically and has the size of the input image. All values of the mask are initialized with this value.", type=float, default=1)
parser.add_argument("--step",  help="Number of iterations (default 20)", type=int, default=20)
parser.add_argument("--beta",  help="Beta (default 0.8)", type=float, default=0.8)
parser.add_argument("--mode",  help="hybrid, input-ouput, or output-output", default="hybrid", choices=['hybrid', 'input-ouput', 'output-output'])
parser.add_argument("--type",  help="Language used to run phase retrieval. cuda (default) or python. ", default="cuda", choices=['cuda', 'python'])
args = parser.parse_args()

if(args.image is None):
    print("Please input image path using --image")
    print("Use -h for help")
    exit()

#np.random.seed(1)
image = imageio.imread(args.image, as_gray=True)
step = args.step
beta = args.beta
type = args.type.lower() 
mode = args.mode.lower()
mask = np.full(image.shape, args.mask)
array_random = np.random.rand(*image.shape) #uniform random

assert step > 0
assert beta > 0

print("Running phase retrieval...")

t1_start = perf_counter()

if(type == "python"):
    result = phase_retrieval_python.fienup_phase_retrieval(image, mask, step, mode, beta, array_random)
elif(type == "cuda"):
    result =  cuPhaseRet.fienup_phase_retrieval(image, mask, step, mode, beta, array_random)

t1_stop = perf_counter() 
print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)

plt.show()
plt.subplot(221)
plt.imshow(image, cmap='gray')
plt.title('Imput Image')
plt.subplot(222)
plt.imshow(result, cmap='gray')
plt.title('Phase Retrieval')
# on headless systems, maximizing the window could be a problem
try:
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
except:
    # simply ignore it, if maximizing is not possible
    pass
plt.show()
