import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

#need to find a way to change GPU inside this code with pyCUDA
#this code below is simple example from documentation

a = numpy.random.randn(4,4)
a = a.astype(numpy.float32, order="C")
# a_gpu = cuda.mem_alloc(a.nbytes)
# cuda.memcpy_htod(a_gpu, a)

res = numpy.empty_like(a)

mod = SourceModule("""
  __global__ void doublify(float *a, float *res)
  {
    int idx = threadIdx.x + threadIdx.y*4;
    res[idx] = a[idx] * 2;
  }
  """)

func = mod.get_function("doublify")
func(cuda.In(a), cuda.Out(res), grid = (1, 1), block=(4, 4, 1))

print(a)
print(res)

