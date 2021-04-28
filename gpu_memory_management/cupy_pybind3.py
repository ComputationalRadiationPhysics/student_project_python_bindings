import cupy as cp
import numpy as np
import gpuMemManagement

mempool = cp.get_default_memory_pool()
      
# def alloc_gpu_memory(parted_images, number_of_gpus):
#   for x in range(0,number_of_gpus):
#     cp.cuda.Device(x).use()
#     limit  = parted_images[x].nbytes + 1024**2 #some extra for other variables
#     if(limit < 1024**3): limit = 1024**3 #minimum 1GB
#     mempool.set_limit(size=limit)
#     print("Device ", x, " ready, size limit ", limit/(1024**3), " GB")

# def alloc_gpu_memory(parted_images, number_of_gpus):
#   for x in range(0,number_of_gpus):
#     cp.cuda.Device(x).use()
#     cp.cuda.Memory(parted_images[x].size) #what about memory for another variables?

def free_gpu_memory(number_of_gpus):
  for x in range(0,number_of_gpus):
    cp.cuda.Device(x).use()
    mempool.free_all_blocks()

# def padding(v, fillval=np.nan):
#   lens = np.array([len(item) for item in v])
#   mask = lens[:,None] > np.arange(lens.max())
#   out = np.full(mask.shape,fillval)
#   out[mask] = np.concatenate(v)
#   return out

if __name__ == "__main__":

  # input data on CPU
  number_of_elements = 23
  number_of_gpus = gpuMemManagement.getDeviceNumber()
  images = np.zeros(number_of_elements)
  for i in range(0, number_of_elements):
    images[i] = i

  #cpu initial data
  update = np.random.rand(number_of_gpus)
  partial_update = np.zeros(number_of_elements)

  #split the input image
  parted_images = np.array_split(images, number_of_gpus)
  partial_update = np.array_split(partial_update, number_of_gpus)

  assert(np.shape(partial_update) == np.shape(parted_images))

  print(partial_update)

  # allocate memory for each GPU, the size is parted image size + update value size + parted update value size (?)
  # need to know how to properly set the size
  print("Number of Device : ", number_of_gpus)
  # alloc_gpu_memory(parted_images, number_of_gpus)

  #3rd try, do evrything in C++, while learning CUDA stream ---------------------------------------------------------------------------------
  #Problem : The plan is to send splitted array immages without iteration.
  #          But if the split is uneven, it is very difficult to convert send the partial image to c++ as numpy. 
  #          For example : (parted_images = parted_images.split(2), with total size 23)
  #          It split to 2 part with size 12 and 11. After that, it is very difficult to convert it to numpy. The function "def padding" in line 25 works,
  #          but after sending it to c++, it become a single one dimensional array with size 23. So it is like the splitting never happened
  #          Maybe pybind has a way to handle this (?)
  #          For now, the split happened in c++. CUDA stream is implemented here
        
  partial_update = np.zeros(number_of_elements)
  print("3rd try")
  print(partial_update)

  partial_update = gpuMemManagement.update_images_stream(images, update, number_of_elements)

  print(partial_update)

  free_gpu_memory(number_of_gpus)

  

