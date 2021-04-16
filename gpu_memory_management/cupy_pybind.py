import cupy as cp
import numpy as np
import gpuMemManagement

mempool = cp.get_default_memory_pool()
      
def alloc_gpu_memory(parted_images, number_of_gpu):
  for x in range(0,number_of_gpu):
    cp.cuda.Device(x).use()
    limit  = parted_images[x].nbytes + 1024**2 #some extra for other variables
    if(limit < 1024**3): limit = 1024**3 #minimum 1GB
    mempool.set_limit(size=limit)
    print("Device ", x, " ready, size limit ", limit/(1024**3), " GB")

def free_gpu_memory(number_of_gpu):
  for x in range(0,number_of_gpu):
    cp.cuda.Device(x).use()
    mempool.free_all_blocks()

if __name__ == "__main__":
  
  # Problem with using CUPY:
  # Easy to use, but cannot share a single array for different GPU, 2 GPU means 2 different variables
  # So iterative operation for multiple GPU cannot be used
  # Open for alternatives, if possible
  # This code use 2 GPU, so there is almost 2 variables for each required data

  # input data on CPU
  num_elem = 23
  number_of_gpu = gpuMemManagement.getDeviceNumber()
  images = np.zeros(num_elem)
  for i in range(0, num_elem):
    images[i] = (i+1)*i

  #cpu initial data
  update = np.random.rand(number_of_gpu)
  partial_update = np.zeros(num_elem)
  

  #split the input image
  parted_images = np.array_split(images, number_of_gpu)
  partial_update = np.array_split(partial_update, number_of_gpu)

  assert(np.shape(partial_update) == np.shape(parted_images))

  print(partial_update)

  # allocate memory for each GPU, the size is parted image size + update value size + parted update value size (?)
  # need to know how to properly set the size
  print("Number of Device : ", number_of_gpu)
  alloc_gpu_memory(parted_images, number_of_gpu)

  for i in range(0, number_of_gpu):
    partial_update[i] = gpuMemManagement.update_images(parted_images[i], update[i], parted_images[i].size, i)

  print(partial_update)
  free_gpu_memory(number_of_gpu)
