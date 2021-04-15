import cupy as cp
import numpy as np
mempool = cp.get_default_memory_pool()
      
def alloc_gpu_memory(parted_images, number_of_gpu):
  for x in range(0,number_of_gpu):
    cp.cuda.Device(x).use()
    print("Device ", x, " ready")
    limit  = parted_images[x].nbytes + 1024**2 #some extra for other variables
    if(limit < 1024**3): limit = 1024**3 #minimum 1GB
    mempool.set_limit(size=limit)

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
  num_elem = 20
  number_of_gpu = 2
  images = np.zeros(num_elem)
  for i in range(0, num_elem):
    images[i] = i+1

  #cpu initial data
  update = np.zeros(number_of_gpu)
  partial_update = np.zeros(number_of_gpu)

  #split the input image
  parted_images = np.array_split(images, number_of_gpu)

  # allocate memory for each GPU, the size is parted image size + update value size + parted update value size (?)
  # need to know how to properly set the size
  alloc_gpu_memory(parted_images, number_of_gpu)

  #allocate array to each GPU
  #GPU 1
  cp.cuda.Device(0).use()
  gpu_parted_image_0 = cp.asarray(parted_images[0])
  gpu_update_0 = cp.random.rand(1)
  gpu_partial_update_0 = cp.zeros(1)
  print(gpu_parted_image_0.device, gpu_parted_image_0)
  print("Device 0 memory used :", cp.get_default_memory_pool().used_bytes(), "bytes with limit ", cp.get_default_memory_pool().get_limit(), " bytes")

  #GPU 2
  cp.cuda.Device(1).use()
  gpu_parted_image_1 = cp.asarray(parted_images[1])
  gpu_update_1 = cp.random.rand(1)
  gpu_partial_update_1 = cp.zeros(1)
  print(gpu_parted_image_1.device, gpu_parted_image_1)
  print("Device 1 memory used :", cp.get_default_memory_pool().used_bytes(), "bytes with limit ", cp.get_default_memory_pool().get_limit(), " bytes")

  #update values
  cp.cuda.Device(0).use()
  for item in gpu_parted_image_0:
    gpu_partial_update_0[0] = item + gpu_update_0[0]

  cp.cuda.Device(1).use()
  for item in gpu_parted_image_1:
    gpu_partial_update_1[0] = item + gpu_update_1[0]
 

  #copy back to CPU
  parted_images[0] = gpu_parted_image_0.get()
  parted_images[1] = gpu_parted_image_1.get()
  update[0] = gpu_update_0[0]
  update[1] = gpu_update_1[0]
  partial_update[0] = gpu_partial_update_0[0]
  partial_update[1] = gpu_partial_update_1[0]

  #sticthing()?

  print("final parted image :\t", parted_images)
  print("final update :\t", update)
  print("final partial update :\t", partial_update)

  free_gpu_memory(number_of_gpu)
