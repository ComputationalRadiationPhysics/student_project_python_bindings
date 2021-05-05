import cupy as cp
import numpy as np
import random
import gpuMemManagement

mempool = cp.get_default_memory_pool()

def free_gpu_memory(number_of_gpus):
  for x in range(0,number_of_gpus):
    cp.cuda.Device(x).use()
    mempool.free_all_blocks()

if __name__ == "__main__":

  # input data on CPU
  number_of_elements = 23
  number_of_gpus = gpuMemManagement.getDeviceNumber()
  images = np.zeros(number_of_elements)
  for i in range(0, number_of_elements):
    images[i] = i

  #cpu initial data
  update = random.random()
  partial_update = np.zeros(number_of_elements)

  print("Number of Device : ", number_of_gpus)

  #4th try
  print("First part, copy image to device")
  gpu_image = cp.zeros(number_of_elements)
  print("Before")
  print(type(gpu_image), gpu_image)

  gpuMemManagement.copy_to_device(gpu_image.data.ptr, images, number_of_elements)
  print("After")
  print(type(gpu_image), gpu_image) 
  
  print("\nSecond part, do partial update")
  gpu_partial_update = cp.asarray(partial_update) #initial array of zero in gpu
  print("Before")
  print(type(gpu_partial_update), gpu_partial_update)

  gpuMemManagement.update_images_v4(gpu_image.data.ptr, gpu_partial_update.data.ptr, update, number_of_elements)
  print("After")
  print(type(gpu_partial_update), gpu_partial_update)

  free_gpu_memory(number_of_gpus)

  

