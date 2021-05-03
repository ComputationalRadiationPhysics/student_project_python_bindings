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

  #4th try, 
  # first part (allocate gpu memory to python without cupy) still fail
  gpu_image = gpuMemManagement.copy_to_device(images, number_of_elements)
  # print(type(gpu_image)) #this become a single float instead of array??
  # print(gpu_image)

  #second part (but with cupy array for now until first part works)
  #still not working because illegal access
  gpu_image = cp.asarray(images)
  gpu_partial_update = cp.asarray(partial_update)
  print(gpu_partial_update) #not illegal

  gpuMemManagement.update_images_v4(gpu_image.data.ptr, gpu_partial_update.data.ptr, update, number_of_elements)
  print(type(gpu_partial_update))
  print(gpu_partial_update) #suddenly become illegal

  free_gpu_memory(number_of_gpus)

  

