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
  print("---part 1---")
  gpu_image = gpuMemManagement.copy_to_device(images, number_of_elements)
  print(type(gpu_image)) #this become a single float instead of array??
  print(gpu_image)

  print("---part 2---")
  #second part, send a cupy to c++ via pybind (the cupy will be replaced by part 1 variables if part 1 worked)
  #for this part I send the adress of cupy image and partial update as integer
  #this part still dont use any variables from part 1
  gpu_image = cp.asarray(images) #this part will removed if part 1 is working
  gpu_partial_update = cp.asarray(partial_update) #this part will removed if part 1 is working
  print(type(gpu_partial_update), gpu_partial_update)

  gpuMemManagement.update_images_v4(gpu_image.data.ptr, gpu_partial_update.data.ptr, update, number_of_elements)

  print(type(gpu_partial_update), gpu_partial_update)

  free_gpu_memory(number_of_gpus)

  

