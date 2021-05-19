import cupy as cp
import numpy as np
import random
import asyncio
import gpuMemManagement

async def run_partial_update(parted_image, update, device):
  cp.cuda.Device(device).use()
  gpu_image = cp.zeros(parted_image.size)
  gpu_partial_update = cp.zeros(parted_image.size)

  gpuMemManagement.copy_to_device(gpu_image.data.ptr, parted_image, parted_image.size, device)   
  gpuMemManagement.update_images(gpu_image.data.ptr, gpu_partial_update.data.ptr, update, parted_image.size, device)

  print('\n', type(gpu_partial_update), gpu_partial_update)

  host_image = cp.asnumpy(gpu_partial_update)

  gpuMemManagement.free_gpu_memory(gpu_image.data.ptr, device)
  gpuMemManagement.free_gpu_memory(gpu_partial_update.data.ptr, device)

  return host_image

async def run_async(parted_image, update, number_of_gpus):
   async_process = [run_partial_update(parted_image[i], update, i) for i in range(number_of_gpus)]
   return await asyncio.gather(*async_process)

if __name__ == "__main__":

  # input data on CPU
  number_of_elements = 47
  number_of_gpus = gpuMemManagement.getDeviceNumber()
  images = np.zeros(number_of_elements)
  for i in range(0, number_of_elements):
    images[i] = i

  #cpu initial data
  update = random.random()

  print("Number of Device : ", number_of_gpus)

  parted_image = np.array_split(images, number_of_gpus)

  loop = asyncio.get_event_loop()
  combined_partial_update = np.hstack(np.asarray(loop.run_until_complete(run_async(parted_image, update, number_of_gpus)), dtype=object))

  assert(images.size ==  combined_partial_update.size)

  print("\n", type(combined_partial_update), combined_partial_update)

  cp.cuda.Device(0).use()
  a = cp.ones(3)
  gpuMemManagement.print_details(a)



  

