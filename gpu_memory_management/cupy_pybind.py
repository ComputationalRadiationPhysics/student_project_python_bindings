import cupy as cp
import numpy as np
import random
import asyncio
import gpuMemManagement

async def run_partial_update(gpu_image, parted_image, partial_update, update, size, device):
  cp.cuda.Device(device).use()
  gpuMemManagement.copy_to_device(gpu_image.data.ptr, parted_image, size, device)   
  gpuMemManagement.update_images(gpu_image.data.ptr, partial_update.data.ptr, update, size, device)
  print(type(partial_update), partial_update)

if __name__ == "__main__":

  # input data on CPU
  number_of_elements = 23
  number_of_gpus = gpuMemManagement.getDeviceNumber()
  images = np.zeros(number_of_elements)
  for i in range(0, number_of_elements):
    images[i] = i

  #cpu initial data
  update = random.random()

  print("Number of Device : ", number_of_gpus)

  parted_image = np.array_split(images, number_of_gpus)

  cp.cuda.Device(0).use()
  gpu_image_0 = cp.zeros(parted_image[0].size)
  gpu_partial_update_0 = cp.zeros(parted_image[0].size)

  cp.cuda.Device(1).use()
  gpu_image_1 = cp.zeros(parted_image[1].size)
  gpu_partial_update_1 = cp.zeros(parted_image[1].size)

  # run_partial_update(gpu_image_0, parted_image[0], gpu_partial_update_0, update, parted_image[0].size, 0)
  # run_partial_update(gpu_image_1, parted_image[1], gpu_partial_update_1, update, parted_image[1].size, 1)

  #I am not sure about this async part, but this give right outputs
  loop = asyncio.get_event_loop()
  tasks = [
      loop.create_task(run_partial_update(gpu_image_0, parted_image[0], gpu_partial_update_0, update, parted_image[0].size, 0)),
      loop.create_task(run_partial_update(gpu_image_1, parted_image[1], gpu_partial_update_1, update, parted_image[1].size, 1)),
  ]
  loop.run_until_complete(asyncio.wait(tasks))
  loop.close()

  cp.cuda.Device(0).use()
  gpuMemManagement.free_gpu_memory(gpu_image_0.data.ptr, 0)
  gpuMemManagement.free_gpu_memory(gpu_partial_update_0.data.ptr, 0)

  cp.cuda.Device(1).use()
  gpuMemManagement.free_gpu_memory(gpu_image_1.data.ptr, 1)
  gpuMemManagement.free_gpu_memory(gpu_partial_update_1.data.ptr, 1)



  

