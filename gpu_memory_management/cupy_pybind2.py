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

  #2nd try, sampe approach as run_cupy.py, separating variables in c++ -------------------------------------------
  #problem : not flexible, still finding another approach for this part
  partial_update = np.array_split(np.zeros(number_of_elements), number_of_gpus)
  print("2nd try")
  print(partial_update)

  for i in range(0, number_of_gpus):
    gpuMemManagement.copy_parted_image_to_device(parted_images[i], parted_images[i].size, i)

  for i in range(0, number_of_gpus):
    partial_update[i] = gpuMemManagement.update_images_v2(update[i], parted_images[i].size, i)

  print(partial_update)

  free_gpu_memory(number_of_gpus)

  

