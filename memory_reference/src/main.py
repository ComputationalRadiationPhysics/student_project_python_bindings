from binding import is_cuda_available
from binding import is_hip_available
from binding import get_available_device
from binding import AlgoCPU

if __name__ == "__main__":

    size = 10

    algos = [AlgoCPU()]

    if is_cuda_available() == True:
        from algogpucuda import AlgoGPUCUDA
        import cupy_ref
        algos.append(AlgoGPUCUDA())

    if is_hip_available() == True:
        from algogpuhip import AlgoGPUHIP
        algos.append(AlgoGPUHIP())
    
    print("Devices :")
    for device in get_available_device():
        print(device[0], end = ' ')
        if(device[1] == True) : print("Ready")
        else : print("Not Ready")

    print()

    for algo in algos:   
        algo.whoami()
        algo.initialize_array(size)

        input = algo.get_input_memory()
        output = algo.get_output_memory()

        if(algo.is_synced_mem() == True):
            input.read()
            input_temp = input.buffer
            output_temp = output.buffer
        else :
            input_temp = input
            output_temp = output

        print(input_temp)
        print(type(input_temp))

        for i in range(size):
            input_temp[i] = 2.0

        if(algo.is_synced_mem() == True): 
            input.write()

        algo.compute(input, output)

        if(algo.is_synced_mem() == True): 
            output.read()

        print(output_temp)
        print(type(output_temp))