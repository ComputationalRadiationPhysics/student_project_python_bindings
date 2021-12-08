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
        
        if(algo.is_synced_mem() == True):
            algo.whoami()
            algo.initialize_array(size)

            input = algo.get_input_memory()
            output = algo.get_output_memory()

            input.read()
            print(input.buffer)
            print(type(input.buffer))

            for i in range(size):
                input.buffer[i] = 2.0
            input.write()

            algo.compute(input, output)

            output.read()
            print(output.buffer)
            print(type(output.buffer))

        else:
            algo.whoami()
            algo.initialize_array(size)

            input = algo.get_input_memory()
            output = algo.get_output_memory()

            print(input)
            print(type(input))

            for i in range(size):
                input[i] = 2.0

            algo.compute(input, output)

            print(output)
            print(type(output))