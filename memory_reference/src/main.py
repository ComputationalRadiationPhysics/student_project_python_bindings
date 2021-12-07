from binding import *

algos = [AlgoCPU()]

if is_cuda_available() == True:
    from algogpu import AlgoGPU
    import cupy_ref
    algos.append(AlgoGPU())

if __name__ == "__main__":

    print("Devices :")
    for device in get_available_device():
        print(device[0], end = ' ')
        if(device[1] == True) : print("Ready")
        else : print("Not Ready")
    print()

    for algo in algos:
        
        algo.whoami()
        algo.initialize_array(10)

        input = algo.get_input_memory()
        output = algo.get_output_memory()

        print(input)
        print(type(input))

        for i in range(10):
            input[i] = 2.0

        algo.compute(input, output)

        print(output)
        print(type(output))