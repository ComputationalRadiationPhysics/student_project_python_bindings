from binding import is_cuda_available
from binding import is_hip_available
from binding import get_available_device
from binding import AlgoCPU
import example_copy

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
    print("manual copy")
    example_copy.example_manual_copy(algos, 10)

    print()
    print("context copy")
    example_copy.example_context_copy(algos, 10)