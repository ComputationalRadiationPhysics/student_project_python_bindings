# Student Project Python Bindings
The student project investigates the performance and memory handling of Python bindings for CUDA C++ code created with pybind11. The original reconstruction algorithm is written in Python and can be found here: https://github.com/tuelwer/phase-retrieval

# Install

## Prerequisites

1. Please install [Miniconda3](https://docs.conda.io/en/latest/miniconda.html). Version 3.8 is recommended. It is recommended to follow the recommended options during the installation process.
2. You need to install [CUDA](https://developer.nvidia.com/cuda-toolkit) on your system. 
3. For Windows, [Microsoft Visual Studio](https://visualstudio.microsoft.com/vs/community/) is needed.
4. Open the environment.yml file, and change the version of CUDA to the same version of the CUDA version your system is using (more details in the file). 
5. If you are using Windows, you may need to use Anaconda Prompt before running commands below

## Linux

```bash
  conda env create -f environment.yml
  conda activate cuda_phase_retrieval
  mkdir build
  cd build
  # Maybe you need to set: -DPYTHON_EXECUTABLE:FILEPATH=
  cmake ..
  cmake --build .
```

## Windows

```bash
  conda env create -f environment.yml
  conda activate cuda_phase_retrieval
  mkdir build
  cd build
  cmake -G "Visual Studio 16 2019" -A x64 ..
  cmake --build .
```

# Usage

## Main Phase Retrieval Program 

The main program location is in **build/src**

```bash
  #List of supported arguments
  python run.py -h

  #Use CUDA version
  python run.py --image "../example_images/a.png" --mask 256 --beta 0.8 --step 100 --mode hybrid --type cuda

  #Use Python version
  python run.py --image "../example_images/a.png" --mask 256 --beta 0.8 --step 100 --mode hybrid --type python
```

## Other Examples

The location of the examples is in **build/src**

```bash
  #Running original phase retrieval algorithm
  python example_python.py

  #Running CUDA phase retrieval functions completely in C++
  python example_cuda.py

  #Running CUDA phase retrieval by getting the objects from C++ to python before running the algorithm in python
  python example_cuda_v2.py

  #Running CUDA phase retrieval without inputting random array and using the auto generated random array instead
  python example_cuda_no_random.py

  #Running CUDA phase retrieval, but returns cupy reference of the result instead of the result itself
  python example_cuda_custom_cupy_result.py
```

## Tests

To run the test, go to **build** folder. then run

```bash
  ctest
```

If you are using Windows, you need some extra arguments

```bash
  ctest -C Debug
```

