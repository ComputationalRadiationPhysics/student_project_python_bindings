# Student Project Python Bindings
The student project investigates the performance and memory handling of Python bindings for CUDA C++ code created with pybind11.
Contains the Python implementation of the algorithm as `git submodule`. The original repository is located at https://github.com/tuelwer/phase-retrieval.

# Install

## Requirements

1. Python 3 (This project is using [Anaconda 3](https://www.anaconda.com/)).
2. [Pybind11](https://anaconda.org/conda-forge/pybind11).
3. For Windows, [Microsoft Visual Studio](https://visualstudio.microsoft.com/vs/community/) is needed.
4. (Optional) [OpenCV](https://opencv.org/opencv-4-5-1/) with minimum version 4.5.1 is required to build and run the optional test.

## Linux

```bash
  mkdir build && cd build
  # maybe you need to set: -DPYTHON_EXECUTABLE:FILEPATH=
  cmake ..
  cmake --build .
```

## Windows

```bash
  mkdir build; cd build
  cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_PREFIX_PATH="<Path to Anaconda>/Anaconda3/Lib/site-packages/pybind11/share/cmake/pybind11" ..
  cmake --build .
```
## Turn on TEST program

```bash
  #Linux
  cmake -DENABLE_TEST=ON ..
  #Windows
  cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_PREFIX_PATH="D:/Anaconda3/Lib/site-packages/pybind11/share/cmake/pybind11" -DENABLE_TEST=ON ..
  cmake --build .
```

# Usage

```bash
  #see argument details
  python run.py -h

  #use CUDA version
  python run.py --image "example_images/a.png" --mask 256 --beta 0.8 --step 100 --mode hybrid --type cuda

  #use Python version
  python run.py --image "example_images/a.png" --mask 256 --beta 0.8 --step 100 --mode hybrid --type python
```