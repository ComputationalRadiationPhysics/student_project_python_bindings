# Student Project Python Bindings
The student project investigates the performance and memory handling of Python bindings for CUDA C++ code created with pybind11.
Contains the Python implementation of the algorithm as `git submodule`. The original repository is located at https://github.com/tuelwer/phase-retrieval.

# Install

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
  cmake -G "Visual Studio 16 2019" -A x64 ..
  cmake --build .
```

# Usage

```bash
  #see argument details
  python run.py -h

  #use CUDA version
  python run.py --image "../example_images/a.png" --mask 256 --beta 0.8 --step 100 --mode hybrid --type cuda

  #use Python version
  python run.py --image "../example_images/a.png" --mask 256 --beta 0.8 --step 100 --mode hybrid --type python
```
