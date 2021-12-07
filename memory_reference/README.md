# Memory Reference Project

This project is to see if we can easily interchange the device used and type received by changing a single line

## Default Build
By default, numpy and CPU will be used for the simple processing
```bash
  mkdir build
  cd build
  cmake ..
  cmake --build .
```
## Enable CUDA
CUDA and cupy can be enabled on build to run the simple processing on NVIDIA GPU
```bash
  mkdir build
  cd build
  cmake -DENABLE_CUDA=ON ..
  cmake --build .
```
## Run Program
After following on of the above build steps, run
```bash
    python src\main.py
```
To run the test, simply
```bash
    pytest test
```
