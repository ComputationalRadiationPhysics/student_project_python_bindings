# About

The following instructions show, how to setup the requirements for the python binding in this folder.

# Linux

## simple_cpp

git

``` bash
  conda install -c conda-forge cmake=3.16
  conda install -c anaconda numpy pytest
  # or
  # pip install cmake numpy
```

``` bash
  # there is a bugfix (https://github.com/pybind/pybind11/pull/2240), which is not in a release yet
  git clone https://github.com/pybind/pybind11.git
  cd pybind11
  git checkout c776e9e
  mkdir build && cd build && cmake -DPYBIND11_TEST=off -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} ..
  cmake --install .
```

## simple_cuda

CUDA > 9.0


# Windows

Instructions are for the Powershell.

## simple_cpp

git

``` bash
  conda install -c conda-forge cmake=3.16
  conda install -c anaconda numpy pytest
  # or
  # pip install cmake numpy
```

``` bash
  # there is a bugfix (https://github.com/pybind/pybind11/pull/2240), which is not in a release yet
  git clone https://github.com/pybind/pybind11.git
  cd pybind11
  git checkout c776e9e
  mkdir build ; cd build ;
  cmake -DPYBIND11_TEST=off -G"Visual Studio 15 2017 Win64" -DCMAKE_PREFIX_PATH="${ENV:CONDA_PREFIX}" -DCMAKE_INSTALL_PREFIX="${ENV:CONDA_PREFIX}" ..
  cmake --install .
```


## simple_cuda

CUDA > 9.0
