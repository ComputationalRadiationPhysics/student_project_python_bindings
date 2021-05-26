cmake_minimum_required(VERSION 3.11)
project(cupyCaster LANGUAGES CXX CUDA)

include("cmake/add_msvc_poperties.cmake")

find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIR})

set(CMAKE_CUDA_ARCHITECTURES 35 CACHE STRING "CUDA architectures")

set(CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD ${CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_STANDARD ${CXX_STANDARD})

add_subdirectory("test/")
