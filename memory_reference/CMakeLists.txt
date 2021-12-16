cmake_minimum_required(VERSION 3.13)

option(ENABLE_CUDA "enable cuda support" OFF)
option(ENABLE_HIP "enable hip support" OFF)
set(ENABLE_TEST ON CACHE BOOL "Enable Python binding tests")

project(binding LANGUAGES CXX)

include("cmake/add_msvc_poperties.cmake")
include("cmake/copy_python_files.cmake")
include("cmake/set_cuda_language.cmake")

find_package(PythonInterp 3.8 REQUIRED)
find_package(PythonLibs 3.8 REQUIRED)
find_package(pybind11 REQUIRED)

set(CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD ${CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(${ENABLE_CUDA} AND ${ENABLE_HIP})
  message(FATAL_ERROR "CUDA and HIP on the same time is not supported yet.")
endif()

if(ENABLE_CUDA)
  enable_language(CUDA)

  set(CMAKE_CUDA_ARCHITECTURES 60 CACHE STRING "CUDA architectures")
  set(CUDA_STANDARD ${CXX_STANDARD})

  add_library(cpyrefcuda INTERFACE)
  target_link_libraries(cpyrefcuda
    INTERFACE
    cudart
    cufft
    curand
    )
  target_compile_definitions(cpyrefcuda INTERFACE ENABLED_CUDA)
  add_library(cpyref::cuda ALIAS cpyrefcuda)
endif()

if(ENABLE_HIP)
  set(CMAKE_CXX_COMPILER hipcc)
  find_package(HIP REQUIRED)

  add_library(cpyrefhip INTERFACE)
  target_compile_definitions(cpyrefhip INTERFACE ENABLED_HIP)
  add_library(cpyref::hip ALIAS cpyrefhip)
endif()

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/include")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/src")
if(ENABLE_TEST)
  enable_testing("test/")
  add_subdirectory("test/")
endif()
