cmake_minimum_required(VERSION 3.13)
project(binding LANGUAGES CXX CUDA)

include("cmake/add_msvc_poperties.cmake")
include("cmake/copy_python_files.cmake")

find_package(PythonInterp 3.8 REQUIRED)
find_package(PythonLibs 3.8 REQUIRED)
find_package(pybind11 REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 60 CACHE STRING "CUDA architectures")
# set(ENABLE_TEST ON CACHE BOOL "Enable Python binding tests")

set(CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD ${CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_STANDARD ${CXX_STANDARD})

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/include")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/src")
# if(ENABLE_TEST)
#   enable_testing("test/")
#   add_subdirectory("test/")
# endif()

# copy example images to build folder
# file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../example_images/
#   DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/example_images/ )
