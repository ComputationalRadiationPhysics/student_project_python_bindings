cmake_minimum_required(VERSION 3.11)
project(gpuMemManagement LANGUAGES CXX CUDA)

include("cmake/add_msvc_poperties.cmake")

find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIR})

set(CMAKE_CUDA_ARCHITECTURES 35 CACHE STRING "CUDA architectures")
# set(ENABLE_TEST ON CACHE BOOL "Enable Python binding tests")

set(CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD ${CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_STANDARD ${CXX_STANDARD})

# copy example images to build folder
# file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../example_images/
#   DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/example_images/ )

add_library(gpuMemManagement SHARED cupy_pybind.cu)

target_include_directories(gpuMemManagement
  PUBLIC
  ${PYTHON_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIR})

target_link_libraries(gpuMemManagement
  ${PYTHON_LIBRARIES}
  ${OpenCV_LIBS}
  cudart
  cufft
  curand)

add_msvc_poperties(gpuMemManagement)

set_target_properties(gpuMemManagement PROPERTIES PREFIX "")

# add a custom target for the test.py script, which will copy test.py
# to the installation folder and update every time there is a change
add_custom_target(pyExampleScript ALL)
add_custom_command(TARGET pyExampleScript PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/cupy_pybind.py $<TARGET_FILE_DIR:gpuMemManagement>
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cupy_pybind.py)
  
add_custom_command(TARGET pyExampleScript PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/cuda_stream_test.py $<TARGET_FILE_DIR:gpuMemManagement>
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cuda_stream_test.py)

install(TARGETS gpuMemManagement DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cupy_pybind.py DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cuda_stream_test.py DESTINATION bin)

# if(ENABLE_TEST)
#   add_subdirectory("test/")
# endif()
