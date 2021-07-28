cmake_minimum_required(VERSION 3.11)
project(cuPhaseRet LANGUAGES CXX CUDA)

include("cmake/add_msvc_poperties.cmake")

find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)
find_package(pybind11 REQUIRED)

# copy example images to build folder
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../example_images/
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/example_images/ )

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/includes)

set(CMAKE_CUDA_ARCHITECTURES 60 CACHE STRING "CUDA architectures")
set(ENABLE_TEST OFF CACHE BOOL "Enable Python binding tests")

set(CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD ${CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_STANDARD ${CXX_STANDARD})

add_library(cuPhaseRet SHARED phase_retrieval.cu)

target_include_directories(
  cuPhaseRet
  PUBLIC
  ${PYTHON_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIR})

target_link_libraries(
  cuPhaseRet
  ${PYTHON_LIBRARIES}
  ${OpenCV_LIBS}
  cudart
  cufft
  curand)

add_msvc_poperties(cuPhaseRet)

set_target_properties(cuPhaseRet PROPERTIES PREFIX "")

# add a custom target for the test.py script, which will copy test.py
# to the installation folder and update every time there is a change
add_custom_target(pyExampleScript ALL)
add_custom_command(TARGET pyExampleScript PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/example_python.py $<TARGET_FILE_DIR:cuPhaseRet>
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/example_python.py)

add_custom_command(TARGET pyExampleScript PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/example_cuda.py $<TARGET_FILE_DIR:cuPhaseRet>
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/example_cuda.py)

add_custom_command(TARGET pyExampleScript PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/example_cuda_v2.py $<TARGET_FILE_DIR:cuPhaseRet>
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/example_cuda_v2.py)

add_custom_command(TARGET pyExampleScript PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/phase_retrieval_python.py $<TARGET_FILE_DIR:cuPhaseRet>
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/phase_retrieval_python.py)

add_custom_command(TARGET pyExampleScript PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/run.py $<TARGET_FILE_DIR:cuPhaseRet>
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/run.py)

add_custom_command(TARGET pyExampleScript PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/phase_algo.hpp $<TARGET_FILE_DIR:cuPhaseRet>
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/phase_algo.hpp)

add_custom_command(TARGET pyExampleScript PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/includes/cupy_ref.py $<TARGET_FILE_DIR:cuPhaseRet>
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/includes/cupy_ref.py)


install(TARGETS cuPhaseRet DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/example_python.py DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/example_cuda.py DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/example_cuda_v2.py DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/phase_retrieval_python.py DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/run.py DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/phase_algo.hpp DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/includes/cupy_ref.py DESTINATION bin)

if(ENABLE_TEST)
  add_subdirectory("test/")
endif()
