cmake_minimum_required(VERSION 3.11)
project(cuPhaseRet LANGUAGES CXX CUDA)

find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)
find_package(pybind11 REQUIRED)

# copy example images to build folder
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../example_images/
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/example_images/ )
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIR})

set(CMAKE_CUDA_ARCHITECTURES 35 CACHE STRING "CUDA architectures")

set(CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD ${CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_STANDARD ${CXX_STANDARD})

# copy example images to build folder
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../example_images/
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/example_images/ )

# this target set ups some properties, which are required for Visual Studio
# on windows, the build can be done with the Visual Studio or ninja generator
add_library(msvcHelper INTERFACE)
if(MSVC)
  set_target_properties(msvcHelper PROPERTIES SUFFIX ".pyd")
endif()

if("${CMAKE_GENERATOR}" MATCHES "Visual Studio*")
  set_target_properties(msvcHelper PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} )
  set_target_properties(msvcHelper PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR} )
  set_target_properties(msvcHelper PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR} )
endif()

add_library(cuPhaseRet SHARED phase_retrieval.cu)

target_include_directories(cuPhaseRet
  PUBLIC
  ${PYTHON_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIR})

target_link_libraries(cuPhaseRet
  msvcHelper
  ${PYTHON_LIBRARIES}
  ${OpenCV_LIBS}
  cudart
  cufft
  curand)

set_target_properties(cuPhaseRet PROPERTIES PREFIX "")

# add a custom target for the test.py script, which will copy test.py
# to the installation folder and update every time there is a change
add_custom_target(pyExampleScript ALL)
add_custom_command(TARGET pyExampleScript PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/example.py $<TARGET_FILE_DIR:cuPhaseRet>
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/example.py)

add_custom_command(TARGET pyExampleScript PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/phase_retrieval_python.py $<TARGET_FILE_DIR:cuPhaseRet>
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/phase_retrieval_python.py)

install(TARGETS cuPhaseRet DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/example.py DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/phase_retrieval_python.py DESTINATION bin)

add_subdirectory("test/")
