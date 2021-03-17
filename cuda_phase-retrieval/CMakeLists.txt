cmake_minimum_required(VERSION 3.11)
project(cuPhaseRet LANGUAGES CXX CUDA)

find_package(PythonInterp 3.6 REQUIRED)
find_package(PythonLibs 3.6 REQUIRED)
find_package(pybind11 REQUIRED)

set(OpenCV_STATIC ON)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIR})

add_library(cuPhaseRet SHARED phase_retrieval.cu)
target_link_libraries(cuPhaseRet
  ${PYTHON_LIBRARIES}
  ${OpenCV_LIBS}
  cudart
  cufft
  curand)

set_target_properties(cuPhaseRet PROPERTIES CUDA_STANDARD 14)
set_target_properties(cuPhaseRet PROPERTIES PREFIX "")

if(MSVC)
  set_target_properties(cuPhaseRet PROPERTIES SUFFIX ".pyd")
endif()

# on windows, the build can be done with the Visual Studio or ninja generator
# this configuration is Visual Studio specific
if("${CMAKE_GENERATOR}" MATCHES "Visual Studio*")
  set_target_properties(cuPhaseRet PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} )
  set_target_properties(cuPhaseRet PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR} )
  set_target_properties(cuPhaseRet PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR} )
endif()

# add a custom target for the test.py script, which will copy test.py
# to the installation folder and update every time there is a change
add_custom_target(pyTestScript ALL)
add_custom_command(TARGET pyTestScript PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/example.py $<TARGET_FILE_DIR:cuPhaseRet>
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/example.py)

install(TARGETS cuPhaseRet DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/example.py DESTINATION bin)

add_custom_command(TARGET pyTestScript PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/phase_retrieval_python.py $<TARGET_FILE_DIR:cuPhaseRet>
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/phase_retrieval_python.py)

install(TARGETS cuPhaseRet DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/phase_retrieval_python.py DESTINATION bin)
