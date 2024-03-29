cmake_minimum_required(VERSION 3.13)

set(_targetName Test_Cupy)
add_library(${_targetName} SHARED test_cupy.cu)

target_link_libraries(${_targetName}
  PUBLIC
  cupyref::headers
  ${PYTHON_LIBRARIES}
  cudart
  cufft
  curand)

add_msvc_poperties(${_targetName})

set_target_properties(${_targetName} PROPERTIES PREFIX "")

set(python_files
  ${PROJECT_SOURCE_DIR}/include/cupy_ref.py
  ${CMAKE_CURRENT_SOURCE_DIR}/test_cupy.py
  )

copy_python_files(${_targetName} "${python_files}")
add_test(NAME ${_targetName} COMMAND pytest)
