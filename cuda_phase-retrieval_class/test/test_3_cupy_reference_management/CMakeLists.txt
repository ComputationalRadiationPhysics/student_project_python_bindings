cmake_minimum_required(VERSION 3.13)

set(_targetName cuPhaseRet_Test)
add_library(${_targetName} SHARED test_binding.cu)

target_link_libraries(${_targetName}
  PUBLIC
  cupyref::headers
  ${PYTHON_LIBRARIES}
  ${OpenCV_LIBS}
  cudart
  cufft
  curand)

add_msvc_poperties(${_targetName})

set_target_properties(${_targetName} PROPERTIES PREFIX "")

set(python_files
  ${PROJECT_SOURCE_DIR}/include/cupy_ref.py
  ${CMAKE_CURRENT_SOURCE_DIR}/test.py
  ${CMAKE_CURRENT_SOURCE_DIR}/test_cupy.py
  )

copy_python_files(${_targetName} "${python_files}")
