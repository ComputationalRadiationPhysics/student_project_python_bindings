cmake_minimum_required(VERSION 3.13)

set(_targetName cuPhaseRet)
add_library(${_targetName} SHARED phase_retrieval.cu)

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
  ${CMAKE_CURRENT_SOURCE_DIR}/example_python.py
  ${CMAKE_CURRENT_SOURCE_DIR}/example_cuda.py
  ${CMAKE_CURRENT_SOURCE_DIR}/example_cuda_v2.py
  ${CMAKE_CURRENT_SOURCE_DIR}/example_cuda_no_random.py
  ${CMAKE_CURRENT_SOURCE_DIR}/phase_retrieval_python.py
  ${CMAKE_CURRENT_SOURCE_DIR}/run.py
  )

copy_python_files(${_targetName} "${python_files}")
