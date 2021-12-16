cmake_minimum_required(VERSION 3.13)

set(_targetName Test_Mem_Ref)
add_library(${_targetName} SHARED test_mem_ref.cpp)

target_link_libraries(${_targetName}
  PUBLIC
  cupyref::headers
  ${PYTHON_LIBRARIES})

if(TARGET cpyref::cuda)
  target_link_libraries(${_targetName} PUBLIC cpyref::cuda)
  set_cuda_language(${_targetName})
endif()

if(ENABLE_HIP)
  target_link_libraries(${_targetName} PUBLIC cpyref::hip)
endif()

add_msvc_poperties(${_targetName})

set_target_properties(${_targetName} PROPERTIES PREFIX "")

set(python_files
  ${PROJECT_SOURCE_DIR}/include/cupy_ref.py
  ${CMAKE_CURRENT_SOURCE_DIR}/algogpu.py
  ${CMAKE_CURRENT_SOURCE_DIR}/test_mem_ref.py
  )

copy_python_files(${_targetName} "${python_files}")
add_test(NAME ${_targetName} COMMAND pytest)
