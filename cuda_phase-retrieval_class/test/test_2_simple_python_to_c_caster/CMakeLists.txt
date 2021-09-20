cmake_minimum_required(VERSION 3.13)

set(_targetName toCppCaster)
add_library(${_targetName} SHARED toCppCaster.cpp)

target_link_libraries(${_targetName}
  PUBLIC
  cupyref::headers
  ${PYTHON_LIBRARIES}
  ${OpenCV_LIBS}
  )

add_msvc_poperties(${_targetName})

set_target_properties(${_targetName} PROPERTIES PREFIX "")

set(python_files
  ${CMAKE_CURRENT_SOURCE_DIR}/test_toCppCaster.py
  ${CMAKE_CURRENT_SOURCE_DIR}/toCpp.py
  )

copy_python_files(${_targetName} "${python_files}")

