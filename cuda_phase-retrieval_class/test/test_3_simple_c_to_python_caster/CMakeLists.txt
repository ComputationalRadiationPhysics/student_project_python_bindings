cmake_minimum_required(VERSION 3.13)

set(_targetName toPythonCaster)
add_library(${_targetName} SHARED toPythonCaster.cpp)

target_link_libraries(${_targetName}
  PUBLIC
  cupyref::headers
  ${PYTHON_LIBRARIES}
  ${OpenCV_LIBS}
  )

add_msvc_poperties(${_targetName})

set_target_properties(${_targetName} PROPERTIES PREFIX "")

set(python_files
  ${CMAKE_CURRENT_SOURCE_DIR}/test_toPythonCaster.py
  ${CMAKE_CURRENT_SOURCE_DIR}/toPython.py
  )

copy_python_files(${_targetName} "${python_files}")


