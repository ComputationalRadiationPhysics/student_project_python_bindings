cmake_minimum_required(VERSION 3.13)

set(_targetName pointerCaster)
add_library(${_targetName} SHARED pointerCaster.cpp)

target_link_libraries(${_targetName}
  PUBLIC
  cupyref::headers
  ${PYTHON_LIBRARIES}
  ${OpenCV_LIBS}
  )

add_msvc_poperties(${_targetName})

set_target_properties(${_targetName} PROPERTIES PREFIX "")

set(python_files
  ${CMAKE_CURRENT_SOURCE_DIR}/test_pointerCaster.py
  ${CMAKE_CURRENT_SOURCE_DIR}/numpy_ref.py
  )

copy_python_files(${_targetName} "${python_files}")

