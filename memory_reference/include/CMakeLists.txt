cmake_minimum_required(VERSION 3.13)

add_library(cupyref_headers INTERFACE)
target_include_directories(cupyref_headers
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PYTHON_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIR}
  )

add_library(cupyref::headers ALIAS cupyref_headers)
