cmake_minimum_required(VERSION 3.13)

if(ENABLE_TEST)
  add_subdirectory("test_1_initial_cupy_reference/")
  add_subdirectory("test_3_cupy_reference_management/")
endif()
