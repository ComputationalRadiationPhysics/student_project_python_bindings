cmake_minimum_required(VERSION 3.13)

set(_targetName binding)
add_library(${_targetName} SHARED binding.cpp)

target_link_libraries(${_targetName}
  PUBLIC
  cupyref::headers
  ${PYTHON_LIBRARIES}
  )

if(TARGET cpyref::cuda)
  target_link_libraries(${_targetName} PUBLIC cpyref::cuda)
  set_cuda_language(${_targetName})
endif()

if(ENABLE_HIP)
  target_link_libraries(${_targetName} PUBLIC cpyref::hip)
endif()

add_custom_target(genexdebug COMMAND ${CMAKE_COMMAND} -E echo "$<TARGET_PROPERTY:binding,SOURCES>")

add_msvc_poperties(${_targetName})

set_target_properties(${_targetName} PROPERTIES PREFIX "")

set(python_files
  ${PROJECT_SOURCE_DIR}/include/cupy_ref.py
  ${PROJECT_SOURCE_DIR}/include/algogpucuda.py
  ${PROJECT_SOURCE_DIR}/include/algogpuhip.py
  ${PROJECT_SOURCE_DIR}/include/hip_mem.py
  ${PROJECT_SOURCE_DIR}/include/hip_ref.py
  ${CMAKE_CURRENT_SOURCE_DIR}/main.py
  )

copy_python_files(${_targetName} "${python_files}")
