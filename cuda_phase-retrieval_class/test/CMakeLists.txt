set(OpenCV_STATIC ON)
find_package( OpenCV 4.5.1 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library(cuPhaseRet_Test SHARED test_binding.cu)

target_include_directories(cuPhaseRet_Test
  PUBLIC
  ${PYTHON_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIR})

target_link_libraries(cuPhaseRet_Test
  ${PYTHON_LIBRARIES}
  ${OpenCV_LIBS}
  cudart
  cufft
  curand)

add_msvc_poperties(cuPhaseRet_Test)

set_target_properties(cuPhaseRet_Test PROPERTIES PREFIX "")

# add a custom target for the test.py script, which will copy test.py
# to the installation folder and update every time there is a change
add_custom_target(pyTestScript ALL)
add_custom_command(TARGET pyTestScript PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/test.py $<TARGET_FILE_DIR:cuPhaseRet_Test>
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test.py)

add_custom_command(TARGET pyTestScript PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/test_cupy.py $<TARGET_FILE_DIR:cuPhaseRet_Test>
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_cupy.py)

add_custom_command(TARGET pyTestScript PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/cupy_ref.py $<TARGET_FILE_DIR:cuPhaseRet_Test>
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cupy_ref.py)

