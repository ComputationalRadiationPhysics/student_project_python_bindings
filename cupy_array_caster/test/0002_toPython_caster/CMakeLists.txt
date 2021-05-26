set(bindingToPython "toPythonCaster")

add_library(${bindingToPython} SHARED toPythonCaster.cpp)

target_include_directories(${bindingToPython}
  PUBLIC
  ${PYTHON_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIR})

target_link_libraries(${bindingToPython}
  ${PYTHON_LIBRARIES}
  )

add_msvc_poperties(${bindingToPython})
set_target_properties(${bindingToPython} PROPERTIES PREFIX "")

# add a custom target for the test_simpleCaster.py script, which will copy
# test_simpleCaster.py to the installation folder and update every time there is a change
set(testToPythonScriptName "testToPythonCaster")
add_custom_target(${testToPythonScriptName} ALL)
add_custom_command(TARGET ${testToPythonScriptName} PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/test_toPythonCaster.py $<TARGET_FILE_DIR:${bindingToPython}>
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_toPythonCaster.py)
add_custom_command(TARGET ${testToPythonScriptName} PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/toPython.py $<TARGET_FILE_DIR:${bindingToPython}>
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/toPython.py)


install(TARGETS ${bindingToPython} DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_toPythonCaster.py DESTINATION bin)
