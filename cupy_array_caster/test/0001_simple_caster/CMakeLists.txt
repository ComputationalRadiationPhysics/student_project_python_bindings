set(bindingName "simpleCaster")

add_library(${bindingName} SHARED simpleCaster.cpp)

target_include_directories(${bindingName}
  PUBLIC
  ${PYTHON_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIR})

target_link_libraries(${bindingName}
  ${PYTHON_LIBRARIES}
  )

add_msvc_poperties(${bindingName})
set_target_properties(${bindingName} PROPERTIES PREFIX "")

# add a custom target for the test_simpleCaster.py script, which will copy
# test_simpleCaster.py to the installation folder and update every time there is a change
set(testScriptName "testSimpleCaster")
add_custom_target(${testScriptName} ALL)
add_custom_command(TARGET ${testScriptName} PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/test_simpleCaster.py $<TARGET_FILE_DIR:${bindingName}>
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_simpleCaster.py)

install(TARGETS ${bindingName} DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_simpleCaster.py DESTINATION bin)
