set(bindingToCpp "toCppCaster")

add_library(${bindingToCpp} SHARED toCppCaster.cpp)

target_include_directories(${bindingToCpp}
  PUBLIC
  ${PYTHON_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIR})

target_link_libraries(${bindingToCpp}
  ${PYTHON_LIBRARIES}
  )

add_msvc_poperties(${bindingToCpp})
set_target_properties(${bindingToCpp} PROPERTIES PREFIX "")

# add a custom target for the test_simpleCaster.py script, which will copy
# test_simpleCaster.py to the installation folder and update every time there is a change
set(testToCppScriptName "testToCppCaster")
add_custom_target(${testToCppScriptName} ALL)
add_custom_command(TARGET ${testToCppScriptName} PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/test_toCppCaster.py $<TARGET_FILE_DIR:${bindingToCpp}>
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_toCppCaster.py)
add_custom_command(TARGET ${testToCppScriptName} PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/toCpp.py $<TARGET_FILE_DIR:${bindingToCpp}>
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/toCpp.py)


install(TARGETS ${bindingToCpp} DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_toCppCaster.py DESTINATION bin)
