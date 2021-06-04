set(bindingPointer "pointerCaster")

add_library(${bindingPointer} SHARED pointerCaster.cpp)

target_include_directories(${bindingPointer}
  PUBLIC
  ${PYTHON_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIR})

target_link_libraries(${bindingPointer}
  ${PYTHON_LIBRARIES}
  )

add_msvc_poperties(${bindingPointer})
set_target_properties(${bindingPointer} PROPERTIES PREFIX "")

# add a custom target for the test_simpleCaster.py script, which will copy
# test_simpleCaster.py to the installation folder and update every time there is a change
set(testPointerScriptName "testPointerCaster")
add_custom_target(${testPointerScriptName} ALL)
add_custom_command(TARGET ${testPointerScriptName} PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/test_pointerCaster.py $<TARGET_FILE_DIR:${bindingPointer}>
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_pointerCaster.py)
add_custom_command(TARGET ${testPointerScriptName} PRE_BUILD
 COMMAND ${CMAKE_COMMAND} -E copy
 ${CMAKE_CURRENT_SOURCE_DIR}/bidirectCaster.hpp $<TARGET_FILE_DIR:${bindingPointer}>
 DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bidirectCaster.hpp)


install(TARGETS ${bindingPointer} DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_pointerCaster.py DESTINATION bin)
