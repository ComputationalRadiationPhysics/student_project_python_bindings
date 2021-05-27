set(bindingBidirect "bidirectCaster")

add_library(${bindingBidirect} SHARED bidirectCaster.cpp)

target_include_directories(${bindingBidirect}
  PUBLIC
  ${PYTHON_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIR})

target_link_libraries(${bindingBidirect}
  ${PYTHON_LIBRARIES}
  )

add_msvc_poperties(${bindingBidirect})
set_target_properties(${bindingBidirect} PROPERTIES PREFIX "")

# add a custom target for the test_simpleCaster.py script, which will copy
# test_simpleCaster.py to the installation folder and update every time there is a change
set(testBidirectScriptName "testBidirectCaster")
add_custom_target(${testBidirectScriptName} ALL)
add_custom_command(TARGET ${testBidirectScriptName} PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/test_bidirectCaster.py $<TARGET_FILE_DIR:${bindingBidirect}>
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_bidirectCaster.py)
add_custom_command(TARGET ${testBidirectScriptName} PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/bidirect.py $<TARGET_FILE_DIR:${bindingBidirect}>
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bidirect.py)


install(TARGETS ${bindingBidirect} DESTINATION bin)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/test_bidirectCaster.py DESTINATION bin)
