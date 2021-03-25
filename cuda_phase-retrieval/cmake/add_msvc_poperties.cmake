# this target set ups some properties, which are required for Visual Studio
# on windows, the build can be done with the Visual Studio or ninja generator
function(add_msvc_poperties tgt)
  if(MSVC)
    set_target_properties(${tgt} PROPERTIES SUFFIX ".pyd")
  endif()

  if("${CMAKE_GENERATOR}" MATCHES "Visual Studio*")
    set_target_properties(${tgt} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} )
    set_target_properties(${tgt} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR} )
    set_target_properties(${tgt} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR} )
  endif()
endfunction()
