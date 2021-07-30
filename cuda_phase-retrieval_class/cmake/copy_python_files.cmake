# copies a list python files to the same location like the target
# only in the build folder
function(copy_python_files tgt python_files)
  if(NOT TARGET "pythonFileCopyTarget${tgt}")
    add_custom_target("pythonFileCopyTarget${tgt}" ALL)
  endif()

  foreach(py_file ${python_files})
    add_custom_command(TARGET "pythonFileCopyTarget${tgt}" PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
      ${py_file} $<TARGET_FILE_DIR:${tgt}>
      DEPENDS ${py_file})
  endforeach()
endfunction()
