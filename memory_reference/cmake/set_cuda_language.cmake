# set the language property of all source files of target to CUDA
function(set_cuda_language tgt)
  get_target_property(_source_files ${tgt} SOURCES)
  foreach(_file ${_source_files})
    set_source_files_properties(${_file} PROPERTIES LANGUAGE CUDA)
  endforeach()
endfunction()
