macro(add_cuda_exec target_name)
  add_executable(${target_name} ${ARGN})
  message(STATUS "Configuring target: ${target_name}...")
  foreach(_src ${ARGN})
    message(STATUS "-  with src file: ${_src}")
  endforeach()
  target_include_directories(
    ${target_name}
    PRIVATE "${WandSpell_SOURCE_DIR}/3rdparty/nanologging/include"
            "${WandSpell_SOURCE_DIR}/src")
endmacro()
