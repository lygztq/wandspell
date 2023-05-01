include("${WandSpell_SOURCE_DIR}/cmake/Utils/cuda_executable.cmake")
add_cuda_exec(cuda_device_info "${CMAKE_CURRENT_LIST_DIR}/cuda_device_info.cu")
