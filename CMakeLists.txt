cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(WandSpell LANGUAGES CXX)

option(ENABLE_CUDA "whether to enable CUDA" ON)
set(CMAKE_CXX_STANDARD 17)

# msvc does not have rpath, we need to put all things into one directory
# TODO: build libraries in lib directory and copy to bin directory
if(MSVC)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

if(ENABLE_CUDA)
  # Get arch of cuda
  include("${CMAKE_CURRENT_LIST_DIR}/cmake/Utils/get_cuda_arch.cmake")
  get_cuda_arch()
  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_DEVICE_CC} CACHE STRING "The compute capability id of this cuda arch")

  # enable cuda language
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "Build with CUDA Computation Capability: ${CMAKE_CUDA_ARCHITECTURES}")
  else()
    message(FATAL_ERROR "Cannot find CUDA toolkit in this machine")
  endif()

  # set cpp standard for cuda
  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()

  # use shared cudart
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  message(STATUS "Use CUDA Compiler: ${CMAKE_CUDA_COMPILER_ID}:${CMAKE_CUDA_COMPILER_VERSION}")
  message(STATUS "CUDA Compiler Path: ${CMAKE_CUDA_COMPILER}")

  add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/src/cuda")
endif()
