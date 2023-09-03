cmake_minimum_required(VERSION 3.8.0)
project(sgemm LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  message(STATUS "No CUDA support")
endif()

add_executable(cpu cpu.cpp)

