# Built in radiance's tree, or standalone against an installed radiance.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(hyv3_fp8 LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

rad_add_plugin(hyv3_fp8 architectures SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/hyv3_fp8.cpp)
