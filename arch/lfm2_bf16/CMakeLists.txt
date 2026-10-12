# A plugin like any other: built here inside radiance's tree, or on its own against an installed
# radiance (`cmake -S arch/lfm2_bf16 -B build -DCMAKE_PREFIX_PATH=<radiance prefix>`).
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(lfm2_bf16 LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

rad_add_plugin(lfm2_bf16 architectures SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/lfm2_bf16.cpp)
