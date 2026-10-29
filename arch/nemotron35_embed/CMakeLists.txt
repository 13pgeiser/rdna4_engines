if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(nemotron35_embed LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

rad_add_plugin(nemotron35_embed architectures SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/nemotron35_embed.cpp)
