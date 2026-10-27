# Built in radiance's tree, or standalone against an installed radiance
# (`cmake -S arch/minimaxh3 -B build -DCMAKE_PREFIX_PATH=<radiance prefix>`); see cmake/RadAddPlugin.cmake.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(minimaxh3 LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

# rad_ref2va_pack.cpp holds the reference sequence's packing and rad_vae_geom.cpp the encoders'
# geometry: pure host arithmetic, which tests/ref2va_pack_test.cpp and tests/vae_geom_test.cpp
# compile in directly.
rad_add_plugin(minimaxh3 architectures SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/minimaxh3.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/rad_ref2va_pack.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/rad_vae_geom.cpp)
