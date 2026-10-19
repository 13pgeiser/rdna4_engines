# Built in radiance's tree, or standalone against an installed radiance
# (`cmake -S arch/minimaxh3 -B build -DCMAKE_PREFIX_PATH=<radiance prefix>`); see cmake/RadAddPlugin.cmake.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(minimaxh3 LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

# rad_ref2va_pack.cpp holds the Ref2VA sequence packing and rad_vae_geom.cpp the encode geometry
# (latent counts, tile plans, normalization tables) -- both pure host arithmetic, device-free, and
# unit-tested on their own (tests/ref2va_pack_test.cpp, tests/vae_geom_test.cpp compile them
# directly). The video pass consumes them as the reference-media path lands.
rad_add_plugin(minimaxh3 architectures SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/minimaxh3.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/rad_ref2va_pack.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/rad_vae_geom.cpp)
