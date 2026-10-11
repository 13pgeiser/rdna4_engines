# gemma4k -- Gemma 4's kernels beyond libr4d: paged attention at head_dim 256/512 (gqa 1/2/4/8, any
# window), pair_argmax (the sharded draft head's cross-rank top-1), and the two sandwich-norm
# joins, which are ops of its own (docs/OPS.md). Ahead of libr4d for Gemma 4 (g4k_rows.cpp).
#
# Built inside radiance's tree (add_subdirectory(gemma4k) beside libr4d) or on its own against an
# installed radiance -- the same rad_add_plugin either way:
#   cmake -S gemma4k -B build-g4k -DCMAKE_PREFIX_PATH=/opt/radiance && cmake --build build-g4k
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(gemma4k LANGUAGES CXX HIP)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

if(NOT RAD_HIP_FOUND)
  message(STATUS "gemma4k: this radiance has no device build (RAD_HIP_FOUND is OFF) -- skipped")
  return()
endif()
# gfx12 only, as libr4d: the kernels are wave32 and read an fp8 cache with RDNA4's hardware
# conversion; the software fallback compiles elsewhere but spills.
rad_plugin_gpu_targets(_g4k_targets "^gfx12")
if(NOT _g4k_targets)
  message(STATUS "gemma4k: no gfx12 target in RAD_GPU_TARGETS (${RAD_GPU_TARGETS}) -- skipped")
  return()
endif()

rad_add_plugin(gemma4k kernels
  SOURCES     ${CMAKE_CURRENT_SOURCE_DIR}/g4k_rows.cpp
  HIP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/g4k_attn.hip ${CMAKE_CURRENT_SOURCE_DIR}/g4k_pick.hip
              ${CMAKE_CURRENT_SOURCE_DIR}/g4k_norm.hip ${CMAKE_CURRENT_SOURCE_DIR}/g4k_escale.hip
  GPU_TARGETS ${_g4k_targets})
string(REPLACE ";" " " _g4k_desc "${_g4k_targets}")
target_compile_definitions(gemma4k PRIVATE "G4K_BUILD_TARGET=\"${_g4k_desc}\"")
set_target_properties(gemma4k PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
