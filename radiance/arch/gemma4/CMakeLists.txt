# gemma4 -- architecture plugin for Gemma 4's text model (Gemma4ForConditionalGeneration /
# Gemma4ForCausalLM: the 26B-A4B MoE and the dense sizes without per-layer embeddings), with a
# z-lab DFlash drafter merged at convert time.
#
# Built in radiance's tree (arch/ discovers it) or standalone against an installed radiance:
#   cmake -S arch/gemma4 -B build-gemma4 -DCMAKE_PREFIX_PATH=/opt/radiance && cmake --build build-gemma4
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(gemma4 LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

rad_add_plugin(gemma4 architectures SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/gemma4.cpp)
