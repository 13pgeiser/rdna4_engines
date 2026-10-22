# libquant -- the built-in quantisers (spec §4.5). Host C++ only.
#
# Links rad_abi and nothing else, like every plugin: abi/rad_quant.h is the whole contract, which
# keeps it substitutable by a third-party quantiser.
# Built in radiance's tree, or standalone against an installed radiance
# (`cmake -S libquant -B build -DCMAKE_PREFIX_PATH=<radiance prefix>`); see cmake/RadAddPlugin.cmake.
if(NOT COMMAND rad_add_plugin)
  cmake_minimum_required(VERSION 3.21)
  project(libquant LANGUAGES CXX)
  find_package(radiance REQUIRED)
  rad_plugin_defaults()
endif()

file(GLOB LQ_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

rad_add_plugin(libquant quantizers SOURCES ${LQ_SRC} INCLUDES ${CMAKE_CURRENT_SOURCE_DIR})

# -ffp-contract=off: quantiser output is a stored file, so two builds must produce the same bytes. A
# compiler-chosen FMA can move a value on a rounding boundary into the next code.
target_compile_options(libquant PRIVATE -ffp-contract=off)

# Optional OpenMP over rows (GPTQ and round-to-nearest over large tables are long serial loops).
# Rows are independent, so the result does not depend on the thread count.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(libquant PRIVATE OpenMP::OpenMP_CXX)
else()
  message(STATUS "libquant: no OpenMP -- quantisers will run single-threaded")
endif()

# GPTQ's Hessian factorisations through LAPACK where OpenBLAS is installed (lq_gptq.cpp): the
# built-in loops are single-threaded and take hours on a 21504-wide down projection.
find_library(LQ_OPENBLAS openblas)
if(LQ_OPENBLAS)
  target_compile_definitions(libquant PRIVATE RAD_LQ_LAPACK=1)
  target_link_libraries(libquant PRIVATE ${LQ_OPENBLAS})
endif()
