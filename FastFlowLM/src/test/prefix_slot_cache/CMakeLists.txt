# PrefixSlotCache tests: no NPU, no XRT, so this builds and runs on Linux CI as well as Windows.
#   cmake -S src/test/prefix_slot_cache -B build-psc && cmake --build build-psc && ctest --test-dir build-psc
cmake_minimum_required(VERSION 3.22)
project(prefix_slot_cache_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

add_executable(test_prefix_slot_cache
    test_prefix_slot_cache.cpp
    "${FLM_SOURCE_DIR}/common/prefix_slot_cache.cpp")
target_include_directories(test_prefix_slot_cache PRIVATE "${FLM_SOURCE_DIR}/include")

enable_testing()
add_test(NAME test_prefix_slot_cache COMMAND test_prefix_slot_cache)
