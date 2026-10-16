# Run by ctest, not by cmake: the build's CTestTestfile.cmake includes rad_perf_tests.cmake
# (written by tests/CMakeLists.txt), which sets RAD_TEST_ENV and RAD_TEST_TIMEOUT, includes this
# file and calls rad_perf_entry once per test binary.
#
# A binary whose source declares a case named perf_... gets the entry <name>_perf, running only
# those cases, alone (RUN_SERIAL), labelled perf. Its unit entry runs everything else. The source
# is read each time ctest loads the tests, so a new perf_ case cannot be left out of both entries
# until the next configure.
function(rad_perf_entry name src bin)
  file(STRINGS "${src}" hits REGEX "^TEST\\(perf_" LIMIT_COUNT 1)
  if(hits)
    add_test(${name}_perf "${bin}" perf_)
    set_tests_properties(${name}_perf PROPERTIES
      LABELS perf
      RUN_SERIAL TRUE
      SKIP_RETURN_CODE 77
      TIMEOUT ${RAD_TEST_TIMEOUT}
      ENVIRONMENT "${RAD_TEST_ENV}")
  endif()
endfunction()
