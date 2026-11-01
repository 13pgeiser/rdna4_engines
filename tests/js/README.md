# Dashboard page tests

The dashboard is one C++ string: `core/server/dashboard.cpp` embeds the whole page,
so its JavaScript has no file of its own to test. These harnesses extract the page
from the translation unit (`page.js`) and **execute the shipped functions**, sliced
by line range, against mocked DOM. A retyped copy would test the copy, not the bytes
that ship — and a sliced function that is defined but never *called* proves nothing,
which has already happened once here.

    node tests/js/poll_spark.test.js
    node tests/js/extremes.test.js

## Why these are not in ctest

They need node, which is not a build dependency of this project and will not be
present on a contributor's box or in the ROCm build image. Wiring them into ctest
would turn a missing optional tool into a red build, so they stay opt-in and are run
by hand when the page changes. `dashboard_test` covers the C++ side only.

## History

Both files came out of the 2026-11-01 dashboard review. Commit `9cc237f` claimed
"18 checks execute the shipped noteSample, triple, acceptTriple, resetWindows and
newEngineRun" while the harness lived only in `/tmp` — true when written, not
reproducible by anyone else, which an independent review correctly rejected. They are
in the tree so that claim can be re-run.
