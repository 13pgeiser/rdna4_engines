qwen3.8-dflash2:27b
-------------------

llm-bench run --warmup 1 --concurrency 1 --openai-base-url http://localhost:9932/v1 --models openai:qwen3.8-dflash2:27b --prompts "Write a rust function to parse JSON safely."
→ Benchmarking 1 model(s) × 1 prompt(s) × 3 run(s) = 3 total requests

Results: 3 succeeded, 0 failed (100% success rate)

╭──────────┬─────────────────────┬──────────┬──────────┬───────┬──────────┬────────────┬─────────╮
│ Provider │ Model               │ P50 (ms) │ P99 (ms) │ Tok/s │ Avg Cost │ Total Cost │ Success │
├──────────┼─────────────────────┼──────────┼──────────┼───────┼──────────┼────────────┼─────────┤
│ openai   │ qwen3.8-dflash2:27b │    39411 │    39456 │   3.0 │      n/a │        n/a │    100% │
╰──────────┴─────────────────────┴──────────┴──────────┴───────┴──────────┴────────────┴─────────╯

qwen3.6-moe:35b-a3b
-------------------

llm-bench run --warmup 1 --concurrency 1 --openai-base-url http://localhost:9932/v1 --models openai:qwen3.6-moe:35b-a3b --prompts "Write a rust function to parse J
SON safely."
→ Benchmarking 1 model(s) × 1 prompt(s) × 3 run(s) = 3 total requests

Results: 3 succeeded, 0 failed (100% success rate)

╭──────────┬─────────────────────┬──────────┬──────────┬───────┬──────────┬────────────┬─────────╮
│ Provider │ Model               │ P50 (ms) │ P99 (ms) │ Tok/s │ Avg Cost │ Total Cost │ Success │
├──────────┼─────────────────────┼──────────┼──────────┼───────┼──────────┼────────────┼─────────┤
│ openai   │ qwen3.6-moe:35b-a3b │     6766 │     6957 │  11.0 │      n/a │        n/a │    100% │
╰──────────┴─────────────────────┴──────────┴──────────┴───────┴──────────┴────────────┴─────────╯

gpt-oss:20b
-----------

llm-bench run --warmup 1 --concurrency 1 --openai-base-url http://localhost:9932/v1 --models openai:gpt-oss:20b --prompts "Write a rust function to parse JSON safel
y."
→ Benchmarking 1 model(s) × 1 prompt(s) × 3 run(s) = 3 total requests

Results: 3 succeeded, 0 failed (100% success rate)

╭──────────┬─────────────┬──────────┬──────────┬───────┬──────────┬────────────┬─────────╮
│ Provider │ Model       │ P50 (ms) │ P99 (ms) │ Tok/s │ Avg Cost │ Total Cost │ Success │
├──────────┼─────────────┼──────────┼──────────┼───────┼──────────┼────────────┼─────────┤
│ openai   │ gpt-oss:20b │    29929 │    29945 │  17.1 │      n/a │        n/a │    100% │
╰──────────┴─────────────┴──────────┴──────────┴───────┴──────────┴────────────┴─────────╯

phi4-mini-it:4b
---------------

llm-bench run --warmup 1 --concurrency 1 --openai-base-url http://localhost:9932/v1 --models openai:phi4-mini-it:4b --prompts "Write a rust function to parse JSON s
afely."
→ Benchmarking 1 model(s) × 1 prompt(s) × 3 run(s) = 3 total requests

Results: 3 succeeded, 0 failed (100% success rate)

╭──────────┬─────────────────┬──────────┬──────────┬───────┬──────────┬────────────┬─────────╮
│ Provider │ Model           │ P50 (ms) │ P99 (ms) │ Tok/s │ Avg Cost │ Total Cost │ Success │
├──────────┼─────────────────┼──────────┼──────────┼───────┼──────────┼────────────┼─────────┤
│ openai   │ phi4-mini-it:4b │    20424 │    20479 │  25.0 │      n/a │        n/a │    100% │
╰──────────┴─────────────────┴──────────┴──────────┴───────┴──────────┴────────────┴─────────╯

nanbeige4.1:3b
--------------

llm-bench run --warmup 1 --concurrency 1 --openai-base-url http://localhost:9932/v1 --models openai:nanbeige4.1:3b --prompts "Write a rust function to parse JSON sa
fely."
→ Benchmarking 1 model(s) × 1 prompt(s) × 3 run(s) = 3 total requests

Results: 3 succeeded, 0 failed (100% success rate)

╭──────────┬────────────────┬──────────┬──────────┬───────┬──────────┬────────────┬─────────╮
│ Provider │ Model          │ P50 (ms) │ P99 (ms) │ Tok/s │ Avg Cost │ Total Cost │ Success │
├──────────┼────────────────┼──────────┼──────────┼───────┼──────────┼────────────┼─────────┤
│ openai   │ nanbeige4.1:3b │    18706 │    18727 │  27.4 │      n/a │        n/a │    100% │
╰──────────┴────────────────┴──────────┴──────────┴───────┴──────────┴────────────┴─────────╯

qwen3.5:9b
----------

llm-bench run --warmup 1 --concurrency 1 --openai-base-url http://localhost:9932/v1 --models openai:qwen3.5:9b --prompts "Write a rust function to parse JSON safely
."
→ Benchmarking 1 model(s) × 1 prompt(s) × 3 run(s) = 3 total requests

Results: 3 succeeded, 0 failed (100% success rate)

╭──────────┬────────────┬──────────┬──────────┬───────┬──────────┬────────────┬─────────╮
│ Provider │ Model      │ P50 (ms) │ P99 (ms) │ Tok/s │ Avg Cost │ Total Cost │ Success │
├──────────┼────────────┼──────────┼──────────┼───────┼──────────┼────────────┼─────────┤
│ openai   │ qwen3.5:9b │    45159 │    45198 │  11.3 │      n/a │        n/a │    100% │
╰──────────┴────────────┴──────────┴──────────┴───────┴──────────┴────────────┴─────────╯
