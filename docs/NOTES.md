# Notes

## Tool-calling smoke test — 2026-09-21

Ollama 0.10.1, temperature 0.2, 10 attempts per scenario, one `list_directory(path)` tool, one warm-up call excluded from timings.

| Model | Scenario | Tool called | Valid JSON | Correct path | Avg s | Median s |
|---|---|---|---|---|---|---|
| `qwen2.5:7b-instruct` | short-path | 10/10 | 10/10 | 10/10 | 4.1 | 4.1 |
| `qwen2.5:7b-instruct` | nested-path | 10/10 | 10/10 | 10/10 | 4.8 | 4.7 |

`qwen3:8b` was not installed yet at the time of this run; it was tested in the run below.

## Tool-calling smoke test — 2026-09-21

Ollama 0.10.1, temperature 0.2, 10 attempts per scenario, one `list_directory(path)` tool, one warm-up call excluded from timings.

| Model | Scenario | Tool called | Valid JSON | Correct path | Avg s | Median s |
|---|---|---|---|---|---|---|
| `qwen3:8b` | short-path | 10/10 | 10/10 | 10/10 | 38.4 | 37.8 |
| `qwen3:8b` | nested-path | 10/10 | 10/10 | 10/10 | 33.2 | 33.1 |
