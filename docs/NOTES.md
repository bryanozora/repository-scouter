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

## Design decision — malformed tool-call handling in the M1 loop — 2026-09-28

PLAN.md's guardrails call for tool-call JSON to be "validated against a schema, with one
retry on malformed output before failing the step." For the M1 agent loop
(`app/agent/loop.py`), we simplified this: a malformed tool call (invalid JSON arguments,
or a tool name the agent invented) is turned into an `"Error: ..."` string and fed back to
the model as a normal `role="tool"` result, consuming one step of the `MAX_STEPS` budget.
The model sees the error and can self-correct on its next turn -- there's no separate
no-cost retry mechanism.

Why: it keeps the loop simple for a first working version, and reuses the same "tools
never raise, they return error strings" contract already built into `app/agent/tools.py`
rather than adding a second error-handling path. The trade-off is that a model which
repeatedly sends malformed calls burns through its step budget faster than a true
no-cost-retry design would. Revisit if M2's evaluation runs (once `report_finding` adds
its own schema validation) show this actually costing scans their step budget in
practice -- worth tightening then, not before there's evidence it matters.

## M1 end-to-end scan observations — 2026-09-28

Two real `python scan.py <url>` runs against `qwen2.5:7b-instruct` (Ollama, default settings):

| Repo | Steps used | Finished how | Notes |
|---|---|---|---|
| `octocat/Hello-World` | 12 (max) | step-limit fallback | Repeated `list_directory("")` identically all 12 steps; fallback call still produced a coherent (if thin) summary. |
| `octocat/Spoon-Knife` | 10 | model's own final answer | Repeated `list_directory("")` 9 times, then stopped and answered on step 10 -- correctly described `README.md`/`index.html`/`styles.css` by filename, but never called `read_file` once. |

Both runs met M1's "produces a sensible architecture explanation" bar, and every guardrail
(step limit, message ordering, fallback) worked as designed. But the repeated-identical-call
pattern is real: the model tends to re-list a directory instead of progressing to `read_file`,
even when the listing hasn't changed. It still converges to a plausible answer from filenames
alone rather than looping forever, so this isn't blocking -- but it means summaries can be
shallower than they should be (guessed from names, not actually read). Worth a prompt tweak
(e.g. explicitly telling the model not to repeat an identical tool call) if M2's evaluation
runs show this pattern persisting on non-trivial repos.

Per-call latency for `qwen2.5:7b-instruct` in the full loop (~12s/call, including a real
GitHub round trip and a growing conversation) is noticeably higher than the isolated
single-tool-call smoke test above (~4-5s/call) -- expected, given longer prompts and a real
tool execution in between each call, not just a raw inference benchmark.

## M1 prompt tuning: before/after re-run — 2026-09-28

Followed up on the repeat-call pattern above with three changes: (1) the system prompt and
initial user message now explicitly say the root listing is already provided, that
`list_directory("")` shouldn't be called again, and suggest reading the README + a dependency
manifest + main source files first; (2) `RepoTools.list_directory` now tracks listed paths and
returns a short "already listed" note on a repeat, mirroring `read_file`'s existing "already
read" behavior; (3) the loop itself now detects an *exact* repeated tool call (same name, same
arguments, argument-order/whitespace-insensitive) and appends a corrective hint to that tool's
result. Re-ran both repos from the run above, same model and settings:

| Repo | Steps used | Finished how | Files actually read |
|---|---|---|---|
| `octocat/Hello-World` | 5 (was 12/max) | model's own final answer (was: step-limit fallback) | `README` |
| `octocat/Spoon-Knife` | 4 (was 10) | model's own final answer | `README.md`, `index.html`, `styles.css` |

Both runs are a clear improvement. `Spoon-Knife` in particular went from never calling
`read_file` at all (guessing an answer from filenames) to reading all three of its files in
order and producing a summary actually grounded in their content. `Hello-World` no longer
exhausts its step budget: it read the README, tried two plausible-but-absent manifest
filenames (`pyproject.toml`, `requirements.txt` -- both cleanly reported as "path not found",
no crash), then repeated `list_directory("")` and `read_file("README", start=2)` once each --
each caught by the new "already listed"/"already read" notes, and the *second* identical
`list_directory("")` additionally got the loop's "you already called this" hint -- and
concluded with a reasonable summary on step 5, instead of looping to the step limit.

The model still occasionally repeats a call once before self-correcting (as `Hello-World`
shows), so this isn't a complete fix for the underlying small-model tendency -- but the two
dedup notes plus the loop-level hint are enough to reliably pull it back onto a finishing
path well within the step budget, on both repos tested.
