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

## Design decision — local search instead of the GitHub code search API — 2026-09-29

For `search_code(query)` (M2 part A), chose to search a bounded, priority-ordered subset of
already-fetched-or-fetchable files locally, rather than call GitHub's `/search/code` endpoint.

**Why not GitHub's code search API:**
- Its rate limit is far stricter than the main REST API we use everywhere else: 10 req/min
  unauthenticated, 30 req/min authenticated, vs. 5000/hour for the contents/tree endpoints.
  A single scan calling `search_code` a few times could burn a meaningful fraction of that
  budget, and it's a shared, service-wide budget if this ever serves concurrent users.
- It only indexes the default branch, skips files over ~384KB, can lag behind a very recent
  push, and needs its own error-handling path for a different failure shape (422s for
  malformed queries, different rate-limit headers, etc.) -- more surface area for a tool that
  only needs to answer "does this term appear anywhere."

**Why local search instead:** reuses the exact same `github_client.read_file` path and rate
budget every other tool already shares; not subject to GitHub's search index at all, so it
works even on a repo pushed seconds ago; and it shares `RepoTools`' per-scan file-content
cache with `read_file`, so a file fetched during a search isn't re-fetched if the model later
reads it directly.

**Trade-off, stated plainly:** it is not exhaustive. Candidates are capped at
`MAX_SEARCH_FILES` (20) files or `MAX_SEARCH_BYTES` (200KB), whichever comes first, and
bounded by the per-tool timeout -- so a match outside that bounded, priority-ordered set
(source code first, then config, then docs/tests/examples) will be missed on a large repo.
The tool's own description says so ("not guaranteed to cover every file"), and a bounded run
always says explicitly when it stopped early and why, rather than silently under-searching.
This was confirmed for real, not just in tests: `search_code("def echo")` against
`pallets/click` (176 candidate files) hit its 15s time budget after 13 files and returned
"no matches... search stopped early: ran out of time", rather than a false "not found"
presented as exhaustive.

## Tool-set-size smoke test (M2a) — 2026-09-28

Ollama 0.10.1, temperature 0.2, 10 attempts per (scenario, tool-set) combination, one warm-up call excluded from timings. Tool schemas are the real ones from app.agent.tools, not hand-rolled. `short-path`/`nested-path` run under both tool-set sizes (the accuracy comparison); `dependencies`/`search-term` only make sense with all 4 tools available.

| Model | Tool set | Scenario | Tool called | Valid JSON | Args OK | Avg s | Median s |
|---|---|---|---|---|---|---|---|
| `qwen2.5:7b-instruct` | 2-tool | short-path | 10/10 | 10/10 | 10/10 | 6.8 | 6.8 |
| `qwen2.5:7b-instruct` | 4-tool | short-path | 10/10 | 10/10 | 10/10 | 6.9 | 6.8 |
| `qwen2.5:7b-instruct` | 2-tool | nested-path | 10/10 | 10/10 | 10/10 | 7.4 | 7.3 |
| `qwen2.5:7b-instruct` | 4-tool | nested-path | 10/10 | 10/10 | 10/10 | 8.0 | 7.9 |
| `qwen2.5:7b-instruct` | 4-tool | dependencies | 10/10 | 10/10 | 10/10 | 6.6 | 6.5 |
| `qwen2.5:7b-instruct` | 4-tool | search-term | 10/10 | 10/10 | 10/10 | 7.3 | 7.2 |

**Takeaway:** no tool-selection accuracy drop going from 2 to 4 tools -- 100% correct tool,
valid JSON, and correct arguments across all 60 real attempts, for both the carried-over
`short-path`/`nested-path` scenarios and the two new ones. Latency ticked up modestly with
more tools available (short-path 6.8s -> 6.9s, nested-path 7.4s -> 8.0s), consistent with a
longer prompt (more tool schemas to consider), not a reliability problem. `qwen2.5:7b-instruct`
handles 4 tools fine; this is worth re-checking once `report_finding` (M2 part B) makes it 5.

## M2a end-to-end scan comparison — 2026-09-29

Re-ran `python scan.py` on the same two repos as the M1 (prompt-tuned) runs above, now with
all 4 tools available:

| Repo | Steps | Tools called (in order) | Finished how |
|---|---|---|---|
| `octocat/Spoon-Knife` | 3 (was 4) | `get_dependencies`, `read_file(README.md)` *(same step)*, `search_code("Spoon-Knife")` | model's own final answer |
| `octocat/Hello-World` | 7 (was 5) | `get_dependencies`, `read_file(README.md)` *(404)*, `read_file(README)`, `get_dependencies` *(repeat)*, `list_directory("")` *(repeat)*, `read_file(main.py)` *(404)*, `search_code("password")` | model's own final answer |

Neither run hit the step limit. Mixed result, stated honestly:

- **Spoon-Knife got faster but shallower.** With only `list_directory`/`read_file` (M1), the
  model read all three files (`README.md`, `index.html`, `styles.css`) in full and produced a
  summary grounded in their actual content. With all 4 tools, it called `get_dependencies`,
  read only the README, then used `search_code("Spoon-Knife")` instead of reading
  `index.html`/`styles.css` directly -- fewer steps, but the resulting summary leans on search
  snippets for those two files rather than their full content (e.g. it says styles.css's
  content "was not found during initial exploration", which is misleading -- the file exists
  and is small, `search_code` just didn't happen to match the query against it). The extra
  tool gave the model a shortcut it took instead of the more thorough path.
- **Hello-World got slower but more thorough.** It tried a plausible-but-wrong README
  filename and a plausible-but-absent entry point (`main.py`) -- both cleanly reported as
  "not found", no crash -- then correctly used `search_code("password")` to check for secrets
  before concluding. It also repeated `get_dependencies` and `list_directory("")` once each;
  both dedup notes fired, and the *loop's* "you already called this" hint additionally fired
  on the repeated `get_dependencies` call, all in the same step -- a good real demonstration
  of the two independent guardrails compounding as designed. Final answer is well-reasoned
  and honestly states what's missing rather than guessing.

Net: more tools didn't reduce reliability (no crashes, no wasted step-limit runs), but they
did change *what* the model chose to do -- sometimes trading thoroughness for speed
(Spoon-Knife) and sometimes the reverse (Hello-World). Worth watching on non-trivial repos in
M5's evaluation suite, where shallow-but-fast answers would show up as missed findings.

## M2a follow-ups: search_code concurrency and a prompt nudge — 2026-09-29

Two small changes after the Spoon-Knife/Hello-World results above.

**1. Bounded concurrency in `search_code`.** Candidate files are now fetched in batches of
`SEARCH_CONCURRENCY` (5) via a small `ThreadPoolExecutor`, instead of one at a time, still
respecting the file-count cap, byte budget, and deadline (checked between batches; a batch
still in flight when the deadline hits is abandoned -- not waited on or counted -- same
"can't cancel a running thread" caveat the loop's own per-tool-call timeout already has).
Re-ran `search_code("def echo")` against `pallets/click` (176 candidate files) for real:

| | Sequential (before) | Concurrent (after) |
|---|---|---|
| Files searched | 13 | 20 |
| Why it stopped | ran out of time (15s) | hit the 20-file cap |

Concurrency measurably improved coverage within the same time budget -- enough that the
*file-count* cap is now the binding constraint on this repo instead of the time budget. Still
didn't find `def echo` (it's further down the alphabetically-sorted source-tier candidate
list than the first 20), which is expected and consistent with the documented trade-off
above: bounded means bounded, not exhaustive.

**2. System prompt nudge: "search_code only locates candidates -- confirm what a file
actually contains with read_file before drawing conclusions about it."** Re-ran
`python scan.py` on `octocat/Spoon-Knife` to check the effect. Result: honestly mixed. This
run took 3 steps -- `get_dependencies()` (none found), `read_file("README.md")` (full read),
then a repeated `get_dependencies()` call (both the RepoTools "already fetched" note and the
loop's "already called" hint fired correctly) -- and concluded without calling `search_code`
or `read_file` on `index.html`/`styles.css` at all. It described both files anyway, using the
filenames and byte sizes already visible in the auto-provided root listing (e.g. correctly
citing styles.css as "256 bytes"), not by reading their content.

So: **it did not reliably start reading files it search-matched or merely saw listed** --
the prompt nudge didn't force the behavior it was aimed at, at least not on this repo/run.
It didn't make anything worse (no crash, no wasted steps, guardrails still fired correctly),
but this is a case where soft prompt guidance alone wasn't enough to change a small model's
behavior. Documenting honestly rather than re-running for a nicer result. Worth revisiting
with a stronger nudge (or accepting as a known small-model limitation) if M5's evaluation
suite shows shallow-but-plausible summaries costing real findings on non-trivial repos.
