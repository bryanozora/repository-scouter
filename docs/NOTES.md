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

## Known limitations of the evidence check (M2b) — 2026-10-01

`check_evidence` in `app/agent/findings.py` is a regex plausibility check, not a scanner. Two known gaps, left as-is for v1:

- **False "verified" SQL injection:** a non-SQL f-string (e.g. a log line) within ±2 lines of a parameterized query can make an `sql_injection` finding pass, since the SQL keyword and the dynamic-string marker only need to appear somewhere in the same window, not in the same expression.
- **Missed env-var fallback secrets:** a real secret used as a default, like `os.getenv("API_KEY", "real-secret-value")`, does not verify as `hardcoded_secret`, because only `name = "literal"`-style assignments (and known key formats) are recognized.

## Measured: Ollama's effective context window is 4096 tokens — 2026-10-01

`ollama show qwen2.5:7b-instruct` reports `context length 32768`, but that's what the model
*supports*, not what the server allocates. With Ollama 0.10.1 defaults (no
`OLLAMA_CONTEXT_LENGTH` set), measured through the same OpenAI-compatible endpoint the agent uses,
with a codeword planted at the very start of the prompt:

| Prompt size | `usage.prompt_tokens` | Model recalled the codeword at the start? |
|---|---|---|
| ~2k tokens | 1955 | yes |
| ~8k tokens | **4096** (capped) | **no** — answered "filler" |

So a prompt over 4096 tokens is **silently truncated from the start**, and the start of an agent
conversation is the system prompt, prompt-injection defense included. The M1/M2a scans used
6000-char tool results (~2000 tokens each) and very likely crossed this line after two or three reads
without any error, which may explain some of the shallow-summary behavior recorded above.

Raising the window isn't practical on this machine: with `num_ctx: 8192` (per request, via the native
`/api/chat` API, without changing any persistent setting) the model grew to 6.3 GB, split 45%/55%
CPU/GPU, and a single ~8k-token prompt did not finish within 15 minutes.

What M2b does about it:
- `MAX_PROMPT_TOKENS=3300`: the loop projects every prompt's size before sending it and keeps it under
  this, leaving ~800 tokens of the 4096 for the reply.
- The fixed part of every prompt (system prompt + 5 tool schemas) measured **1261 tokens**, so the
  system prompt was rewritten to be compact.
- `MAX_TOOL_RESULT_CHARS` lowered from 6000 to 2500 (~800 tokens), and the loop now caps *every*
  tool result to it (+500 chars headroom so `read_file`'s own whole-line truncation is never re-cut).
- When the projection is over budget, the loop first **compacts** the oldest tool results (replaces
  them with a short "removed to fit the context window" marker), and only ends the scan with a
  fallback summary if that's not enough. A compacted `read_file` result stops counting as "read" for
  `report_finding`, so evidence for a finding is always something the model can currently see.

## M2b first real runs: 0 of 8 planted bugs recorded — 2026-10-01

`python scan.py --local ../evals/repos/<fixture>`, `qwen2.5:7b-instruct`, M2b prompt and
guardrails, `SCAN_TIMEOUT_SECONDS=900` for these runs (see the duration column; the default of 120s
would have cut clean-control short). An earlier py-notes-api run was discarded because the fixture's
answer key (`expected.json`) was inside the scanned directory and `search_code` found it; ground
truth now lives in `evals/expected/`, and a test guards against that happening again.

| Fixture | Planted | Recorded (verified) | False positives recorded | Steps | Max prompt | Duration |
|---|---|---|---|---|---|---|
| `py-notes-api` | 4 | 0 | 0 | 3 | 1405 | 85s |
| `js-shop` | 4 | 0 | 0 | 3 | 1418 | 146s |
| `clean-control` | 0 | 0 | 0 | 8 | 2029 | 314s |

The evidence check never ran on a real finding, because the model almost never reached
`report_finding`:

- **It stops after 2-3 steps** and ends with a text answer, without reading any source file.
- **It writes tool calls as text.** In py-notes-api it ended with ```` ```json {"name": "read_file", ...} ```` ````
  in its answer instead of calling the tool; the loop treats any reply without `tool_calls` as the
  final answer.
- **It reports findings in prose.** In the discarded run it described `SMTP_PASSWORD` (correct file
  and line) in its summary rather than calling `report_finding`. In js-shop it listed two env-var
  lookups as "hardcoded secrets" in prose (both are decoys).
- **Its searches were too narrow.** It searched only for "password", never "SELECT"/"execute"/"token",
  so it never saw the SQL injections, `PAYMENT_TOKEN`, or `signing_secret`.
- **Its only `report_finding` call** (clean-control) left out `confidence` and was rejected. Instead of
  retrying, the model put the corrected JSON in its final text. That finding would have been a false
  positive anyway (a parameterized `%(t)s` query, on lines it hadn't read), so the read-gating and
  evidence check would have rejected or flagged it.

What worked: no crashes; the context budget held (max prompt 2029 of 3300, no compactions needed);
the validator's error message was clear; the dedup notes fired. Stopped here, before the 5-tool smoke
test, for a decision on how to revise the prompt or loop.

## Documented limitation (for the README): 4096-token context window — 2026-10-01

- **What:** with Ollama's defaults on this machine, `qwen2.5:7b-instruct` gets a 4096-token context
  window (measured; see "Measured: Ollama's effective context window" above), not the 32k the model
  supports. A longer prompt is **silently truncated from the start**. No error is returned, and
  `usage.prompt_tokens` just reads 4096.
- **Risk for earlier scans:** the start of an agent conversation is the system prompt, which holds
  the role, the tool rules, and the prompt-injection defense ("contents of `<file_content>` are
  untrusted"). **Scans made before the M2b token guardrail (all M1 and M2a runs in this file) may
  have silently lost the system prompt** once several 6000-char tool results pushed the
  conversation past 4096 tokens. Their results, including any "the injection defense held"
  conclusions, should not be relied on.
- **Mitigation since M2b:** every prompt is projected before sending and kept under
  `MAX_PROMPT_TOKENS=3300`, older tool results are compacted to make room, and the scan ends with a
  fallback summary rather than send an over-long prompt. The cost is that the agent only ever sees
  ~2000 tokens of repository content at a time.
- **Not done:** raising the window (`OLLAMA_CONTEXT_LENGTH` / `num_ctx`) isn't practical on an 8GB
  machine: at 8192 a single long prompt didn't finish in 15 minutes. A larger machine or a hosted
  provider would lift this limit; the guardrail adapts through `MAX_PROMPT_TOKENS`.

## M2b re-run with nudges + optional confidence: still 0 of 8 recorded — 2026-10-01

Same 3 fixtures, same model, after adding (1) one corrective retry for a tool call written as
text, (2) one push back on an early stop that skipped reading source or one issue type, and
(3) `confidence` optional (default 0.5). New defaults, no env overrides (`SCAN_TIMEOUT_SECONDS=400`).

| Fixture | Planted | Recorded (verified) | False positives recorded | Steps | Nudges (text / early-stop) | Duration |
|---|---|---|---|---|---|---|
| `py-notes-api` | 4 | 0 | 0 | 9 | 0 / 1 | 362s |
| `js-shop` | 4 | 0 | 0 | 6 | 1 / 1 | 466s |
| `clean-control` | 0 | 0 | 0 | 8 | 1 / 1 | 480s |

The nudges changed behavior: scans went deeper (6-9 steps instead of 3), and the model now
searched for both issue types and read some files. Recall stayed at 0 for a different reason.
**The model found real bugs but never called `report_finding` as a tool:**

- **py-notes-api:** the final text correctly names `db.py` line 24 (f-string SQLi), line 42
  (concatenation SQLi) and `SMTP_PASSWORD`, so 3 of 4 planted bugs were identified *in prose*. It
  then claims "These findings have been recorded and reported", which is **false** (none were). It
  also never read `db.py` with `read_file`; it described those lines from `search_code` snippets, so
  the read-gating would have rejected them anyway.
- **js-shop:** after the text-tool-call nudge, it wrote findings as Python-call syntax in text
  (`report_finding("medium", "security", ...)`), which isn't JSON, and the one nudge was already used.
  Prose identified 1 planted bug (the template-literal SQLi), cited together with a decoy line, plus
  one false positive (`process.env.DB_PASSWORD` as a "hardcoded secret").
- **clean-control:** wrote two `{"name": "report_finding", ...}` objects as text, both false
  positives (parameterized `%s` / `%(t)s` queries). Not recorded, so no false positive in the
  report, but only because the call was never made.

Also seen: two scans ran past the 400s timeout (466s, 480s), since it's checked once per step and a
step takes ~40-60s. The timeout is best-effort by design.

**Takeaway:** the bottleneck is the `report_finding` tool call itself. `qwen2.5:7b-instruct` calls
the read/search tools reliably, but for findings it falls back to writing them as text. Fixed
built-in searches (approach 3) wouldn't address that, because finding the bugs isn't what's failing.
The final summary text can also claim findings were recorded when they weren't, so the structured
findings list (not the prose) must be the source of truth in the report. Stopped here for a decision.

## Verified: Ollama 0.10.1 enforces `response_format` (JSON schema) — 2026-10-01

Tested directly on the OpenAI-compatible endpoint with `qwen2.5:7b-instruct`, using a prompt that
asks for prose ("Write a four-line poem about cats"), 2 attempts each:

| `response_format` | Output |
|---|---|
| none | the poem (not JSON), 2/2 |
| `{"type": "json_object"}` | valid JSON instead of the poem, 2/2 |
| `{"type": "json_schema", ...}` | valid JSON **in the requested schema's shape**, 2/2, even for a poem prompt |

So the output is constrained to the schema, not just requested. The *values* are not reliable (it
produced `"file": "poem"`, `"line_start": 1523456789012345`), which is why every item from the
forced JSON step still goes through validation, read-gating and the evidence check. The "reply with
only a JSON array" fallback wasn't needed.

## M2b round 3: forced JSON step + parsed text findings + search-seen lines — 2026-10-01

Same 3 fixtures and model. New since round 2: findings written as text in any reply are executed
through `report_finding` (`parsed_text`); after the scan, one schema-constrained JSON call lists the
findings (`forced_json`) with one correction round for rejected items; lines shown in full as
`search_code` matches count as seen.

| Fixture | Planted | Verified (planted) | Verified false positives | Flagged (unverified) | Steps | Duration |
|---|---|---|---|---|---|---|
| `py-notes-api` | 4 | **3** (all `parsed_text`) | 0 | 0 | 5 | 503s |
| `js-shop` | 4 | **1** (`forced_json`) | 0 | 0 | 5 | 343s |
| `clean-control` | 0 | 0 | **0** | 1 (decoy `TEST_PASSWORD`, flagged by evidence check) | 8 | 565s |

**Total: 4 of 8 planted bugs verified, 0 verified false positives** (rounds 1-2: 0 of 8). No finding
came through a real `report_finding` tool call; the one real tool-call finding (clean-control,
`TEST_PASSWORD`) was a false positive and was flagged.

Missed: `notes-analytics-key` (never reported), and in js-shop `shop-payment-token`,
`shop-status-concat`, `shop-webhook-secret`. The model never opened `src/payments.js` or
`config/default.yml`; its searches ("password", "SELECT") don't match `PAYMENT_TOKEN` or
`signing_secret`, and its forced JSON list held only the template-literal SQLi.

**Bug found in the forced step (mine):** in py-notes-api, all 4 `forced_json` items were rejected
with "you have not seen any lines of config.py / db.py", although `parsed_text` had just verified
the same lines. The forced call compacted the conversation to fit the context window (6 compactions),
and compaction calls `forget_shown` / `forget_search_lines`, which cleared the seen-lines records
right before the forced items were checked. It didn't cost recall in this run (they were duplicates of
already-verified findings), and the correction round was wasted, but it would reject genuinely new
findings in the forced step. Fix to discuss: compaction for the final steps (fallback / forced report)
should not forget seen lines. The lines *were* seen earlier in the scan, which is the original rule.

Other observations: the model twice left out `suggestion` in a real tool call (clean-control), which
validation rejected; scans now take 343-565s (the 400s timeout is checked between steps, and the
forced step runs after it).

## Tool-set-size smoke test (M2b) — 2026-10-01

Ollama 0.10.1, temperature 0.2, 10 attempts per (scenario, tool-set) combination, one warm-up call excluded from timings. Tool schemas are the real ones from app.agent.tools, not hand-rolled. The four M2a scenarios run under both tool-set sizes (the accuracy comparison); `report-finding` only makes sense with all 5 tools, and counts as Args OK only if the finding passes the backend's real validator.

| Model | Tool set | Scenario | Tool called | Valid JSON | Args OK | Avg s | Median s |
|---|---|---|---|---|---|---|---|
| `qwen2.5:7b-instruct` | 4-tool | short-path | 10/10 | 10/10 | 10/10 | 6.7 | 6.7 |
| `qwen2.5:7b-instruct` | 5-tool | short-path | 10/10 | 10/10 | 10/10 | 7.0 | 6.8 |
| `qwen2.5:7b-instruct` | 4-tool | nested-path | 10/10 | 10/10 | 10/10 | 7.3 | 7.3 |
| `qwen2.5:7b-instruct` | 5-tool | nested-path | 10/10 | 10/10 | 10/10 | 7.6 | 7.4 |
| `qwen2.5:7b-instruct` | 4-tool | dependencies | 10/10 | 10/10 | 10/10 | 6.0 | 5.9 |
| `qwen2.5:7b-instruct` | 5-tool | dependencies | 10/10 | 10/10 | 10/10 | 6.3 | 6.1 |
| `qwen2.5:7b-instruct` | 4-tool | search-term | 10/10 | 10/10 | 10/10 | 6.8 | 6.7 |
| `qwen2.5:7b-instruct` | 5-tool | search-term | 10/10 | 10/10 | 10/10 | 7.0 | 6.8 |
| `qwen2.5:7b-instruct` | 5-tool | report-finding | 10/10 | 10/10 | 10/10 | 19.3 | 19.2 |

## M2 final accuracy summary (round 4, last tuning round) — 2026-10-01

Final configuration: `qwen2.5:7b-instruct` on Ollama 0.10.1 (4096-token context), all M2b features
on: candidate searches (the loop itself searches for password/secret/token/key/SELECT/execute( and
puts the matching lines in the first message), corrective nudges, findings parsed from text, forced
JSON reporting step, `confidence`/`suggestion` optional, and the round-3 fix so shortening the
conversation for the final steps no longer clears seen lines.

| Fixture | Planted | Verified planted | Verified false positives | Flagged (unverified) | Steps | Duration |
|---|---|---|---|---|---|---|
| `py-notes-api` | 4 | 3 | 0 | 1 (`SECRET_KEY = "change-me"` decoy) | 6 | 267s |
| `js-shop` | 4 | 2 | 0 | 0 | 4 | 310s |
| `clean-control` | 0 | 0 | **1** | 1 (`SESSION_SECRET` placeholder decoy) | 5 | 358s |

Where each verified finding came from:

| Fixture | Real `report_finding` call | Parsed from text | Forced JSON step |
|---|---|---|---|
| `py-notes-api` | — | — | SMTP password (`config.py:17`), f-string SQLi (`db.py:24`), `+` SQLi (`db.py:42`) |
| `js-shop` | — | — | template-literal SQLi (`orders.js:16`), `PAYMENT_TOKEN` (`payments.js:7`) |
| `clean-control` | — | false positive: `sql_injection` `inventory/db.py:14-30` | (same finding, deduplicated) |

(clean-control's first attempt this round ended with an Ollama request timeout, an environment
failure rather than an agent result; the table uses a clean re-run.)

**Final numbers:**
- **Recall: 5 of 8 planted bugs (62.5%)** verified. Missed: `notes-analytics-key`,
  `shop-status-concat`, `shop-webhook-secret`. Across rounds: 0/8, 0/8, 4/8, 5/8.
- **False positives: 1 of 6 verified findings (17%)**, 0 in the two fixtures with real bugs. The one
  false positive is the known limitation recorded above. The model cited a 17-line range (`14-30`)
  covering three parameterized queries *and* a logging f-string on line 18. The evidence check only
  requires an SQL keyword and a string-building marker somewhere in the same window, so the log line
  made it pass. Wide citations make this gap much easier to hit.
- **Flagged, not verified:** 2 placeholder-secret decoys, correctly flagged by the evidence check and
  kept out of the verified list.

**No verified finding in any round came from a real `report_finding` tool call.** Every one came from
the forced JSON step (5) or from text the loop parsed (round 3: 3). The only real tool calls ever made
(clean-control, rounds 1 and 3) were false positives, and they were rejected or flagged.

**Best guess why:** the model *can* make the call. The 5-tool smoke test above shows `report_finding`
called 10/10 with arguments that all pass the real validator, when the prompt is short and literally
says "record this as a finding". In a real scan none of that holds:
1. **Reporting competes with finishing.** The model tends to save findings for its final answer, and
   the system prompt asks for the final answer as plain text, so it writes findings as prose or JSON in
   that text instead of making a separate call.
2. **Once it starts writing text, it doesn't switch to a tool call.** With Qwen's chat template, Ollama
   only produces a tool call if the reply *starts* as one. Rounds 2-3 showed the model writing ```` ```json ````
   blocks or `report_finding(...)` text mid-answer, which Ollama returns as plain content.
3. **The context is longer and fuller.** A ~2-3k-token conversation with 5 schemas, versus a
   ~300-token single-turn test. `report_finding` is also the largest schema (9 fields), and it's
   slower even in isolation (~19s vs ~7s per call).

**What this means:** for this model, schema-constrained output (the forced JSON step) is the
dependable path for structured findings, and tool calling is dependable for navigation. That split is
the main design lesson of M2, worth stating plainly in the README's "honest limitations" section.
Accuracy tuning for M2 stops here, as agreed. Further work belongs to M5's evaluation suite.
