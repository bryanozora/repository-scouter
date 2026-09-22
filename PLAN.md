# Repository Scouter — Project Plan

An agentic AI tool that explores a public GitHub repository on its own and produces a structured report covering architecture, bugs, security issues, and suggested improvements.

---

## 1. Goals and Scope

**Goal:** Given a public GitHub repo URL, an LLM agent autonomously navigates the codebase using tools, then generates a structured report with evidence-backed findings.

**In scope (v1)**
- Public repositories only
- Read-only analysis (never executes repo code)
- Web UI with live streaming of the agent's steps
- Shareable report pages
- A small evaluation suite that measures agent quality

**Out of scope (v1)**
- Private repositories
- Running or building the scanned code
- Auto-opening issues or pull requests (candidate for v2)

---

## 2. Architecture

```
[Browser / Next.js]
      │  POST /scan {repo_url}      → returns scan_id
      │  GET  /scan/{id}/stream     (SSE: agent step events)
      │  GET  /scan/{id}            (final report)
      ▼
[FastAPI backend]
      │
      ├── Agent loop (LLM provider abstraction — default: Ollama + tool use)
      │      ├─ tool: list_directory
      │      ├─ tool: read_file
      │      ├─ tool: search_code
      │      ├─ tool: get_dependencies
      │      └─ tool: report_finding
      │
      ├── Repo access layer → GitHub REST API (tree + contents)
      └── Storage → SQLite
```

**Key design decision:** Access repositories through the GitHub API instead of `git clone`. This keeps untrusted code off the server, is lighter, and makes limits easy to enforce.

**LLM provider:** The agent talks to the LLM through a small provider abstraction (provider + model selected via `.env`), not a hardcoded SDK. The default provider is Ollama, called through its OpenAI-compatible endpoint (`http://localhost:11434/v1`), so other OpenAI-compatible providers can be added later without changing the agent loop.

---

## 3. Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Backend | Python, FastAPI | Async, easy SSE support |
| LLM | Provider abstraction over an OpenAI-compatible API; default provider: Ollama (`http://localhost:11434/v1`), model `qwen2.5:7b-instruct` (optional comparison candidate `qwen3:8b`) | Provider and model selectable via `.env`; sized for 8GB RAM; other providers can be added later |
| Frontend | Next.js, Tailwind | Live log + report UI |
| Streaming | Server-Sent Events | Simpler than WebSockets for one-way updates |
| Storage | SQLite | Move to Postgres if needed |
| Deploy | Vercel (frontend); backend run locally alongside Ollama | A live public demo is limited since Ollama runs locally — ship a demo GIF/video plus clear local-run instructions instead; a hosted OpenAI-compatible provider for a public demo is an optional stretch goal |

---

## 4. Agent Design

### Tools

| Tool | Purpose | Guardrails |
|---|---|---|
| `list_directory(path)` | List folder contents | Skip `node_modules`, `.git`, `dist`, `vendor`, etc. |
| `read_file(path, start, end)` | Read a file, optionally a line range | Size cap, reject binary files |
| `search_code(query)` | Search for a pattern across the repo | Cap the number of results |
| `get_dependencies()` | Read dependency manifests | Only known files (package.json, requirements.txt, etc.) |
| `report_finding(...)` | Record a structured finding | Schema validation |

### Loop

1. Provide the agent with initial context: repo name, primary language, top-level structure.
2. The agent decides which tool to call next.
3. Each tool result is returned to the agent and streamed to the UI.
4. The loop ends when the agent declares it is done **or** a limit is hit.
5. Final step: the agent writes a summary based on the recorded findings.

### Mandatory guardrails

- Maximum steps per scan (10–15, lower than a frontier-model default to suit small local models) and a maximum total token budget
- Stricter file-size caps and a shorter context-window budget so prompts fit small local models
- Tool-call and finding JSON validated against a schema, with one retry on malformed output before failing the step
- Track files already read to prevent repeated reads
- Timeouts per tool call and per scan
- Per-IP rate limiting to protect API costs
- Prompt injection defense: file contents are **data, not instructions**; state this explicitly in the system prompt
- All tools are strictly read-only

### Prompt strategy

- Role: a careful senior reviewer who does not make baseless accusations
- Every finding must include evidence (file + line range)
- The agent may express uncertainty via a confidence score
- v1 scope is intentionally narrow to suit a small local model: prioritize hardcoded secrets and SQL injection only; broader categories (auth, input handling, outdated dependencies, etc.) are a v2 stretch goal

---

## 5. Output Schema

**Finding (`report_finding`)**

```json
{
  "severity": "high | medium | low | info",
  "category": "security | bug | performance | maintainability",
  "file": "src/auth.py",
  "line_start": 42,
  "line_end": 55,
  "description": "...",
  "suggestion": "...",
  "confidence": 0.8
}
```

**Final report:** repo summary, short architecture overview, findings sorted by severity, and run statistics (steps, tokens, cost, duration).

**Verification:** after the agent reports a finding, the backend checks that the referenced file and line range actually exist. Findings that fail this check are flagged or dropped.

---

## 6. API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/scan` | Create a scan, return `scan_id` |
| GET | `/scan/{id}/stream` | SSE events: `step`, `finding`, `done`, `error` |
| GET | `/scan/{id}` | Final result (shareable report page) |

---

## 7. Repository Structure

```
repository-scouter/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI routes
│   │   ├── agent/
│   │   │   ├── loop.py        # agent loop
│   │   │   ├── tools.py       # tool definitions + implementations
│   │   │   └── prompts.py
│   │   ├── github_client.py
│   │   └── models.py
│   └── tests/
├── frontend/
├── evals/
│   ├── repos/                 # test repos with known, planted bugs
│   └── run_eval.py
├── docs/                      # diagrams, design notes
├── .env.example
├── PLAN.md
└── README.md
```

---

## 8. Milestones

### M0 — Setup (~0.5 day)
- [ ] Create the GitHub repo and folder structure
- [ ] Add `.env.example`
- [ ] Install Ollama, pull `qwen2.5:7b-instruct` and `qwen3:8b` (candidates for the M1 model comparison — skip `qwen3:8b` if it needs a newer Ollama version than you have), and get a GitHub token
- [ ] Manually call the GitHub API to list a repo's contents

### M1 — Agent MVP in the terminal (1–2 days)
- [ ] Tool-calling smoke test: compare `qwen2.5:7b-instruct` and `qwen3:8b` (if it runs on your Ollama version) on a single-tool-call task through the Ollama OpenAI-compatible endpoint (10 attempts each); record correct tool-call rate, JSON validity, and speed in `docs/NOTES.md`, and pick the default model based on those results
- [ ] Implement `list_directory` and `read_file` (only these two tools for v1)
- [ ] Basic agent loop with a step limit (10–15)
- [ ] **Done when:** `python scan.py <url>` produces a sensible architecture explanation

### M2 — Structured findings (2–3 days)
- [ ] Add `search_code`, `get_dependencies`, `report_finding`
- [ ] Output validation and tool error handling
- [ ] Guardrails: read-file tracking, token limit, timeouts
- [ ] **Done when:** scans produce consistent JSON findings on 3 different repos

### M3 — API and streaming (1–2 days)
- [ ] FastAPI endpoints with SSE
- [ ] Persist results to SQLite
- [ ] Rate limiting

### M4 — Frontend (2–3 days)
- [ ] URL input form
- [ ] Live agent step log
- [ ] Report page with severity filters
- [ ] Shareable result links

### M5 — Evaluation (~1 day)
- [ ] Prepare 3–5 test repos with planted bugs (SQL injection, hardcoded secrets, etc.)
- [ ] Script that computes recall and false positives
- [ ] Compare results across models (e.g. `qwen2.5:7b-instruct` vs `qwen3:8b`)
- [ ] Record results, and honest limitations of small local models, in the README

### M6 — Polish and deploy (1–2 days)
- [ ] Complete README, flow diagram, demo GIF/video
- [ ] Deploy the frontend (optional) and write clear local-run instructions — a full public live demo isn't feasible with a local-only LLM
- [ ] Add sample scan results from popular repos

**Estimated total:** 10–14 days at a relaxed pace.

---

## 9. Evaluation Plan

This is the main differentiator from typical vibe-coded projects. Document results in a table, including which model produced each row:

| Test repo | Model | Bugs planted | Caught | False positives | Latency |
|---|---|---|---|---|---|
| test-repo-1 | `qwen2.5:7b-instruct` | 5 | 4 | 1 | 12s |

Compare results across models (e.g. `qwen2.5:7b-instruct` vs `qwen3:8b`, and optionally a hosted model) and report the trade-offs in latency, tool-calling reliability, and accuracy. Document honest limitations of running small local models — missed findings, weaker reasoning on ambiguous cases, and less reliable structured output compared to larger hosted models.

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Local model is slow / limited context window | Step limits, keep prompts and tool results compact, cap file read size |
| Small local model unreliable at tool calling | Validate tool-call JSON, retry on malformed calls, keep tool schemas simple |
| Agent loops or rereads files | Track opened files, enforce step limit |
| Hallucinated findings | Require file + line evidence, verify lines exist |
| Very large repos | Prioritize key files, skip generated/vendor directories |
| Prompt injection via repo contents | Treat file contents as data, keep all tools read-only |
| GitHub API rate limits | Authenticated requests, caching, backoff |

---

## 11. README Checklist

- One-sentence description and demo GIF
- Agent flow diagram
- Features and usage
- Tech stack with reasons for each choice
- "Challenges and What I Learned" section
- Evaluation results
- Honest limitations
- A note that the project was vibe coded, plus which parts you understand most deeply

---

## 12. Stretch Goals (v2)

- Human-in-the-loop "open GitHub issue" action with explicit confirmation
- Result caching keyed by repo + commit SHA
- Cost-per-scan comparison across models
- Optional hosted, OpenAI-compatible LLM provider to enable a live public demo
- Support for private repos via OAuth
- Diff-aware scanning of pull requests

---

## 13. First Steps

1. Create an empty GitHub repo named `repository-scouter`.
2. Add this file as `PLAN.md` in the root (or `docs/`).
3. Install Ollama, pull `qwen2.5:7b-instruct` (and `qwen3:8b` as an optional comparison candidate, if your Ollama version supports it), and get your GitHub token.
4. Start M1: write an agent that can read one small repo.
