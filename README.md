# RepoScribe

**An agentic documentation generator that cannot hallucinate.**

Point RepoScribe at a code package and it generates an **API reference**, a **user
guide**, and a **changelog** — where *every documented symbol and every citation is
verified against a ground-truth symbol table* extracted from the source with
tree-sitter. If the model invents a function that doesn't exist or cites the wrong
line, the guardrail catches it and prunes it before anything is written. That turns
"do the docs look right?" into hard, reproducible numbers: coverage, hallucination
rate, and citation validity.

Built for the course capstone. It mirrors the class labs (Gemini via `google-genai`,
pydantic-typed structured output, a hand-orchestrated `asyncio` pipeline) and runs in
Google Colab. The default target is the real **`@lwc/module-resolver`** package from
the [Salesforce LWC](https://github.com/salesforce/lwc) monorepo.

---

## What it does (the short version)

```
walk repo ─▶ extract symbols (tree-sitter) ─▶ SymbolTable  ← the ground truth
                                                   │
   plan ─▶ route files ─▶ [parallel] write sections (RAG-grounded, cited)
                                                   │
                              critique + revise (reflection)
                                                   │
              check groundedness vs SymbolTable ─▶ prune anything unverified
                                                   │
                    human approval gate ─▶ write api_reference / user_guide / changelog
```

See [docs/architecture.md](docs/architecture.md) for the full diagram, the design
decisions, and the map of which agentic patterns live where.

## Agentic patterns implemented

Planning · Routing (with a confidence gate) · Parallelization (`asyncio.gather`) ·
RAG with `file:line` citations · Reflection (critique + revise) · Tool use
(tree-sitter symbols, repo walker, `git log`) · Guardrails (input sanitization +
output groundedness) · Memory (semantic vector index + episodic `WorkspaceState`) ·
Human-in-the-loop approval · an Evaluation harness. (Requirement was ≥4.)

---

## Quick start

**Everything runs offline with a deterministic mock LLM — no API key needed.** That is
the intended way to reproduce the tests and eval numbers. A real Gemini key only turns
on the `--live` path.

### Option A — Colab (primary)

Open [`RepoScribe_Colab.ipynb`](RepoScribe_Colab.ipynb) and run the cells top to bottom.
The notebook installs deps, runs the offline tests + eval, generates the docs, and
renders them inline. To run *live* against real Gemini, add your key in **Colab →
Secrets** as `GEMINI_API_KEY` and run the live cell.

### Option B — Local

```bash
cd reposcribe
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or:  pip install -e ".[dev]"

# 1. Tests (offline, deterministic)
pytest -q

# 2. Evaluation harness (offline) — prints metrics + guardrail ablation
python eval/run_eval.py

# 3. Scripted demo — generate docs for @lwc/module-resolver and print them
python demo/demo.py

# 4. The CLI directly
PYTHONPATH=src python -m reposcribe run \
    --repo eval/fixtures/lwc-module-resolver --out ./out --mock --yes
```

Generated docs land in `out/`: `api_reference.md`, `user_guide.md`, `changelog.md`,
plus a `workspace_state.json` trace (the episodic-memory record of the run).

### Going live (real Gemini)

```bash
cp .env.example .env          # then put a real key in .env (it is git-ignored)
python eval/run_eval.py --live
python demo/demo.py --live
PYTHONPATH=src python -m reposcribe run --repo eval/fixtures/lwc-module-resolver --out ./out --yes
```

## CLI reference

```
python -m reposcribe run --repo PATH [--out DIR] [--mock] [--yes]
                         [--no-rag] [--no-reflection] [--model NAME]
```

| Flag | Meaning |
| --- | --- |
| `--repo` | package/directory to document (required) |
| `--out` | output directory (default `./out`) |
| `--mock` | use the offline deterministic LLM — no API key |
| `--yes` | skip the human approval gate (for scripts/CI) |
| `--no-rag` | ablation: turn off RAG grounding |
| `--no-reflection` | ablation: skip the critique/revise pass |
| `--model` | override the Gemini model (default `gemini-2.5-flash`) |

## Configuration & secrets

Copy `.env.example` → `.env` and set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`). Loading
order follows the course-lab pattern exactly: `.env` / real environment first, then
Colab Secrets (`google.colab.userdata.get("GEMINI_API_KEY")`). **No key is ever
hard-coded**; `.env` is git-ignored and `.env.example` ships placeholders only. The
`--mock` path needs no key at all. Optional overrides: `GEMINI_MODEL`,
`GEMINI_EMBED_MODEL`, `LLM_PROVIDER`.

## Project layout

```
reposcribe/
├── src/reposcribe/       config, models, llm (+ MockLLM), repo/symbols/rag tools,
│                         agents, guardrails, pipeline (orchestration + CLI)
├── tests/                pytest suite (offline, LLM mocked)
├── eval/                 test_cases.json, run_eval.py, eval_report.md,
│                         fixtures/ (frozen @lwc/module-resolver source)
├── docs/architecture.md  diagram, decisions, patterns map, limitations
├── demo/demo.py          scripted local demo
└── RepoScribe_Colab.ipynb  primary Colab driver
```

## Reproducibility

The target package's `src/` is vendored frozen under
`eval/fixtures/lwc-module-resolver/`, and the whole pipeline runs offline via the
deterministic `MockLLM` + hashed bag-of-words embeddings. So `pytest`,
`python eval/run_eval.py`, and `python demo/demo.py` all produce the **same numbers on
any machine with no network and no API key** — currently 22/22 tests passing and
14/14 eval cases passing (coverage 1.0, hallucination 0.0, citation validity 1.0). See
[eval/eval_report.md](eval/eval_report.md) for methodology, the guardrail ablation, and
an honest failure analysis.

## Limitations (model choice, cost, rate limits, hardware)

- **Model choice.** `gemini-2.5-flash` is fast and cheap but a smaller model than Pro;
  on a larger or messier codebase it may omit a symbol or write a fluent-but-imprecise
  description. The groundedness guardrail catches invented symbols and bad citations, but
  it does **not** verify that prose is semantically correct — see the failure analysis in
  [eval/eval_report.md](eval/eval_report.md).
- **Cost & rate limits.** A run is low-volume (a handful of calls for this package), so
  cost is negligible. Live runs can still hit Gemini's free-tier **429 (rate limit)** or
  **503 (overloaded)**; the client retries with backoff (`llm.py`), which adds latency but
  not failures. Very large targets would multiply calls and could approach quota.
- **Latency.** Offline (`--mock`) is instant; a live run is a few seconds per section.
- **Hardware.** No GPU needed — all compute is remote (Gemini) and the local vector index
  is tiny numpy. Runs fine in Colab's free CPU tier and on any laptop.
- **Scope.** Symbol extraction is implemented for TypeScript/TSX/JavaScript/Python; other
  languages return no symbols. Designed for one bounded package, not a whole monorepo.

For the full architectural limitations, see [docs/architecture.md](docs/architecture.md).

## Use of AI (course disclosure)

This project was built with AI assistance (Claude Code) for scaffolding, drafting, and
review. All design decisions, the choice of target and patterns, the evaluation
methodology, and every artifact were directed and verified by me. The LLM idioms
(client/retry wrapper, structured output, embeddings, secret loading) deliberately
mirror the course labs. RepoScribe *itself* is an LLM application, and its guardrails
exist precisely to keep an LLM's output honest — the eval report documents where that
does and does not hold.
