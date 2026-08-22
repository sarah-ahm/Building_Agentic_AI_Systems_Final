# RepoScribe — 4-minute demo script

Talking points + on-screen actions for the recorded demo. Timings are targets; the
whole thing is ~4:00. Everything shown runs **offline** (no key on camera — the live
path is mentioned, not run, so no secret ever appears).

> **Before recording:** `cd reposcribe`, have a terminal + editor ready, run
> `python -m pytest -q` once so pip/imports are warm. Never show `.env` or a real key.

---

## 0:00 – 0:30 — The problem & the one-line pitch

**Say:**
> "Documentation generators built on LLMs hallucinate — they describe functions that
> don't exist and cite the wrong lines. RepoScribe is an agentic doc generator with one
> defining property: **every symbol it documents and every citation it makes is verified
> against a ground-truth symbol table extracted from the source.** So instead of asking
> 'do the docs look right?', I get hard numbers: coverage, hallucination rate, citation
> validity. My target is a real package — Salesforce's `@lwc/module-resolver`."

**Show:** the README top, then `docs/architecture.md` diagram on screen briefly.

## 0:30 – 1:15 — Architecture & the agentic patterns

**Show:** the mermaid diagram in `docs/architecture.md`.

**Say — trace the flow, naming patterns as you point:**
> "It's a hand-orchestrated `asyncio` pipeline, following the course labs rather than a
> framework, so every stage is testable. **Tool use:** a repo walker and tree-sitter
> extract a `SymbolTable` — that's the oracle. **Planning** produces a doc plan;
> **routing** with a confidence gate labels each file; writers run in **parallel** and
> use **RAG** to pull the right code chunks and cite `file:line`. A **reflection** pass
> critiques and revises. Then the **output guardrail** checks every draft against the
> symbol table and prunes anything ungrounded — and a **human approval gate** runs before
> anything is written. That's planning, routing, parallelization, RAG, reflection, tools,
> guardrails, two kinds of memory, HITL, and evaluation — well past the four required."

## 1:15 – 2:15 — Run it (the money shot)

**Show + run:**
```bash
python demo/demo.py
```

**Say while it runs:**
> "This runs the full pipeline offline on the frozen package. It walked 9 files,
> extracted 41 symbols, identified 8 public API symbols from the barrel `index.ts`
> re-exports, and generated the docs."

**Point at the rendered API reference and the metrics line:**
> "Here's `resolveModule` documented with its real signature and a citation to
> `resolve-module.ts`. And the metrics: **coverage 1.0** — all 8 public symbols —
> **hallucinated 0**, **invalid citations 0**. Notice the internal helpers like
> `readJson` are *not* here — the router and public/internal split keep them out."

**Optional show:** `out/api_reference.md` and `out/workspace_state.json` (the memory trace).

## 2:15 – 3:10 — Why you can trust it: tests + eval + ablation

**Show + run:**
```bash
python eval/run_eval.py
```

**Say:**
> "This is the evaluation harness — 14 cases scored against the symbol table:
> 8 coverage, a citation check, 3 discrimination cases that internal helpers must
> *not* be documented, a groundedness check, and a **prompt-injection** case where a
> malicious instruction planted in a source comment must be neutralized, not obeyed.
> **14 out of 14 pass.**"

**Point at the ablation lines:**
> "The key experiment is the guardrail ablation. I fault-inject a draft with one made-up
> symbol and one bogus citation. **Guardrail off:** hallucination 0.33, citation validity
> 0.5 — the garbage flows straight through. **Guardrail on:** 0.0 and 1.0 — both are
> caught and pruned. That's the difference between docs that *look* right and docs that
> are *verified* right."

**Optional show:** `python -m pytest -q` → "22 tests, all offline, no key needed —
fully reproducible for a grader."

## 3:10 – 3:45 — Honesty & Colab

**Show:** `eval/eval_report.md` failure section, then the Colab notebook.

**Say:**
> "I'm explicit about limits in the eval report: the guardrail verifies that symbols and
> citations are *real*, not that the prose is semantically perfect — a fluent-but-wrong
> sentence would still pass, and catching that needs an entailment check, which is future
> work. Everything runs in this Colab notebook: it installs deps, runs the tests and eval,
> and renders the docs. The same code path talks to real Gemini with a key added in Colab
> Secrets — the key is never in the code."

## 3:45 – 4:00 — Close

**Say:**
> "So: a runnable agentic system, ten-plus patterns, a real open-source target, and an
> evaluation that proves the central claim — verified, non-hallucinated documentation.
> Thanks."

---

### Quick-reference facts (don't misspeak)

- Target: `@lwc/module-resolver` — 9 files walked, 41 symbols, **8 public** API symbols.
- Public symbols: `resolveModule`, `RegistryType`, `ModuleResolverConfig`, `RegistryEntry`,
  `ModuleRecord`, `AliasModuleRecord`, `DirModuleRecord`, `NpmModuleRecord`.
- Offline results: **22/22 tests**, **14/14 eval**, coverage 1.0, hallucination 0.0, citation validity 1.0.
- Ablation: guardrail OFF → (0.333, 0.5); ON → (0.0, 1.0).
- Patterns (≥4 required, ~11 shipped): planning, routing+gate, parallelization, RAG w/
  citations, reflection, tool use (×3), input+output guardrails, memory (×2), HITL, eval.
- Provider: Gemini (`gemini-2.5-flash`, `gemini-embedding-001`); offline mock for reproducibility.
