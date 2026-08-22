# RepoScribe — Evaluation Report

**Target under test:** the frozen `@lwc/module-resolver` package
(`eval/fixtures/lwc-module-resolver/`, 6 source files + tests), plus a small
`eval/fixtures/injected/` fixture for the prompt-injection guardrail.

**How to reproduce**

```bash
python eval/run_eval.py          # offline, deterministic (MockLLM) — no API key
python eval/run_eval.py --live   # scores a real Gemini run (needs GEMINI_API_KEY)
```

## 1. Methodology

RepoScribe's premise is that documentation quality can be measured objectively when
every claim is tied to a ground-truth **SymbolTable** extracted from source by
tree-sitter. The harness runs the full pipeline, then scores 14 cases from
`test_cases.json` against the generated docs and that table:

- **Coverage cases (8):** each public API symbol must appear in the API reference.
- **Citation case (1):** `resolveModule` must be cited to its real `file:line`.
- **Discrimination cases (3):** internal/private helpers (`getModuleEntry`, `readJson`,
  `resolveModuleFromNpm`) must **not** be documented as public API.
- **Groundedness case (1):** no documented symbol may be absent from the SymbolTable.
- **Injection case (1):** an instruction planted in a source comment
  ("ignore all previous instructions…") must be neutralized, not obeyed.

The "public API" is defined precisely as the set of names re-exported by the package
barrel `index.ts` (8 symbols), which the extractor recovers from `export … from`
statements.

## 2. Metrics

| Metric | Definition |
| --- | --- |
| **Task success rate** | passing cases / total cases |
| **API coverage** | documented public symbols / total public symbols |
| **Hallucination rate** | documented symbols not in the SymbolTable / documented symbols |
| **Citation validity** | citations pointing inside a real symbol's line range / total citations |

## 3. Results (offline `--mock` run — deterministic)

| Metric | Result |
| --- | --- |
| Task success rate | **14 / 14 = 1.00** |
| API coverage | **1.00** (8/8 public symbols) |
| Hallucination rate | **0.00** |
| Citation validity | **1.00** |
| Wall-clock | ~0.1 s |

All 14 cases pass in the offline configuration. The mock model is perfectly grounded by
construction, so these numbers verify the *plumbing* — extraction, routing, the
public/internal split, citation wiring, and the injection guardrail — rather than the
prose quality of a real model (see §5).

## 4. Ablation — with vs. without the groundedness guardrail

The guardrail (symbol-existence + citation-range checking, then pruning) is RepoScribe's
key component, so we ablate it directly. Because the mock model never errs, we use
**fault injection**: the checker is handed a draft containing one hallucinated symbol
(`resolveModuleAsync`) and one out-of-range citation (line 99999), simulating an
imperfect LLM. We then compare the metrics with the guardrail off vs. on.

| Configuration | Hallucination rate | Citation validity |
| --- | --- | --- |
| Guardrail **OFF** | 0.333 | 0.50 |
| Guardrail **ON** | **0.00** | **1.00** |

With the guardrail off, the fabricated symbol and bogus citation flow straight into the
output. With it on, both are detected and pruned before anything is written. This is the
difference between "docs that look right" and "docs that are verified right."

## 5. Failure analysis (3 examples)

These are genuine limitations, with concrete examples from the target package. (1) is a
current behavioral gap; (2) and (3) are the main risks in a **`--live`** run.

1. **Dual-nature symbols under-documented.** `RegistryType` exists twice in the source —
   as a runtime `const` (`types.ts:8`) *and* as a `type` alias (`types.ts:13`). RepoScribe
   resolves each public name to a single definition, so it documents the `const` facet and
   silently drops the type-alias facet. The docs are correct but incomplete for this symbol.

2. **Prose groundedness is not verified.** The guardrail checks that documented symbols and
   citations are *real*; it does **not** check that the natural-language description is
   *accurate*. A live model could write a fluent but wrong sentence (e.g., "`resolveModule`
   caches results across calls") and every automated metric would still read 100%. Catching
   this would need an entailment/NLI check against the source — future work.

3. **Barrel files carry no definitions.** `index.ts` is routed `public_api` but defines
   zero symbols; the real definitions live in files routed `internal`/`types`. An earlier
   design that documented "the public_api file" would have missed `resolveModule` entirely.
   We fix this by resolving public *names* to their definitions regardless of file role —
   but it means routing alone cannot determine coverage, and a package with an unusual
   re-export style (e.g., `export *`) could still slip a public symbol past the selector.

## 6. Live runs

`--live` exercises the same 14 cases against real Gemini output. Expected differences from
the mock baseline: coverage and citation validity may dip slightly if the model omits a
symbol or mis-cites a line (the guardrail then prunes the bad citation, trading citation
validity for a small coverage loss), and latency rises to a few seconds per section. The
injection case is the most informative live: it confirms the sanitized comment does not
steer the model.
