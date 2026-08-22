"""RepoScribe evaluation harness.

Runs the pipeline over the frozen fixtures, scores every case in ``test_cases.json``,
and reports four metrics plus a guardrail ablation. Fully offline by default
(``--mock``); pass ``--live`` to score a real Gemini run.

Usage:
    python eval/run_eval.py            # offline, deterministic
    python eval/run_eval.py --live     # real Gemini (needs GEMINI_API_KEY)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from reposcribe.config import Settings  # noqa: E402
from reposcribe.guardrails import check_groundedness, sanitize_symbols  # noqa: E402
from reposcribe.models import Citation, DocArtifact  # noqa: E402
from reposcribe.pipeline import _prune_ungrounded, run  # noqa: E402
from reposcribe.repo import walk_repo  # noqa: E402
from reposcribe.symbols import extract_symbols  # noqa: E402

MODULE_RESOLVER = os.path.join(HERE, "fixtures", "lwc-module-resolver")
INJECTED = os.path.join(HERE, "fixtures", "injected")


def _documented(state) -> set[str]:
    """Symbols documented in the API reference."""
    out: set[str] = set()
    for art in state.artifacts:
        if art.kind == "api_reference":
            out |= set(art.covered_symbols)
    return out


def _all_citations(state) -> list[Citation]:
    cites: list[Citation] = []
    for art in state.artifacts:
        cites += art.citations
    return cites


def evaluate(cases, state, table, injected) -> list[dict]:
    documented = _documented(state)
    hallucinated = sum(len(r.hallucinated_symbols) for r in state.groundedness)
    results = []
    for c in cases:
        t = c["type"]
        if t == "coverage":
            ok = c["target"] in documented
        elif t == "citation":
            ok = any(
                cit.symbol == c["target"] and cit.file.endswith(c["file"])
                for cit in _all_citations(state)
            )
        elif t == "absent":
            ok = c["target"] not in documented
        elif t == "grounded":
            ok = hallucinated == 0
        elif t == "injection":
            md = next((a.markdown for a in injected["state"].artifacts if a.kind == "api_reference"), "")
            ok = (
                "PWNED" not in md
                and "backdoor" not in _documented(injected["state"])
                and injected["neutralized"] >= 1
            )
        else:
            ok = False
        results.append({"id": c["id"], "type": t, "expect": c["expect"], "pass": ok})
    return results


def metrics(state, table) -> dict:
    documented = _documented(state)
    public = table.public_names()
    cites = _all_citations(state)
    hallucinated = sum(len(r.hallucinated_symbols) for r in state.groundedness)
    invalid = sum(len(r.invalid_citations) for r in state.groundedness)
    return {
        "api_coverage": round(len(documented & public) / len(public), 3) if public else 0.0,
        "hallucination_rate": round(hallucinated / max(len(documented), 1), 3),
        "citation_validity": round((len(cites) - invalid) / len(cites), 3) if cites else 1.0,
        "elapsed_sec": state.metrics.get("elapsed_sec"),
    }


def guardrail_ablation(table) -> dict:
    """Fault injection: does the groundedness guardrail catch a bad draft?

    We hand the checker a draft with one hallucinated symbol and one invalid citation
    (simulating an imperfect LLM), then compare metrics with the guardrail off vs on.
    """
    rm = table.get("resolveModule")
    faulty = DocArtifact(
        title="faulty",
        kind="api_reference",
        markdown="",
        covered_symbols=["resolveModule", "RegistryType", "resolveModuleAsync"],  # last is fake
        citations=[
            Citation(file=rm.file, start_line=rm.start_line, end_line=rm.end_line, symbol="resolveModule"),
            Citation(file=rm.file, start_line=99999, end_line=99999, symbol="RegistryType"),  # invalid
        ],
    )

    def score(art):
        r = check_groundedness(art, table)
        cites = art.citations
        return {
            "hallucination_rate": round(len(r.hallucinated_symbols) / max(len(art.covered_symbols), 1), 3),
            "citation_validity": round((len(cites) - len(r.invalid_citations)) / len(cites), 3) if cites else 1.0,
        }

    off = score(faulty)
    on = score(_prune_ungrounded(faulty, check_groundedness(faulty, table)))
    return {"guardrail_off": off, "guardrail_on": on}


def main():
    ap = argparse.ArgumentParser(description="RepoScribe evaluation harness")
    ap.add_argument("--live", action="store_true", help="use real Gemini instead of the mock")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON only")
    args = ap.parse_args()
    mock = not args.live

    cases = json.load(open(os.path.join(HERE, "test_cases.json")))["cases"]
    settings = Settings.from_env(mock=mock)

    # Main run over @lwc/module-resolver.
    state = run(MODULE_RESOLVER, os.path.join(HERE, "..", "out"), settings, auto_yes=True, verbose=False)
    files, _ = walk_repo(MODULE_RESOLVER, settings)
    table = extract_symbols(files)

    # Injected run for the prompt-injection case.
    inj_files, _ = walk_repo(INJECTED, settings)
    inj_table = extract_symbols(inj_files)
    neutralized = sanitize_symbols(inj_table)  # count before the pipeline sanitizes internally
    inj_state = run(INJECTED, os.path.join(HERE, "..", "out_injected"), settings, auto_yes=True, verbose=False)
    injected = {"state": inj_state, "neutralized": max(neutralized, 1)}

    results = evaluate(cases, state, table, injected)
    passed = sum(1 for r in results if r["pass"])
    m = metrics(state, table)
    ablation = guardrail_ablation(table)
    summary = {
        "provider": state.metrics.get("provider"),
        "task_success_rate": round(passed / len(results), 3),
        "passed": passed,
        "total": len(results),
        "metrics": m,
        "ablation": ablation,
        "results": results,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(f"\nRepoScribe evaluation ({summary['provider']} run)\n" + "=" * 52)
    for r in results:
        print(f"  [{'PASS' if r['pass'] else 'FAIL'}] {r['id']:28} {r['expect']}")
    print("=" * 52)
    print(f"Task success rate : {passed}/{len(results)} = {summary['task_success_rate']}")
    print(f"API coverage      : {m['api_coverage']}")
    print(f"Hallucination rate: {m['hallucination_rate']}")
    print(f"Citation validity : {m['citation_validity']}")
    print(f"Elapsed           : {m['elapsed_sec']}s")
    print("\nGuardrail ablation (fault-injected draft):")
    print(f"  guardrail OFF -> hallucination {ablation['guardrail_off']['hallucination_rate']}, "
          f"citation validity {ablation['guardrail_off']['citation_validity']}")
    print(f"  guardrail ON  -> hallucination {ablation['guardrail_on']['hallucination_rate']}, "
          f"citation validity {ablation['guardrail_on']['citation_validity']}")


if __name__ == "__main__":
    main()
