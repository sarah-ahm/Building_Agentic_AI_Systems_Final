"""The orchestrator: wires every stage together, by hand (in the spirit of lab05_A).

Flow:  walk (guardrail) -> extract symbols (oracle) -> sanitize (guardrail) ->
build RAG index -> plan -> route (parallel) -> write API sections (parallel, RAG) ->
reflect -> groundedness guard -> write user guide -> changelog -> HITL gate -> persist.

Concurrency uses ``asyncio.gather`` + ``asyncio.to_thread`` (lab idiom) via a helper
that also works inside Colab's already-running event loop.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import functools
import os
import time

from . import agents
from .config import Settings
from .guardrails import check_groundedness, sanitize_symbols
from .llm import _role_for_path, make_llm
from .models import (
    DocArtifact,
    GroundednessReport,
    RouterDecision,
    Symbol,
    WorkspaceState,
)
from .rag import VectorIndex, chunk_symbols
from .repo import git_log, walk_repo
from .symbols import extract_symbols

CONFIDENCE_THRESHOLD = 0.5


def apply_confidence_gate(decision: RouterDecision, path: str) -> RouterDecision:
    """Routing gate: below the confidence threshold, trust a filename heuristic instead."""
    if decision.confidence < CONFIDENCE_THRESHOLD:
        return RouterDecision(
            role=_role_for_path(path.lower()),
            confidence=decision.confidence,
            reasoning="low confidence -> filename heuristic",
        )
    return decision


# --- concurrency helper (Colab-safe) -------------------------------------------

def _run_async(coro):
    """Run a coroutine, whether or not an event loop is already running (Colab)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        return ex.submit(asyncio.run, coro).result()


def parallel_map(callables, concurrency: int = 4):
    """Run zero-arg callables concurrently (fan-out/fan-in), bounded by a semaphore."""

    async def _amap():
        sem = asyncio.Semaphore(concurrency)

        async def one(fn):
            async with sem:
                return await asyncio.to_thread(fn)

        return await asyncio.gather(*[one(fn) for fn in callables])

    return _run_async(_amap())


# --- main pipeline -------------------------------------------------------------

def run(
    repo_path: str,
    out_dir: str,
    settings: Settings,
    *,
    use_rag: bool = True,
    use_reflection: bool = True,
    auto_yes: bool = False,
    verbose: bool = True,
) -> WorkspaceState:
    """Generate grounded docs for ``repo_path`` and (after approval) write to ``out_dir``."""
    t0 = time.perf_counter()
    llm = make_llm(settings)
    package = os.path.basename(os.path.normpath(repo_path))
    state = WorkspaceState(repo_path=repo_path, package=package)

    def log(msg: str) -> None:
        state.trace.append(msg)
        if verbose:
            print(msg)

    # 1. Walk the repo (input guardrail).
    files, skipped = walk_repo(repo_path, settings)
    state.skipped = skipped
    log(f"[walk] {len(files)} source files, {len(skipped)} skipped")

    # 2. Extract the ground-truth SymbolTable (tool).
    table = extract_symbols(files)
    log(f"[symbols] {len(table.symbols)} symbols; {len(table.public_names())} public")

    # 3. Sanitize untrusted source comments (input guardrail).
    neutralized = sanitize_symbols(table)
    if neutralized:
        log(f"[guardrail] neutralized {neutralized} possible prompt-injection docstring(s)")

    # 4. Build the RAG index (semantic memory).
    index = None
    if use_rag:
        index = VectorIndex(llm.embed).build(chunk_symbols(table.symbols, files))
        log(f"[rag] indexed {len(index.chunks)} chunks")

    # 5. Plan (PLANNING).
    state.plan = agents.plan_docs(llm, table, files)
    log(f"[plan] artifacts={state.plan.artifacts}")

    # 6. Route every file (ROUTING, parallel) with a confidence gate.
    def route(f):
        names = [s.name for s in table.in_file(f.path)]
        return f.path, apply_confidence_gate(agents.route_file(llm, f, names), f.path)

    for path, decision in parallel_map([functools.partial(route, f) for f in files], settings.concurrency):
        state.routes[path] = decision
    test_files = {p for p, d in state.routes.items() if d.role == "test"}
    log(f"[route] roles={ {d.role for d in state.routes.values()} }; {len(test_files)} test file(s)")

    # 7. Resolve the public API to concrete symbol definitions (skip test-file defs).
    public_symbols = _public_definitions(table, test_files)
    log(f"[select] documenting {len(public_symbols)} public symbols")

    # PLANNING drives execution: build exactly the artifacts the planner chose, and
    # document files in the planner's priority order (more symbols -> higher priority).
    wanted = list(state.plan.artifacts) if (state.plan and state.plan.artifacts) else [
        "api_reference",
        "user_guide",
        "changelog",
    ]
    priority = {m.file: m.priority for m in state.plan.modules} if state.plan else {}
    built: dict[str, DocArtifact] = {}

    # 8. API reference — parallel section writers grouped by file, in plan priority order.
    if "api_reference" in wanted:
        groups = _group_by_file(public_symbols)
        groups.sort(key=lambda g: (-priority.get(g[0], 0), g[0]))

        def write_group(file_and_syms):
            file, syms = file_and_syms
            art = agents.write_section(
                llm,
                title=f"API Reference — {file}",
                kind="api_reference",
                symbols=syms,
                index=index,
                use_rag=use_rag,
                instructions="Write one reference entry per symbol.",
            )
            # 9. Reflection: one critique + revise pass.
            if use_reflection:
                crit = agents.critique(llm, art)
                if not crit.meets:
                    log(f"[reflect] revising '{file}' section: {crit.issues[:2]}")
                    art = agents.write_section(
                        llm,
                        title=f"API Reference — {file}",
                        kind="api_reference",
                        symbols=syms,
                        index=index,
                        use_rag=use_rag,
                        instructions="Write one reference entry per symbol.",
                        feedback="; ".join(crit.fixes) or "improve clarity and accuracy",
                    )
            return art

        sections = parallel_map(
            [functools.partial(write_group, g) for g in groups], settings.concurrency
        )

        # 10. Groundedness guard on every section; drop any ungrounded claim.
        api_sections: list[DocArtifact] = []
        for art in sections:
            report = check_groundedness(art, table)
            if not report.grounded:
                art = _prune_ungrounded(art, report)
                report = check_groundedness(art, table)
                log(f"[guardrail] pruned ungrounded claims in '{art.title}'")
            state.groundedness.append(report)
            api_sections.append(art)

        built["api_reference"] = _assemble(
            api_sections, title=f"API Reference — {package}", kind="api_reference"
        )

    # 11. User guide (RAG-grounded), also guarded.
    if "user_guide" in wanted:
        user_guide = agents.write_section(
            llm,
            title=f"{package} — User Guide",
            kind="user_guide",
            symbols=public_symbols,
            index=index,
            use_rag=use_rag,
            instructions=(
                "Write a short user guide for this package's public API: what it does, how to "
                "call the main entry point, and a minimal usage example. Reference symbols by name."
            ),
        )
        ug_report = check_groundedness(user_guide, table)
        if not ug_report.grounded:
            user_guide = _prune_ungrounded(user_guide, ug_report)
            ug_report = check_groundedness(user_guide, table)
        state.groundedness.append(ug_report)
        built["user_guide"] = user_guide

    # 12. Changelog (git tool).
    if "changelog" in wanted:
        commits = git_log(repo_path, subpath=None) or git_log(os.getcwd(), subpath=repo_path)
        built["changelog"] = agents.build_changelog(commits, package)

    # Assemble in the planner's chosen order, skipping any artifact it did not request.
    state.artifacts = [built[k] for k in wanted if k in built]

    # metrics
    elapsed = time.perf_counter() - t0
    state.metrics = {
        "n_files": len(files),
        "n_symbols": len(table.symbols),
        "n_public": len(table.public_names()),
        "coverage": round(_coverage(state, table), 4),
        "hallucinated": sum(len(r.hallucinated_symbols) for r in state.groundedness),
        "invalid_citations": sum(len(r.invalid_citations) for r in state.groundedness),
        "elapsed_sec": round(elapsed, 2),
        "provider": llm.provider,
        "use_rag": use_rag,
        "use_reflection": use_reflection,
    }
    log(f"[metrics] {state.metrics}")

    # 13. Human-in-the-loop approval gate.
    if not auto_yes and not _approve(state):
        log("[hitl] rejected — nothing written")
        return state

    # 14. Persist artifacts + episodic memory.
    _write_outputs(out_dir, state)
    log(f"[write] wrote docs to {out_dir}")
    return state


# --- helpers -------------------------------------------------------------------

def _public_definitions(table, test_files: set[str]) -> list[Symbol]:
    """Concrete Symbol for each public name, preferring a real definition over a barrel."""
    out: list[Symbol] = []
    for name in sorted(table.public_names()):
        candidates = [s for s in table.symbols if s.name == name and s.file not in test_files]
        if not candidates:
            continue
        # Prefer a symbol with a signature/docstring (the definition) over a bare re-export.
        candidates.sort(key=lambda s: (s.docstring is None, s.signature is None))
        out.append(candidates[0])
    return out


def _group_by_file(symbols: list[Symbol]) -> list[tuple[str, list[Symbol]]]:
    groups: dict[str, list[Symbol]] = {}
    for s in symbols:
        groups.setdefault(s.file, []).append(s)
    return sorted(groups.items())


def _prune_ungrounded(art: DocArtifact, report: GroundednessReport) -> DocArtifact:
    """Remove hallucinated symbols and invalid citations so the artifact is grounded."""
    bad_syms = set(report.hallucinated_symbols)
    bad_cites = {(c.file, c.start_line) for c in report.invalid_citations}
    return art.model_copy(
        update={
            "covered_symbols": [s for s in art.covered_symbols if s not in bad_syms],
            "citations": [c for c in art.citations if (c.file, c.start_line) not in bad_cites],
        }
    )


def _demote_headings(md: str) -> str:
    """Push every markdown heading one level deeper so sections nest under one H1."""
    out = []
    for line in md.splitlines():
        out.append("#" + line if line.lstrip().startswith("#") else line)
    return "\n".join(out)


def _assemble(sections: list[DocArtifact], *, title: str, kind) -> DocArtifact:
    md = [f"# {title}", ""]
    citations = []
    covered = []
    for s in sections:
        md.append(_demote_headings(s.markdown.strip()))
        md.append("")
        citations += s.citations
        covered += s.covered_symbols
    return DocArtifact(
        title=title,
        kind=kind,
        markdown="\n".join(md),
        citations=citations,
        covered_symbols=list(dict.fromkeys(covered)),
    )


def _coverage(state: WorkspaceState, table) -> float:
    public = table.public_names()
    if not public:
        return 0.0
    documented = set()
    for art in state.artifacts:
        if art.kind == "api_reference":
            documented |= set(art.covered_symbols)
    return len(documented & public) / len(public)


def _approve(state: WorkspaceState) -> bool:
    print("\n" + "=" * 60)
    print(f"RepoScribe is ready to write {len(state.artifacts)} document(s) for '{state.package}':")
    for art in state.artifacts:
        print(f"  - {art.kind:14} {len(art.covered_symbols):>2} symbols, {len(art.citations):>2} citations")
    print(f"Coverage: {state.metrics.get('coverage')}  |  Hallucinated: {state.metrics.get('hallucinated')}")
    print("=" * 60)
    try:
        return input("Write these files? [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:  # non-interactive environment
        return False


def _write_outputs(out_dir: str, state: WorkspaceState) -> None:
    os.makedirs(out_dir, exist_ok=True)
    names = {"api_reference": "api_reference.md", "user_guide": "user_guide.md", "changelog": "changelog.md"}
    for art in state.artifacts:
        with open(os.path.join(out_dir, names[art.kind]), "w", encoding="utf-8") as fh:
            fh.write(art.markdown.rstrip() + "\n")
    with open(os.path.join(out_dir, "workspace_state.json"), "w", encoding="utf-8") as fh:
        fh.write(state.model_dump_json(indent=2))


# --- CLI -----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="reposcribe", description="Agentic documentation generator")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="generate documentation for a repo")
    r.add_argument("--repo", required=True, help="path to the target package/repo")
    r.add_argument("--out", default="./out", help="output directory for generated docs")
    r.add_argument("--no-rag", action="store_true", help="ablation: skip RAG grounding")
    r.add_argument("--no-reflection", action="store_true", help="ablation: skip the critique/revise pass")
    r.add_argument("--yes", action="store_true", help="skip the human approval gate")
    r.add_argument("--mock", action="store_true", help="use the offline deterministic LLM (no API key)")
    r.add_argument("--model", default=None, help="override the Gemini model")

    args = parser.parse_args(argv)
    settings = Settings.from_env(mock=args.mock)
    if args.model:
        settings.model = args.model

    try:
        run(
            args.repo,
            args.out,
            settings,
            use_rag=not args.no_rag,
            use_reflection=not args.no_reflection,
            auto_yes=args.yes,
        )
    except (NotADirectoryError, FileNotFoundError) as e:
        parser.error(
            f"{e}. Pass an existing package directory with --repo "
            "(e.g. eval/fixtures/lwc-module-resolver)."
        )
    except RuntimeError as e:  # e.g. missing API key without --mock
        parser.error(str(e))


if __name__ == "__main__":
    main()
