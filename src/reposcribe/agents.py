"""The agent stages: plan, route, write, and critique.

Each is a small function that builds a prompt and asks the LLM for a pydantic-typed
result (the lab ``generate_json`` idiom). Prompts embed compact ``SYMBOL:`` / ``MODULE:``
/ ``FILE:`` hint lines so the offline MockLLM can produce faithful, grounded output.
"""

from __future__ import annotations

from .models import (
    ArtifactKind,
    DocArtifact,
    DocPlan,
    RouterDecision,
    SourceFile,
    Symbol,
    SymbolTable,
)

PLANNER_SYSTEM = "You plan documentation for a software package. Be decisive and concise."
ROUTER_SYSTEM = "You classify the role a source file plays in a package."
WRITER_SYSTEM = (
    "You are a precise technical writer. Document ONLY the symbols you are given. "
    "Never invent, rename, or add symbols. Cite each symbol's source as file:line. "
    "Base every statement on the provided signatures and context."
)
CRITIC_SYSTEM = "You review documentation drafts strictly for accuracy and clarity."


# --- Planning ------------------------------------------------------------------

def plan_docs(llm, table: SymbolTable, files: list[SourceFile]) -> DocPlan:
    """PLANNING stage: choose artifacts and order the files worth documenting."""
    module_lines = "\n".join(
        f"MODULE: {f.path} | other | {len(table.in_file(f.path))}" for f in files
    )
    prompt = (
        "Plan documentation for this package. Choose which artifacts to produce "
        "(api_reference, user_guide, changelog) and list the modules in priority order "
        "(more symbols usually means higher priority).\n\n"
        f"Files and symbol counts:\n{module_lines}"
    )
    return llm.generate_json(prompt, DocPlan, system=PLANNER_SYSTEM)


# --- Routing -------------------------------------------------------------------

def route_file(llm, file: SourceFile, symbol_names: list[str]) -> RouterDecision:
    """ROUTING stage: classify one file's role, with a confidence for the gate."""
    prompt = (
        f"FILE: {file.path}\n"
        f"Language: {file.language}\n"
        f"Symbols: {', '.join(symbol_names[:10]) or '(none)'}\n\n"
        "Classify the file's role as one of: public_api (an entry/barrel that defines the "
        "package's exported surface), types (type/interface declarations), internal (helpers "
        "not part of the public API), config, test, or other. Return a confidence in [0,1]."
    )
    return llm.generate_json(prompt, RouterDecision, system=ROUTER_SYSTEM)


# --- Writing (RAG-grounded) ----------------------------------------------------

def _retrieve_context(index, symbols: list[Symbol], k: int = 3) -> str:
    if index is None:
        return ""
    seen: set[str] = set()
    blocks: list[str] = []
    for s in symbols:
        for chunk, _score in index.query(f"{s.kind} {s.name} {s.signature or ''}", k=k):
            if chunk.id not in seen:
                seen.add(chunk.id)
                blocks.append(f"[{chunk.file}:{chunk.start_line}] {chunk.text}")
    return "\n\n".join(blocks[:8])


def write_section(
    llm,
    *,
    title: str,
    kind: ArtifactKind,
    symbols: list[Symbol],
    index=None,
    use_rag: bool = True,
    instructions: str = "",
    feedback: str = "",
) -> DocArtifact:
    """WRITING stage (also the RAG consumer): produce a cited DocArtifact."""
    symbol_lines = "\n".join(
        f"SYMBOL: {s.name} | {s.kind} | {s.file}:{s.start_line}-{s.end_line} | {s.signature or ''}"
        for s in symbols
    )
    context = _retrieve_context(index, symbols) if use_rag else ""
    feedback_block = f"\n\nA reviewer asked you to fix: {feedback}" if feedback else ""
    prompt = (
        f"TITLE: {title}\nKIND: {kind}\n\n"
        f"{instructions or 'Write reference entries for EXACTLY these symbols.'} "
        "Give each a short, accurate description and cite its source as file:line in the "
        "citations field. Set covered_symbols to exactly the symbol names you documented. "
        "Do not add symbols that are not listed.\n\n"
        f"SYMBOLS TO DOCUMENT:\n{symbol_lines}\n\n"
        f"CONTEXT (retrieved source):\n{context or '(none)'}"
        f"{feedback_block}"
    )
    return llm.generate_json(prompt, DocArtifact, system=WRITER_SYSTEM)


# --- Reflection ----------------------------------------------------------------

def critique(llm, artifact: DocArtifact):
    """REFLECTION stage: score a draft and say what to fix."""
    from .models import Critique

    prompt = (
        "Review this documentation draft. Check that it reads clearly and that each entry "
        "corresponds to a documented symbol with a citation. Score 1-5; set meets=true only "
        "if score >= 4. List concrete issues and fixes.\n\nDRAFT:\n" + artifact.markdown[:4000]
    )
    return llm.generate_json(prompt, Critique, system=CRITIC_SYSTEM)


# --- Changelog (deterministic, no LLM) -----------------------------------------

def build_changelog(entries: list[str], package: str) -> DocArtifact:
    """Assemble a changelog artifact from git one-liners (no LLM call needed)."""
    if entries:
        body = "\n".join(f"- {e}" for e in entries)
        md = f"# Changelog — {package}\n\nRecent commits touching this package:\n\n{body}\n"
    else:
        md = (
            f"# Changelog — {package}\n\n"
            "_No version-control history was available for this snapshot, so no changelog "
            "could be derived._\n"
        )
    return DocArtifact(title=f"Changelog — {package}", kind="changelog", markdown=md)
