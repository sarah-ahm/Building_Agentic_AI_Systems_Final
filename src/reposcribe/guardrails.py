"""Guardrails: input sanitization and output groundedness.

Input side (lethal-trifecta defense): source comments are untrusted text that flows
into an LLM prompt, so we neutralize embedded instructions ("ignore previous
instructions...") before they reach the model.

Output side: every documented symbol must exist in the SymbolTable and every citation
must point at a real symbol's line range. This is the check that turns "are the docs
correct?" into a hard, automatable metric.
"""

from __future__ import annotations

import re

from .models import Citation, DocArtifact, GroundednessReport, SymbolTable

_INJECTION_PATTERNS = [
    r"ignore (all )?(the )?previous instructions",
    r"disregard (the )?(above|previous)",
    r"you are now",
    r"system prompt",
    r"reveal (your|the) (system )?prompt",
    r"exfiltrate|send .* to https?://",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def contains_injection(text: str | None) -> bool:
    return bool(text) and bool(_INJECTION_RE.search(text))


def sanitize_text(text: str | None) -> tuple[str | None, bool]:
    """Return (cleaned_text, was_flagged). Suspicious spans are redacted, not obeyed."""
    if not text:
        return text, False
    if not _INJECTION_RE.search(text):
        return text, False
    cleaned = _INJECTION_RE.sub("[redacted: possible instruction in source text]", text)
    return cleaned, True


def sanitize_symbols(table: SymbolTable) -> int:
    """Neutralize injection attempts in symbol docstrings. Returns count neutralized."""
    flagged = 0
    for s in table.symbols:
        cleaned, hit = sanitize_text(s.docstring)
        if hit:
            s.docstring = cleaned
            flagged += 1
    return flagged


def _citation_valid(c: Citation, table: SymbolTable) -> bool:
    """Valid if the citation points inside a real symbol's line range in a real file."""
    for s in table.symbols:
        if s.file == c.file and s.start_line <= c.start_line <= s.end_line:
            return True
    return False


def check_groundedness(artifact: DocArtifact, table: SymbolTable) -> GroundednessReport:
    """Verify a generated artifact against the ground-truth SymbolTable."""
    documented = list(dict.fromkeys(artifact.covered_symbols))  # de-dupe, keep order
    hallucinated = [name for name in documented if not table.exists(name)]
    invalid = [c for c in artifact.citations if not _citation_valid(c, table)]

    public = table.public_names()
    covered_public = {name for name in documented if name in public}
    coverage = len(covered_public) / len(public) if public else 0.0

    return GroundednessReport(
        documented=documented,
        hallucinated_symbols=hallucinated,
        invalid_citations=invalid,
        coverage=round(coverage, 4),
        grounded=(not hallucinated and not invalid),
    )
