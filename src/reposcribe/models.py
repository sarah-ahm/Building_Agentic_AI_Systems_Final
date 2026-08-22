"""Pydantic data models shared across RepoScribe.

These double as the *structured-output schemas* handed to Gemini (exactly like the
course labs pass a pydantic model as ``response_schema``) and as the internal data
carried between pipeline stages.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SymbolKind = Literal[
    "module", "class", "interface", "function", "method", "enum", "type_alias", "constant"
]
Language = Literal["typescript", "javascript", "python", "other"]
FileRole = Literal["public_api", "types", "internal", "config", "test", "other"]
ArtifactKind = Literal["api_reference", "user_guide", "changelog"]


class Symbol(BaseModel):
    """One declaration extracted from source: the ground-truth unit of documentation."""

    name: str
    kind: SymbolKind
    language: Language
    file: str
    start_line: int
    end_line: int
    signature: str | None = None
    docstring: str | None = None
    exported: bool = False


class SymbolTable(BaseModel):
    """The oracle: every real symbol in the repo, plus the public-API surface."""

    symbols: list[Symbol] = Field(default_factory=list)
    # Names re-exported from the entry barrel (e.g. index.ts). This is the package's
    # public API. Empty means "fall back to every exported symbol".
    reexports: list[str] = Field(default_factory=list)

    def exists(self, name: str) -> bool:
        return any(s.name == name for s in self.symbols)

    def get(self, name: str) -> Symbol | None:
        return next((s for s in self.symbols if s.name == name), None)

    def public_names(self) -> set[str]:
        """Public API = re-exported names if a barrel exists, else all exported symbols."""
        if self.reexports:
            return set(self.reexports)
        return {s.name for s in self.symbols if s.exported}

    def in_file(self, file: str) -> list[Symbol]:
        return [s for s in self.symbols if s.file == file]


class SourceFile(BaseModel):
    """A source file that passed the input guardrails."""

    path: str
    language: Language
    size_bytes: int
    text: str


class Chunk(BaseModel):
    """A retrievable unit for RAG: one symbol plus a short contextual prefix."""

    id: str
    symbol_name: str
    file: str
    start_line: int
    end_line: int
    text: str


class Citation(BaseModel):
    """A pointer back into the source, verified against the SymbolTable."""

    file: str
    start_line: int
    end_line: int
    symbol: str | None = None


class RouterDecision(BaseModel):
    """Routing output: what role a file plays, with a confidence for the gate."""

    role: FileRole
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class ModulePlan(BaseModel):
    """One item in the documentation plan."""

    file: str
    role: FileRole
    priority: int = 1
    rationale: str = ""


class DocPlan(BaseModel):
    """Planner output: which artifacts to produce and which files to document."""

    artifacts: list[ArtifactKind]
    modules: list[ModulePlan]
    notes: str = ""


class DocArtifact(BaseModel):
    """A generated document plus the machine-checkable claims it makes."""

    title: str
    kind: ArtifactKind
    markdown: str
    citations: list[Citation] = Field(default_factory=list)
    covered_symbols: list[str] = Field(default_factory=list)


class Critique(BaseModel):
    """Reflection output: does the draft meet the bar, and how to fix it."""

    score: int = Field(ge=1, le=5)
    meets: bool
    issues: list[str] = Field(default_factory=list)
    fixes: list[str] = Field(default_factory=list)


class GroundednessReport(BaseModel):
    """Output guardrail result for one artifact."""

    documented: list[str] = Field(default_factory=list)
    hallucinated_symbols: list[str] = Field(default_factory=list)
    invalid_citations: list[Citation] = Field(default_factory=list)
    coverage: float = 0.0
    grounded: bool = True


class WorkspaceState(BaseModel):
    """The run's working + episodic memory, persisted to JSON at the end of a run."""

    repo_path: str
    package: str = ""
    plan: DocPlan | None = None
    routes: dict[str, RouterDecision] = Field(default_factory=dict)
    artifacts: list[DocArtifact] = Field(default_factory=list)
    groundedness: list[GroundednessReport] = Field(default_factory=list)
    skipped: list[dict] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
