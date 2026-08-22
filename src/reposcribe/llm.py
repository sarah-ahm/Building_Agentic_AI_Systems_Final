"""LLM + embedding access.

``GeminiClient`` mirrors the course labs almost line-for-line: a shared ``_call``
retry wrapper (429 -> wait 60s, 503 -> wait 10s, up to 4 attempts), ``generate`` for
text, ``generate_json`` for pydantic-typed structured output via
``types.GenerateContentConfig(response_schema=...)``, and ``embed`` returning
L2-normalized vectors (so a dot product IS cosine similarity, exactly as in lab08).

``MockLLM`` implements the same interface deterministically so the whole pipeline,
the tests, and the evaluation harness run fully offline with no API key.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any

import numpy as np

from .models import Citation, Critique, DocArtifact, DocPlan, ModulePlan, RouterDecision

# --- Protocol -------------------------------------------------------------------
# Any client exposes: generate(prompt, system) -> str
#                     generate_json(prompt, schema, system) -> pydantic instance
#                     embed(texts, task) -> np.ndarray  (rows L2-normalized)


class GeminiClient:
    """Thin wrapper over google-genai, following the lab idioms."""

    provider = "gemini"

    def __init__(self, api_key: str, model: str, embed_model: str):
        from google import genai  # imported lazily so offline/mock runs need no SDK

        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.embed_model = embed_model

    def _call(self, **kwargs):
        """Call Gemini, retrying on transient 429 (rate limit) / 503 (overloaded)."""
        from google.genai import errors

        for attempt in range(4):  # first call + up to 3 retries
            try:
                return self.client.models.generate_content(model=self.model, **kwargs)
            except errors.ClientError as e:
                if getattr(e, "code", None) == 429 and attempt < 3:
                    print("  Rate limited (429). Waiting 60s, then retrying...")
                    time.sleep(60)
                else:
                    raise
            except errors.ServerError as e:
                if getattr(e, "code", None) == 503 and attempt < 3:
                    print("  Model overloaded (503). Waiting 10s, then retrying...")
                    time.sleep(10)
                else:
                    raise

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Return plain text for a prompt."""
        return (self._call(contents=prompt, config={"system_instruction": system}).text or "").strip()

    def generate_json(self, prompt: str, schema: Any, system: str | None = None):
        """Return a schema-validated object (pydantic model or typed list)."""
        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0,
        )
        return self._call(contents=prompt, config=cfg).parsed

    def embed(self, texts: list[str], task: str) -> np.ndarray:
        """Embed strings and L2-normalize, so dot product == cosine similarity."""
        from google.genai import types

        resp = self.client.models.embed_content(
            model=self.embed_model,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task),
        )
        vecs = np.array([e.values for e in resp.embeddings], dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


# --- Deterministic offline stand-in --------------------------------------------

_SYMBOL_RE = re.compile(
    r"^SYMBOL:\s*(?P<name>.+?)\s*\|\s*(?P<kind>.+?)\s*\|\s*"
    r"(?P<file>.+?):(?P<start>\d+)-(?P<end>\d+)(?:\s*\|\s*(?P<sig>.*))?$",
    re.MULTILINE,
)
_MODULE_RE = re.compile(
    r"^MODULE:\s*(?P<file>.+?)\s*\|\s*(?P<role>\w+)\s*\|\s*(?P<count>\d+)$", re.MULTILINE
)


class MockLLM:
    """Deterministic client so tests and `--mock` runs need no network or key.

    It reads small machine-readable hint lines that the agent prompts embed
    (``SYMBOL:``, ``MODULE:``, ``FILE:``) and produces faithful, grounded output.
    """

    provider = "mock"

    def generate(self, prompt: str, system: str | None = None) -> str:
        return "OK"

    def generate_json(self, prompt: str, schema: Any, system: str | None = None):
        name = getattr(schema, "__name__", str(schema))
        if name == "RouterDecision":
            return self._route(prompt)
        if name == "DocPlan":
            return self._plan(prompt)
        if name == "DocArtifact":
            return self._write(prompt)
        if name == "Critique":
            return Critique(score=5, meets=True, issues=[], fixes=[])
        raise ValueError(f"MockLLM has no handler for schema {name!r}")

    def embed(self, texts: list[str], task: str) -> np.ndarray:
        return _bow_embed(texts)

    # -- per-schema handlers --
    @staticmethod
    def _route(prompt: str) -> RouterDecision:
        m = re.search(r"^FILE:\s*(.+)$", prompt, re.MULTILINE)
        path = (m.group(1).strip() if m else "").lower()
        return RouterDecision(role=_role_for_path(path), confidence=0.9, reasoning="mock heuristic")

    @staticmethod
    def _plan(prompt: str) -> DocPlan:
        modules = [
            ModulePlan(file=mm.group("file"), role=mm.group("role"), priority=int(mm.group("count")))
            for mm in _MODULE_RE.finditer(prompt)
        ]
        return DocPlan(
            artifacts=["api_reference", "user_guide", "changelog"],
            modules=modules,
            notes="mock plan",
        )

    @staticmethod
    def _write(prompt: str) -> DocArtifact:
        title_m = re.search(r"^TITLE:\s*(.+)$", prompt, re.MULTILINE)
        kind_m = re.search(r"^KIND:\s*(\w+)$", prompt, re.MULTILINE)
        title = title_m.group(1).strip() if title_m else "Reference"
        kind = kind_m.group(1).strip() if kind_m else "api_reference"

        lines = [f"# {title}", ""]
        citations: list[Citation] = []
        covered: list[str] = []
        for sm in _SYMBOL_RE.finditer(prompt):
            name, kind_s = sm.group("name"), sm.group("kind")
            file, start, end = sm.group("file"), int(sm.group("start")), int(sm.group("end"))
            sig = (sm.group("sig") or "").strip()
            lines += [f"## `{name}` ({kind_s})", ""]
            if sig:
                lines += ["```", sig, "```", ""]
            lines += [f"Defined at `{file}:{start}`.", ""]
            citations.append(Citation(file=file, start_line=start, end_line=end, symbol=name))
            covered.append(name)
        if not covered:
            lines += ["_No symbols provided._", ""]
        return DocArtifact(
            title=title,
            kind=kind,
            markdown="\n".join(lines),
            citations=citations,
            covered_symbols=covered,
        )


def _role_for_path(path: str) -> str:
    """Filename -> file role, used by the mock router (and as a routing fallback)."""
    base = path.rsplit("/", 1)[-1]
    if ".spec." in base or ".test." in base or "__tests__" in path:
        return "test"
    if base in {"index.ts", "index.js", "index.mjs"}:
        return "public_api"
    if base in {"types.ts", "types.js"}:
        return "types"
    if base.endswith(("package.json", ".config.js", ".config.ts", ".config.mjs")):
        return "config"
    return "internal"


def _bow_embed(texts: list[str], dim: int = 256) -> np.ndarray:
    """Deterministic bag-of-words embedding (hashed tokens), L2-normalized.

    Captures real lexical overlap, so RAG retrieval is stable offline.
    """
    out = np.zeros((len(texts), dim), dtype="float32")
    for i, t in enumerate(texts):
        for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", t.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim
            out[i, h] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def make_llm(settings) -> Any:
    """Factory: a MockLLM in mock mode, otherwise a real GeminiClient."""
    if settings.mock:
        return MockLLM()
    if not settings.api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY. Add it to a local .env or Colab Secrets, "
            "or run with --mock for an offline dry run."
        )
    return GeminiClient(settings.api_key, settings.model, settings.embed_model)
