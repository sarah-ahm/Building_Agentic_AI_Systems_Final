"""Retrieval-augmented generation over the codebase (semantic memory).

Each symbol becomes one retrievable chunk (signature + docstring + a short body
slice) with a one-line contextual prefix, in the spirit of Anthropic's contextual
retrieval. The ``VectorIndex`` reproduces lab08's ``faiss.IndexFlatIP`` semantics in
plain numpy: vectors are L2-normalized, so an inner product is cosine similarity.
numpy (not faiss) keeps the dependency footprint tiny and Colab-friendly.
"""

from __future__ import annotations

import numpy as np

from .models import Chunk, SourceFile, Symbol


def _body_slice(file: SourceFile | None, start: int, end: int, max_lines: int = 40) -> str:
    if file is None:
        return ""
    lines = file.text.splitlines()
    return "\n".join(lines[start - 1 : min(end, start - 1 + max_lines)])


def chunk_symbols(symbols: list[Symbol], files: list[SourceFile]) -> list[Chunk]:
    """Build one retrievable chunk per symbol, with a contextual prefix."""
    by_path = {f.path: f for f in files}
    chunks: list[Chunk] = []
    for i, s in enumerate(symbols):
        context = f"In file {s.file}, {s.kind} `{s.name}`."
        parts = [context, s.signature or "", s.docstring or "", _body_slice(by_path.get(s.file), s.start_line, s.end_line)]
        text = "\n".join(p for p in parts if p).strip()
        chunks.append(
            Chunk(
                id=f"c{i}",
                symbol_name=s.name,
                file=s.file,
                start_line=s.start_line,
                end_line=s.end_line,
                text=text[:2000],
            )
        )
    return chunks


class VectorIndex:
    """A brute-force cosine index (numpy stand-in for faiss.IndexFlatIP)."""

    def __init__(self, embed_fn):
        self.embed_fn = embed_fn
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None

    def build(self, chunks: list[Chunk]) -> "VectorIndex":
        self.chunks = chunks
        if chunks:
            self.matrix = self.embed_fn([c.text for c in chunks], "RETRIEVAL_DOCUMENT")
        return self

    def query(self, text: str, k: int = 5) -> list[tuple[Chunk, float]]:
        """Return the top-k chunks by cosine similarity to ``text``."""
        if not self.chunks or self.matrix is None:
            return []
        q = self.embed_fn([text], "RETRIEVAL_QUERY")[0]
        scores = self.matrix @ q  # normalized vectors -> dot product == cosine
        order = np.argsort(-scores)[:k]
        return [(self.chunks[int(i)], float(scores[int(i)])) for i in order]
