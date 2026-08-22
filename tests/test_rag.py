"""Chunking and the numpy vector index (RAG / semantic memory)."""

from __future__ import annotations

from reposcribe.llm import MockLLM
from reposcribe.rag import VectorIndex, chunk_symbols


def test_one_chunk_per_symbol(table, fixture_files):
    chunks = chunk_symbols(table.symbols, fixture_files)
    assert len(chunks) == len(table.symbols)
    assert all(c.text for c in chunks)


def test_chunk_carries_location(table, fixture_files):
    chunks = chunk_symbols(table.symbols, fixture_files)
    rm = next(c for c in chunks if c.symbol_name == "resolveModule")
    assert rm.file.endswith("resolve-module.ts")
    assert rm.start_line > 0


def test_retrieval_finds_relevant_symbol(table, fixture_files):
    index = VectorIndex(MockLLM().embed).build(chunk_symbols(table.symbols, fixture_files))
    hits = index.query("resolveModule importee dirname config", k=3)
    assert hits, "expected at least one retrieval hit"
    assert "resolveModule" in {c.symbol_name for c, _ in hits}


def test_empty_index_returns_nothing():
    index = VectorIndex(MockLLM().embed).build([])
    assert index.query("anything", k=3) == []
