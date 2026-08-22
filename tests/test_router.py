"""Routing: role classification and the confidence gate."""

from __future__ import annotations

from reposcribe.llm import MockLLM
from reposcribe.models import RouterDecision, SourceFile
from reposcribe.pipeline import apply_confidence_gate


def _route(path: str) -> RouterDecision:
    f = SourceFile(path=path, language="typescript", size_bytes=1, text="")
    prompt = f"FILE: {path}\n"
    return MockLLM().generate_json(prompt, RouterDecision)


def test_mock_router_classifies_by_role():
    assert _route("src/index.ts").role == "public_api"
    assert _route("src/types.ts").role == "types"
    assert _route("src/utils.ts").role == "internal"
    assert _route("src/__tests__/resolve.spec.ts").role == "test"


def test_high_confidence_decision_is_kept():
    d = RouterDecision(role="public_api", confidence=0.95, reasoning="clear")
    assert apply_confidence_gate(d, "src/index.ts") is d


def test_low_confidence_falls_back_to_heuristic():
    # Model is unsure AND wrong; the gate should override with the filename heuristic.
    d = RouterDecision(role="other", confidence=0.2, reasoning="unsure")
    gated = apply_confidence_gate(d, "src/types.ts")
    assert gated.role == "types"
    assert "heuristic" in gated.reasoning
