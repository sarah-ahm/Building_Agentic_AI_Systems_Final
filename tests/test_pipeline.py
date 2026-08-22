"""End-to-end pipeline smoke tests (offline, MockLLM)."""

from __future__ import annotations

from reposcribe.config import Settings
from reposcribe.pipeline import run


def test_end_to_end_mock_run(fixture_path, tmp_path):
    state = run(
        fixture_path,
        str(tmp_path),
        Settings.from_env(mock=True),
        auto_yes=True,
        verbose=False,
    )
    # All three artifacts produced.
    kinds = {a.kind for a in state.artifacts}
    assert kinds == {"api_reference", "user_guide", "changelog"}
    # Fully grounded on the frozen fixture.
    assert state.metrics["coverage"] == 1.0
    assert state.metrics["hallucinated"] == 0
    assert state.metrics["invalid_citations"] == 0
    # Files written.
    assert (tmp_path / "api_reference.md").exists()
    assert (tmp_path / "user_guide.md").exists()
    assert (tmp_path / "workspace_state.json").exists()


def test_ablation_without_rag_still_runs(fixture_path, tmp_path):
    state = run(
        fixture_path,
        str(tmp_path),
        Settings.from_env(mock=True),
        use_rag=False,
        auto_yes=True,
        verbose=False,
    )
    assert state.metrics["use_rag"] is False
    assert state.metrics["n_public"] == 8


def test_rejection_writes_nothing(fixture_path, tmp_path, monkeypatch):
    # Simulate a human denying the approval gate.
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    state = run(fixture_path, str(tmp_path), Settings.from_env(mock=True), verbose=False)
    assert not (tmp_path / "api_reference.md").exists()
    assert state.artifacts  # artifacts were built, just not written
