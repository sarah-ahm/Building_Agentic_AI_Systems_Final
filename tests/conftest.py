"""Shared pytest fixtures. Everything runs offline (MockLLM), no API key needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from reposcribe.config import Settings
from reposcribe.repo import walk_repo
from reposcribe.symbols import extract_symbols

FIXTURE = Path(__file__).resolve().parent.parent / "eval" / "fixtures" / "lwc-module-resolver"


@pytest.fixture
def settings() -> Settings:
    return Settings.from_env(mock=True)


@pytest.fixture
def fixture_path() -> str:
    return str(FIXTURE)


@pytest.fixture
def fixture_files(settings):
    files, _ = walk_repo(str(FIXTURE), settings)
    return files


@pytest.fixture
def table(fixture_files):
    return extract_symbols(fixture_files)
