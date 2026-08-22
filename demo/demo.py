"""Scripted RepoScribe demo.

Runs the full pipeline on the frozen @lwc/module-resolver fixture and prints the plan,
the generated API reference, and the groundedness metrics. Offline by default.

    python demo/demo.py           # offline (MockLLM), no key needed
    python demo/demo.py --live    # real Gemini (needs GEMINI_API_KEY)
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from reposcribe.config import Settings  # noqa: E402
from reposcribe.pipeline import run  # noqa: E402

FIXTURE = os.path.join(HERE, "..", "eval", "fixtures", "lwc-module-resolver")
OUT = os.path.join(HERE, "..", "out")


def main():
    ap = argparse.ArgumentParser(description="RepoScribe demo")
    ap.add_argument("--live", action="store_true", help="use real Gemini instead of the mock")
    ap.add_argument("--repo", default=FIXTURE, help="target package to document")
    args = ap.parse_args()

    settings = Settings.from_env(mock=not args.live)
    print(f"\n=== RepoScribe demo ({'live Gemini' if args.live else 'offline mock'}) ===\n")

    state = run(args.repo, OUT, settings, auto_yes=True, verbose=True)

    print("\n----- Generated API reference (first 40 lines) -----")
    api = next((a for a in state.artifacts if a.kind == "api_reference"), None)
    if api:
        print("\n".join(api.markdown.splitlines()[:40]))

    print("\n----- Metrics -----")
    for k, v in state.metrics.items():
        print(f"  {k}: {v}")
    print(f"\nDocs written to: {os.path.relpath(OUT)}/  (api_reference.md, user_guide.md, changelog.md)")


if __name__ == "__main__":
    main()
