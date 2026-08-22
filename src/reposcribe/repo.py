"""Filesystem tools: a guarded repo walker and a git-log reader.

``walk_repo`` is RepoScribe's first tool AND its input guardrail: it stays inside the
target root (no path traversal), skips vendored/build directories and generated
``.d.ts`` files, and rejects oversized or non-text files. ``git_log`` is the second
tool, feeding the changelog; it degrades gracefully when the target isn't a git repo.
"""

from __future__ import annotations

import os
import subprocess

from .config import SKIP_DIRS, SOURCE_EXTENSIONS, Settings
from .models import SourceFile
from .symbols import detect_language


def walk_repo(root: str, settings: Settings) -> tuple[list[SourceFile], list[dict]]:
    """Return (source files, skipped entries) for a directory tree.

    Skipped entries record *why* each file was dropped, so the guardrail is auditable.
    """
    root = os.path.realpath(root)
    if not os.path.isdir(root):
        raise NotADirectoryError(f"Not a directory: {root}")

    files: list[SourceFile] = []
    skipped: list[dict] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune noisy directories in place so os.walk never descends into them.
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            path = os.path.join(dirpath, name)
            real = os.path.realpath(path)
            # Guardrail: never follow a symlink out of the target root.
            if not real.startswith(root + os.sep) and real != root:
                skipped.append({"path": path, "reason": "outside_root"})
                continue
            ext = os.path.splitext(name)[1]
            if ext not in SOURCE_EXTENSIONS or name.endswith(".d.ts"):
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size > settings.max_file_bytes:
                skipped.append({"path": path, "reason": f"too_large ({size} bytes)"})
                continue
            try:
                text = open(path, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                skipped.append({"path": path, "reason": "not_utf8_text"})
                continue
            files.append(
                SourceFile(
                    path=os.path.relpath(real, root),
                    language=detect_language(name),
                    size_bytes=size,
                    text=text,
                )
            )

    files.sort(key=lambda f: f.path)
    return files, skipped


def git_log(root: str, limit: int = 20, subpath: str | None = None) -> list[str]:
    """Return recent one-line commits touching ``subpath``, or [] if not a git repo."""
    cmd = ["git", "-C", root, "log", f"-n{limit}", "--pretty=format:%h %ad %s", "--date=short"]
    if subpath:
        cmd += ["--", subpath]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    return [line for line in out.stdout.splitlines() if line.strip()]
