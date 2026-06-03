#!/usr/bin/env python3
"""Rebuild the teaching site if the source markdown was just edited.

Invoked from .claude/settings.json as a PostToolUse hook on Write|Edit.
Claude Code passes the tool event as JSON on stdin; we read it, check
whether the edited path was the teaching-guide markdown, and run
docs/build_teaching_site.py only when it was. Every other edit is a
no-op so the hook stays cheap.

Watched source: docs/PLANCK_TEACHING_GUIDE.md
Generated output: docs/index.html
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WATCHED = REPO_ROOT / "docs" / "PLANCK_TEACHING_GUIDE.md"
BUILDER = REPO_ROOT / "docs" / "build_teaching_site.py"


def edited_path_from_stdin() -> Path | None:
    """Parse the hook event JSON and return the edited file path, if any.

    Returns None when stdin is empty, unparseable, or the event does not
    name a file_path. We deliberately treat any error as "skip" rather
    than failing the hook — a hook failure noisily interrupts the session.
    """
    raw = sys.stdin.read()
    if not raw.strip():
        return None
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        return None
    tool_input = event.get("tool_input") or {}
    path_str = tool_input.get("file_path")
    if not isinstance(path_str, str) or not path_str:
        return None
    return Path(path_str).resolve()


def main() -> int:
    edited = edited_path_from_stdin()
    if edited is None or edited != WATCHED.resolve():
        return 0

    # Rebuild. Pipe both streams to /dev/null so a passing build is silent
    # in the session log; show stderr only if it fails so the user notices.
    proc = subprocess.run(
        [sys.executable, str(BUILDER)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(
            f"[teaching-site] rebuild failed (exit {proc.returncode})\n"
            f"{proc.stderr}"
        )
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
