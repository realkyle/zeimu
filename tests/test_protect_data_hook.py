"""Tests for the Claude Code PreToolUse hook that guards data/raw/ and eval/data/."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

HOOK_PATH = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "protect_data.py"


def _load_hook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("protect_data", HOOK_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["protect_data"] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


hook = _load_hook()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "eval" / "data").mkdir(parents=True)
    (tmp_path / "src").mkdir()
    return tmp_path.resolve()


def _bash(root: Path, command: str, cwd: Path | None = None):
    return hook.DataGuard(root).check("Bash", {"command": command}, cwd or root)


BLOCKED_COMMANDS = [
    "rm -rf data/raw",
    "rm -rf data",  # parent of data/raw
    "rm -rf .",  # project root
    "rm -rf *",  # glob in project root
    "rm data/raw/www.nta.go.jp/page.html",
    "rm -r eval",
    "rm eval/data/dev.jsonl",
    "sudo rm -rf data/raw",
    "FOO=1 rm -rf data/raw",
    "cd data && rm -rf raw",  # follows cd
    "ls && rm -rf ./data/raw/",
    "mv data/raw /tmp/old",
    "mv new.jsonl eval/data/dev.jsonl",  # overwrite
    "cp other.jsonl eval/data/dev.jsonl",
    "echo x > eval/data/dev.jsonl",
    "echo x >> data/raw/a.html",
    "echo x | tee eval/data/heldout.jsonl",
    "sed -i '' 's/a/b/' eval/data/dev.jsonl",
    "find data/raw -name '*.html' -delete",
    "find . -exec rm {} +",
    "ls data/raw | xargs rm",
    "git clean -fdx",
    "git clean -fd eval",
    "git reset --hard",
    "git reset --hard origin/main",
    "git checkout -- eval/data",
    "git checkout .",
    "git restore eval/data/dev.jsonl",
    "git rm -r eval/data",
    "uv run zeimu fetch --refresh",
    "zeimu fetch --refresh",
    "python -c \"import shutil; shutil.rmtree('data/raw')\"",
    "rm -rf 'data/raw",  # unparseable but names protected data: fail closed
]


@pytest.mark.parametrize("command", BLOCKED_COMMANDS)
def test_destructive_commands_are_blocked(root: Path, command: str) -> None:
    verdict = _bash(root, command)
    assert verdict.blocked, command
    assert verdict.reason


ALLOWED_COMMANDS = [
    "ls data/raw",
    "cat data/raw/www.nta.go.jp/page.html",
    "grep -r 相続 data/raw",
    "uv run zeimu fetch",  # cached fetch only adds missing pages
    "uv run zeimu chunk",  # writes data/chunks.jsonl, which is in git
    "uv run pytest",
    "rm -rf .venv .pytest_cache",
    "rm -rf eval/results/run1",
    "rm data/chunks.jsonl",  # committed, recoverable from git
    "cp data/raw/a.html /tmp/a.html",  # copying *out* is fine
    "cp notes.txt .",  # parent of data/raw as a copy destination does not overwrite it
    "git status -sb && git fetch",
    "git checkout main",
    "git checkout -b feat/x",
    "git clean -n",
    "git clean -fd src",
    "git reset HEAD~1",
    "uv run pytest 2>&1 | tail -5",
    "echo hi > /tmp/out.txt",
    "rm -rf $TMPDIR/x",  # unexpanded variable: unknowable, allowed (documented limitation)
]


@pytest.mark.parametrize("command", ALLOWED_COMMANDS)
def test_safe_commands_are_allowed(root: Path, command: str) -> None:
    assert not _bash(root, command).blocked, command


def test_relative_paths_resolve_against_cwd(root: Path) -> None:
    assert _bash(root, "rm -rf raw", cwd=root / "data").blocked
    assert not _bash(root, "rm -rf raw", cwd=root / "src").blocked


@pytest.mark.parametrize(
    ("relative", "blocked"),
    [
        ("data/raw/www.nta.go.jp/page.html", True),
        ("eval/data/heldout.jsonl", True),
        ("eval/data/held_out_v2.jsonl", True),
        ("eval/data/HeldOut.jsonl", True),
        ("eval/data/dev.jsonl", False),  # Claude drafts dev cases; the owner verifies them
        ("data/chunks.jsonl", False),
        ("src/zeimu/cli.py", False),
    ],
)
def test_file_edit_tools(root: Path, relative: str, blocked: bool) -> None:
    guard = hook.DataGuard(root)
    for tool in ("Write", "Edit", "MultiEdit"):
        verdict = guard.check(tool, {"file_path": str(root / relative)}, root)
        assert verdict.blocked is blocked, (tool, relative)


def test_other_tools_are_ignored(root: Path) -> None:
    verdict = hook.DataGuard(root).check("Read", {"file_path": str(root / "data/raw/a")}, root)
    assert not verdict.blocked


def _run_hook(root: Path, payload: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HOOK_PATH)],
        input=payload,
        capture_output=True,
        text=True,
        env={"CLAUDE_PROJECT_DIR": str(root)},
        check=False,
    )


def test_cli_exit_codes_follow_the_hook_protocol(root: Path) -> None:
    def payload(command: str) -> str:
        tool_input = {"command": command}
        return json.dumps({"tool_name": "Bash", "tool_input": tool_input, "cwd": str(root)})

    blocked = _run_hook(root, payload("rm -rf data/raw"))
    assert blocked.returncode == 2
    assert "protect_data.py" in blocked.stderr

    allowed = _run_hook(root, payload("ls data/raw"))
    assert allowed.returncode == 0
    assert allowed.stderr == ""


def test_cli_unreadable_input_does_not_block(root: Path) -> None:
    result = _run_hook(root, "not json")
    assert result.returncode == 1  # non-blocking error: tool calls keep working
    assert "unreadable input" in result.stderr
