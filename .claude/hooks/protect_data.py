#!/usr/bin/env python3
"""Claude Code PreToolUse hook: block tool calls that would destroy irreplaceable data.

Protected:
- ``data/raw/``: cached NTA HTML snapshots. Gitignored, and re-fetching returns today's
  page, not the version the chunks cite, so a deleted snapshot is gone for good.
- ``eval/data/``: evaluation cases. Uncommitted cases are not in git yet, and the
  held-out set (any file whose name contains "heldout") is written by the owner alone.

Claude Code sends the pending tool call as JSON on stdin. Exit code 2 blocks the call and
shows stderr to Claude; exit 0 allows it. A CLAUDE.md rule is advice, this is enforcement.
It matches command shapes, so it cannot catch every possible command; it is a seatbelt,
not a sandbox. Standard library only, so it runs on any ``python3`` without ``uv``.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

PROTECTED_DIRS = (Path("data/raw"), Path("eval/data"))
HELDOUT_NAME = re.compile(r"held[-_ ]?out", re.IGNORECASE)

_SEPARATORS = frozenset({";", "&&", "||", "|", "&", "(", ")"})
_REDIRECTS = frozenset({">", ">>", ">|", "&>", "&>>"})
_WRAPPERS = frozenset({"sudo", "command", "nohup", "time", "env", "exec"})
_DELETERS = frozenset({"rm", "rmdir", "unlink", "shred", "truncate"})
_COPIERS = frozenset({"cp", "rsync", "install", "ln", "tee"})
_FILE_EDIT_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit"})
# Inline scripts (python -c, perl -e, ...) that name a protected path and a destructive call.
_SCRIPT_DESTROYS = re.compile(r"rmtree|unlink|remove\(|rename|write_text|write_bytes")
_INTERPRETER = re.compile(r"python[\d.]*|perl|ruby|node|sh|bash|zsh")


@dataclass(frozen=True)
class Verdict:
    """Outcome of checking one tool call: ``reason`` is None when the call is allowed."""

    reason: str | None = None

    @property
    def blocked(self) -> bool:
        """True when the tool call must not run."""
        return self.reason is not None


ALLOW = Verdict()


class DataGuard:
    """Decides whether a Claude Code tool call would delete or overwrite protected data."""

    def __init__(self, project_root: Path) -> None:
        """Bind the guard to a project; protected dirs are resolved against ``project_root``."""
        self._root = project_root.resolve()
        self._protected = tuple((self._root / d).resolve() for d in PROTECTED_DIRS)

    def check(self, tool_name: str, tool_input: dict[str, object], cwd: Path) -> Verdict:
        """Return the verdict for one pending tool call."""
        if tool_name == "Bash":
            command = tool_input.get("command")
            return self.check_command(command, cwd) if isinstance(command, str) else ALLOW
        if tool_name in _FILE_EDIT_TOOLS:
            path = tool_input.get("file_path") or tool_input.get("notebook_path")
            return self.check_file_edit(path, cwd) if isinstance(path, str) else ALLOW
        return ALLOW

    def check_file_edit(self, file_path: str, cwd: Path) -> Verdict:
        """Block edits to cached raw pages and to the owner-only held-out set."""
        path = self._resolve(file_path, cwd)
        raw_dir, eval_dir = self._protected
        if _is_within(path, raw_dir):
            return Verdict(f"{file_path} is a cached NTA snapshot (data/raw/ is read-only)")
        if _is_within(path, eval_dir) and HELDOUT_NAME.search(path.name):
            return Verdict(f"{file_path} is the held-out set, which only the owner writes")
        return ALLOW

    def check_command(self, command: str, cwd: Path) -> Verdict:
        """Block shell commands that delete, move, overwrite or reset protected data."""
        mentions_protected = any(str(d) in command for d in PROTECTED_DIRS)
        try:
            tokens = _tokenize(command)
        except ValueError:
            # Unparseable (e.g. unbalanced quotes): fail closed only if it names protected data.
            if mentions_protected:
                return Verdict("could not parse a command that names protected data")
            return ALLOW
        for segment in _split_segments(tokens):
            verdict, cwd = self._check_segment(segment, cwd, mentions_protected)
            if verdict.blocked:
                return verdict
        return ALLOW

    def _check_segment(
        self, segment: list[str], cwd: Path, pipeline_mentions_protected: bool
    ) -> tuple[Verdict, Path]:
        """Check one simple command; also returns the cwd after it (to follow ``cd``)."""
        words, redirect_targets = _strip_redirects(segment)
        for target in redirect_targets:
            if self._hits(target, cwd, include_ancestors=False):
                return Verdict(f"redirect overwrites protected path {target}"), cwd
        words = _strip_wrappers(words)
        if not words:
            return ALLOW, cwd
        program, args = os.path.basename(words[0]), words[1:]
        if _runs_inline_script(program, args):
            # Only for interpreters: prose that merely mentions these words (a commit
            # message, a PR body) must not be blocked.
            script = " ".join(args)
            names_protected = any(str(d) in script for d in PROTECTED_DIRS)
            if names_protected and _SCRIPT_DESTROYS.search(script):
                return Verdict("inline script deletes or rewrites a protected path"), cwd
        if program == "cd":
            return ALLOW, self._resolve(args[0] if args else str(self._root), cwd)
        if program == "xargs":
            # `ls data/raw | xargs rm`: the paths arrive on stdin, so judge by the pipeline.
            inner = [a for a in args if not a.startswith("-")]
            if inner and inner[0] in _DELETERS | {"mv"} and pipeline_mentions_protected:
                return Verdict(f"xargs {inner[0]} on protected data"), cwd
            return ALLOW, cwd
        return self._check_program(program, args, cwd), cwd

    def _check_program(self, program: str, args: list[str], cwd: Path) -> Verdict:
        """Apply the per-program rules to one command's arguments."""
        paths = [a for a in args if not a.startswith("-")]
        if program in _DELETERS:
            hit = self._first_hit(paths, cwd, include_ancestors=program != "truncate")
            return Verdict(f"{program} would delete protected path {hit}") if hit else ALLOW
        if program == "mv" and paths:
            *sources, dest = paths
            hit = self._first_hit(sources, cwd, include_ancestors=True)
            hit = hit or self._first_hit([dest], cwd, include_ancestors=False)
            return Verdict(f"mv would move or overwrite protected path {hit}") if hit else ALLOW
        if program in _COPIERS and paths:
            dests = paths if program == "tee" else paths[-1:]
            hit = self._first_hit(dests, cwd, include_ancestors=False)
            return Verdict(f"{program} would overwrite protected path {hit}") if hit else ALLOW
        if program == "sed" and any(a.startswith("-i") or a == "--in-place" for a in args):
            hit = self._first_hit(paths, cwd, include_ancestors=False)
            return Verdict(f"sed -i would rewrite protected path {hit}") if hit else ALLOW
        if program == "find" and ("-delete" in args or _find_execs_delete(args)):
            hit = self._first_hit(paths, cwd, include_ancestors=True)
            return Verdict(f"find would delete under protected path {hit}") if hit else ALLOW
        if program == "git":
            return self._check_git(args, cwd)
        runs_zeimu = program == "zeimu" or args[:2] == ["run", "zeimu"]
        if runs_zeimu and "fetch" in args and "--refresh" in args:
            return Verdict("zeimu fetch --refresh overwrites the data/raw/ snapshot cache")
        return ALLOW

    def _check_git(self, args: list[str], cwd: Path) -> Verdict:
        """Block git subcommands that discard untracked or uncommitted protected files."""
        sub = next((a for a in args if not a.startswith("-")), None)
        rest = args[args.index(sub) + 1 :] if sub else []
        paths = [a for a in rest if not a.startswith("-")]
        if sub == "clean":
            if "-n" in rest or "--dry-run" in rest:
                return ALLOW
            if not paths or self._first_hit(paths, cwd, include_ancestors=True):
                return Verdict("git clean would delete untracked or ignored protected files")
        if sub == "reset" and "--hard" in rest:
            return Verdict("git reset --hard would discard uncommitted eval cases")
        if sub in {"checkout", "restore", "rm"}:
            hit = self._first_hit(paths, cwd, include_ancestors=True)
            if hit:
                return Verdict(f"git {sub} would discard or delete protected path {hit}")
        return ALLOW

    def _first_hit(self, tokens: list[str], cwd: Path, *, include_ancestors: bool) -> str | None:
        """Return the first token that targets protected data, or None."""
        return next((t for t in tokens if self._hits(t, cwd, include_ancestors)), None)

    def _hits(self, token: str, cwd: Path, include_ancestors: bool) -> bool:
        """True if ``token`` is (inside) a protected dir, or optionally one of its parents.

        Parents count for deletes (``rm -rf data`` removes data/raw) but not for writes
        (``cp x .`` does not overwrite data/raw).
        """
        if "$" in token:
            return False  # unexpanded variable: unknowable here; documented limitation
        path = self._resolve(_glob_base(token), cwd)
        for protected in self._protected:
            if _is_within(path, protected):
                return True
            if include_ancestors and path in protected.parents:
                return True
        return False

    def _resolve(self, token: str, cwd: Path) -> Path:
        """Resolve a path token the way the shell would, relative to ``cwd``."""
        return (cwd / Path(token).expanduser()).resolve()


def _is_within(path: Path, directory: Path) -> bool:
    """True if ``path`` is ``directory`` or anything under it."""
    return path == directory or directory in path.parents


def _glob_base(token: str) -> str:
    """Cut a glob down to the directory it expands in: ``data/raw/*`` -> ``data/raw``."""
    match = re.search(r"[*?\[]", token)
    if not match:
        return token
    head = token[: match.start()]
    return head.rsplit("/", 1)[0] if "/" in head else "."


def _tokenize(command: str) -> list[str]:
    """Split a shell command into words and operators (``;``, ``&&``, ``>``, ...)."""
    lexer = shlex.shlex(command.replace("\n", " ; "), posix=True, punctuation_chars="();<>|&")
    lexer.whitespace_split = True
    return list(lexer)


def _split_segments(tokens: list[str]) -> list[list[str]]:
    """Split tokens into simple commands at ``;``, ``&&``, ``||``, ``|``, ``&`` and parens."""
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in _SEPARATORS:
            segments.append([])
        else:
            segments[-1].append(token)
    return [s for s in segments if s]


def _strip_redirects(segment: list[str]) -> tuple[list[str], list[str]]:
    """Separate output-redirect targets from the command words (``2>`` lexes as ``2``, ``>``)."""
    words: list[str] = []
    targets: list[str] = []
    tokens = iter(segment)
    for token in tokens:
        if token in _REDIRECTS:
            target = next(tokens, None)
            if target is not None and not target.startswith("&"):
                targets.append(target)
            if words and words[-1].isdigit():
                words.pop()
        else:
            words.append(token)
    return words, targets


def _strip_wrappers(words: list[str]) -> list[str]:
    """Drop ``VAR=value`` prefixes and wrappers like ``sudo``/``env`` to reach the real program."""
    index = 0
    while index < len(words) and (
        words[index] in _WRAPPERS or re.fullmatch(r"[A-Za-z_]\w*=.*", words[index])
    ):
        index += 1
    return words[index:]


def _runs_inline_script(program: str, args: list[str]) -> bool:
    """True for interpreters (``python -c``, ``perl -e``, ``bash -c``, ``uv run python``)."""
    if program == "uv":
        return any(_INTERPRETER.fullmatch(a) for a in args[:3])
    return bool(_INTERPRETER.fullmatch(program))


def _find_execs_delete(args: list[str]) -> bool:
    """True for ``find ... -exec rm ...`` (or ``-execdir``)."""
    return any(
        a in {"-exec", "-execdir"} and i + 1 < len(args) and args[i + 1] in _DELETERS
        for i, a in enumerate(args)
    )


def main() -> int:
    """Read the hook payload from stdin and return the exit code Claude Code expects."""
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as error:
        # Exit 1 = non-blocking error: a broken payload must not freeze every tool call.
        print(f"protect_data hook: unreadable input ({error}); allowing", file=sys.stderr)
        return 1
    root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])
    cwd = Path(payload.get("cwd") or root)
    verdict = DataGuard(root).check(
        str(payload.get("tool_name", "")), payload.get("tool_input") or {}, cwd
    )
    if verdict.blocked:
        print(
            f"Blocked by .claude/hooks/protect_data.py: {verdict.reason}. "
            "This data can't be recreated. If it is really intended, ask the owner to run "
            "it themselves.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
