"""Smoke test: the package installs and every planned module imports."""

import importlib

import pytest

import zeimu

MODULES = ["ingest", "index", "answer", "verify", "tools", "agent", "evaluation", "api"]


def test_version_is_set() -> None:
    assert zeimu.__version__ == "0.1.0"


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(f"zeimu.{name}")
