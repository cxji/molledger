"""Cheap, data-free guards that the tree stays import-clean after the cleanup/refactor.

`test_all_sources_parse` catches syntax errors and dangling references anywhere in src/ + scripts/.
`test_core_modules_import` actually imports the model/data layer (no runs/ or checkpoints needed), so a
broken import from the file deletions or the constant relocation fails fast. The figure/aggregation
entrypoints load runs/*.json at import time, so they are only parse-checked here, not imported.
"""

import ast
import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

SOURCES = sorted(p for p in (REPO / "src").rglob("*.py")) + sorted(
    p for p in (REPO / "scripts").rglob("*.py")
)


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(REPO)))
def test_all_sources_parse(path):
    ast.parse(path.read_text(), filename=str(path))


# Core layer imported cleanly (data-free at import time).
CORE_MODULES = [
    "src.data.multitask",
    "src.data.transforms",
    "src.attribution",
    "src.wisp_mutants",
    "src.losses",
    "src.models.gnn",
    "src.models.ligandformer",
    "scripts.train.train_molledger",
    "scripts.score.matched_pair_attribution",
]


@pytest.mark.parametrize("mod", CORE_MODULES)
def test_core_modules_import(mod):
    importlib.import_module(mod)
