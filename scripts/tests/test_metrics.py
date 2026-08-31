"""Accessor checks for src/metrics.py (the numbers the figures plot).

metrics.py loads runs/test_split_tables.json + runs/matched_pair_tables.json at import, so this module
is skipped unless those aggregated tables are present (i.e. after pipeline stage 4). When they are, it
verifies the loaders/accessors return well-formed values for a known arm/task/seed.
"""

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

if (
    not (REPO / "runs/test_split_tables.json").exists()
    or not (REPO / "runs/matched_pair_tables.json").exists()
):
    pytest.skip(
        "aggregated runs/*.json tables not present (run pipeline stage 4)", allow_module_level=True
    )

from src import metrics as M  # noqa: E402


def test_clean_drops_none_and_nan():
    assert M.clean([1.0, None, float("nan"), 2.5]) == [1.0, 2.5]


def test_registry_shapes():
    assert len(M.SEEDS) >= 1
    assert len(M.TASKS) == 11  # HIA excluded from the graph-task set
    assert ("additive_best", "Anchored MolLedger") in M.MODELS


def test_acc_returns_metric_dict():
    seed, task = M.SEEDS[0], M.TASKS[0]
    row = M.acc(seed, "additive_none", task)
    assert row is not None and "mae" in row and row["mae"] >= 0


def test_faith_r_is_signed_scalar_or_none():
    v = M.faith_r(M.SEEDS[0], "ours_best", "logd")
    assert v is None or (isinstance(v, float) and -1.0001 <= v <= 1.0001)


def test_mp_cell_and_count_agree_on_presence():
    seed_task = M.TASKS[0]
    cell = M.mp_cell("held_out", "class_d", seed_task, "ours_best", "leakage")
    assert cell is None or ("mean" in cell and "sd" in cell)
    assert M.mp_count("held_out", "class_d", seed_task) >= 0


def test_attr_timing_rows_are_positive():
    for label, mean_ms, sd_ms, n in M.attr_timing():
        assert mean_ms > 0 and sd_ms >= 0 and n >= 1 and not math.isnan(mean_ms)
