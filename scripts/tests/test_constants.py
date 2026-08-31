"""Guards for the shared graph-task constants in src/data/multitask.py.

GRAPH_TASK_SPECS / GRAPH_TASK_COLS must still be exactly "every non-binary TASK_REGISTRY entry"
(HIA, the one binary task, dropped), and every downstream importer must see the same objects.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.multitask import (  # noqa: E402
    GRAPH_TASK_COLS,
    GRAPH_TASK_SPECS,
    TASK_REGISTRY,
)


def test_graph_task_specs_are_the_non_binary_tasks():
    expected = [t for t in TASK_REGISTRY if t.label_kind != "binary"]
    assert GRAPH_TASK_SPECS == expected
    assert GRAPH_TASK_COLS == [i for i, t in enumerate(TASK_REGISTRY) if t.label_kind != "binary"]


def test_hia_is_the_only_dropped_task():
    dropped = [t.name for t in TASK_REGISTRY if t not in GRAPH_TASK_SPECS]
    assert dropped == ["hia"]
    assert len(GRAPH_TASK_SPECS) == 11
    assert "hia" not in [t.name for t in GRAPH_TASK_SPECS]


def test_cols_index_back_to_the_kept_tasks():
    # GRAPH_TASK_COLS[k] is the original registry column of GRAPH_TASK_SPECS[k].
    assert [TASK_REGISTRY[i] for i in GRAPH_TASK_COLS] == GRAPH_TASK_SPECS


def test_importers_share_the_same_objects():
    from scripts.train.train_molledger import GRAPH_TASK_SPECS as gts
    from scripts.train.train_molledger import TASKS

    assert gts is GRAPH_TASK_SPECS
    assert TASKS is GRAPH_TASK_SPECS
