"""Functional smoke tests for the training/scoring pipeline.

Drives the real scripts/train/train_molledger.py CLI (one epoch, a tiny slice of the real
local dataset) for every --head, then round-trips the saved checkpoint through
build_model_from_checkpoint and scripts/score/matched_pair_attribution.py's loader. Not a
substitute for a full training run -- it exists to catch import/signature breakage across the
model/loss/scoring code without paying for the real ~300-epoch x 3-seed sweep.

Skipped if the local data cache isn't present (run scripts/pipeline/01_data.sh first).
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

if not (REPO / "data/raw/multitask/conformers.pt").exists():
    pytest.skip(
        "data/raw/multitask/ not present (run scripts/pipeline/01_data.sh)",
        allow_module_level=True,
    )

import torch  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402
from torch_geometric.utils import scatter  # noqa: E402

import scripts.score.matched_pair_attribution as mpa  # noqa: E402
import scripts.train.train_molledger as tba  # noqa: E402
from src.attribution import attribute  # noqa: E402
from src.data.descriptors import attach_descriptors  # noqa: E402
from src.data.multitask import load_multitask_splits  # noqa: E402
from src.models.gnn import build_model_from_checkpoint  # noqa: E402

N_TRAIN, N_EVAL = 64, 32


@pytest.fixture(scope="module")
def raw_tiny_splits():
    """12-column (registry-space) splits, as load_multitask_splits itself returns -- train()'s own
    subset_tasks() call drops HIA down to 11 columns, so this must NOT be pre-subset here."""
    full = load_multitask_splits(seed=42, conformer_cache_path=None)
    return {
        "train": full["train"][:N_TRAIN],
        "valid": full["valid"][:N_EVAL],
        "test": full["test"][:N_EVAL],
    }


@pytest.fixture(scope="module")
def tiny_splits(raw_tiny_splits):
    """11-column (graph-task-space) splits, for tests that call model/loss code directly."""
    return {s: tba.subset_tasks(dl) for s, dl in raw_tiny_splits.items()}


@pytest.fixture(autouse=True)
def _tiny_data(monkeypatch, raw_tiny_splits):
    """Every train() call loads this tiny slice instead of the real ~11k-molecule split."""
    monkeypatch.setattr(tba, "load_multitask_splits", lambda **kw: raw_tiny_splits)
    # Force CPU: GNANModel's sparse-COO shell expansion isn't implemented on the MPS backend, and
    # that's an Apple-Silicon/torch limitation unrelated to what this test is checking.
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


def _run_train(monkeypatch, tmp_path, head, *extra):
    argv = [
        "train_molledger.py",
        "--head",
        head,
        "--epochs",
        "1",
        "--eval_every",
        "1",
        "--batch_size",
        "16",
        "--hidden_dim",
        "16",
        "--num_layers",
        "2",
        "--lr",
        "1e-3",
        "--init_seed",
        "1",
        "--checkpoint_dir",
        str(tmp_path),
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    tba.main()
    (ckpt,) = tmp_path.glob("*/best.pt")
    return ckpt


@pytest.mark.parametrize("head", ["additive", "pooled", "gnan", "ligandformer"])
def test_train_one_epoch_and_reload(monkeypatch, tmp_path, tiny_splits, head):
    ckpt_path = _run_train(monkeypatch, tmp_path, head)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model, backbone = build_model_from_checkpoint(ck)
    model.eval()

    batch = next(iter(DataLoader(tiny_splits["valid"], batch_size=8)))
    pred, scores = tba.forward_model(model, batch, backbone)
    assert pred.shape == (batch.num_graphs, tba.NTASK)
    assert torch.isfinite(pred).all()

    if scores is not None:  # additive / gnan: exact per-atom decomposition
        summed = scatter(scores, batch.batch, dim=0, reduce="sum")
        assert torch.allclose(pred, summed, atol=1e-3)


def test_additive_global_context_train_step(monkeypatch, tmp_path):
    ckpt_path = _run_train(
        monkeypatch,
        tmp_path,
        "additive",
        "--additive_global_context",
        "--additive_context_dim",
        "4",
        "--anchor_rule",
        "--lambda_anchor",
        "0.1",
    )
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ck["additive_global_context"] is True
    assert ck["additive_context_dim"] == 4


def test_matched_pair_attribution_load_and_attribute(monkeypatch, tmp_path, tiny_splits):
    """Round-trips an additive checkpoint through matched_pair_attribution.load_model and scores
    two molecules with --method additive, exercising the same path 03_score.sh drives."""
    ckpt_path = _run_train(monkeypatch, tmp_path, "additive")
    model, kind, _ = mpa.load_model(str(ckpt_path), "cpu")
    assert kind == "additive"
    data = tiny_splits["valid"][:2]
    attrs, preds, aux = attribute(model, data, "additive", backbone="gin", device="cpu")
    assert len(attrs) == 2
    assert preds.shape == (2, tba.NTASK)


def test_descriptors_3d_contact_edges():
    """The contact-mode edge logic folded from the deleted src/data/geom_edges.py into
    src/data/descriptors.py still produces a full-length feature vector."""
    conf_cache = str(REPO / "data/raw/multitask/conformers.pt")
    train = load_multitask_splits(seed=42, conformer_cache_path=conf_cache)["train"][:8]
    splits = {"train": train}
    dim = attach_descriptors(splits)
    assert dim > 0
    assert splits["train"][0].desc.shape == (1, dim)
