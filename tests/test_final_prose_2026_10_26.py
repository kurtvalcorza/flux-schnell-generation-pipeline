"""Final prose pass (Kurt's ruling, 2026-10-26): the prose-rule items flagged on the confirming run are fixed.

Run-status sentences, timing promises and unlabelled expectations are removed or recast
as records of named runs.
Fails on PR #12 head 785d06e.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "flux_schnell_generation_colab.ipynb"


def _markdown(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_final_prose_pass():
    md = _markdown(NB)
    for gone in (
        'has not yet been run on Colab',
        'MSE around 0.5..1.5',
        'the load taking a few minutes',
    ):
        assert gone not in md, gone
    for kept in (
        'is meant to be enough.',
        'Look for: the flow-matching MSE, label accuracy',
    ):
        assert kept in md, kept
