"""Run-derived prose is a record of named runs, never an expectation (Kurt's ruling, 2026-10-26).

Runs: Kaggle T4 of blob 52226812 (2026-09-20); Colab T4 of blobs d26eb848 (2026-10-04),
6ea19d2e and 54c8d314 (2026-10-25).
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "tutorials" / "flux_schnell_generation_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "
".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_expectations_and_run_status_sentences_are_gone():
    md = _markdown()
    for stale in ("the suite takes about 45 to 52 minutes", "hosted runs of this revision are pending",
                  "took about 16 to 17 minutes", "trained for about 16 to 17 minutes",
                  "No run of this activity is recorded yet", "the printed memory should fall back",
                  "CLIP readings barely moved", "usually takes a few minutes"):
        assert stale not in md, stale


def test_confirming_run_of_54c8d314_is_recorded():
    md = _markdown()
    assert "blobs `6ea19d2e` and `54c8d314`, 2026-10-25" in md
    assert "label accuracy 0.667 → 0.833, mean reference similarity 71.95 → 72.99" in md
    assert "Song Sparrow (80.18 → 81.60, 78.67 and 83.53)" in md
    assert "two and four in the run of `54c8d314`" in md
    assert "took between 45.3 and 52.2 minutes" in md and "1,088.6 s" in md
