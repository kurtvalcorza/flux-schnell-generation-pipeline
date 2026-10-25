"""Prose numbers stay true across every recorded hosted run (Kurt's ruling, 2026-10-25).

Runs: Kaggle T4 of blob 52226812 (2026-09-20); Colab T4 of blobs d26eb848 (2026-10-04) and 6ea19d2e (2026-10-25).
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "tutorials" / "flux_schnell_generation_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_single_run_directions_and_values_are_gone():
    md = _markdown()
    for stale in ("three moved up (Song Sparrow, White-throated Sparrow, House Finch)",
                  "went from 71.95 to 71.89, essentially", "in that run three of the six species' reference similarities fell",
                  "CLIP readings of the generations did not move", "Peak training memory was 12.23 GB.",
                  "took 3,129.5 s of suite wall time"):
        assert stale not in md, stale


def test_flipped_directions_and_ranges_are_stated():
    md = _markdown()
    assert "moved in opposite directions in the two runs" in md
    assert "71.95 → 71.89 in the first and 71.95 → 72.55 in the second" in md
    assert "0.834 → 0.591–0.595 → 0.532–0.534 → 0.523–0.524" in md
    assert "3,129.5 s and 2,717.6 s" in md and "12.22 to 12.23 GB" in md
