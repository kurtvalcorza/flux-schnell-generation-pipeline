"""Stage 8 (lab environment): export the validation-selected canonical-seed classifier and its reference logits."""
# ruff: noqa: E501
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdi_core as core  # noqa: E402
import stage_features as sf  # noqa: E402

REFERENCE_PER_CLASS = 3


def reference_rows(rows: list[dict], order: list[str]) -> list[dict]:
    """Fixed real examples for the reload check: the first three test images of each class by image id."""
    test = sorted((r for r in rows if r["split"] == "test"), key=lambda r: r["image_id"])
    return [r for label in order for r in [x for x in test if x["label"] == label][:REFERENCE_PER_CLASS]]


def main(cfg_path: str) -> None:
    cfg = core.read_json(cfg_path)
    base, work, out = Path(cfg["base_dir"]), Path(cfg["work_dir"]), Path(cfg["out_dir"])
    record = core.verify_frozen(out / "experiment_config.json", base)
    from safetensors.numpy import load_file

    selection, order = record["selection"], record["class_order"]
    head = load_file(str(base / record["validation"][selection["arm"]][str(selection["seed"])]["head"]))
    manifest = {
        "classes": order,
        "class_index": {c: i for i, c in enumerate(order)},
        "decision_rule": "argmax over logits; softmax scores are uncalibrated",
        "backbone": sf.BACKBONE,
        "preprocessing": core.PREPROCESSING,
        "selection": {**selection, "arm_name": core.ARM_NAMES[selection["arm"]]},
        "experiment_record_sha256": record["record_sha256"],
        "training_data": "no images or features are embedded; the head was fitted on features of the selected training budget (and, for arm C, generated images)",
        "intended_use": "tutorial reconstruction and inference on product-A-like surface images; not a production inspection system",
    }
    written = core.write_artifact(out / "classifier", head["weight"], head["bias"], manifest)
    rows = core.read_split_manifest(out / "split_manifest.csv")
    ref = reference_rows(rows, order)
    data_root = work / "data" / ("sdi" if cfg["data_source"] == "bosch" else "byod")
    model = sf.build_backbone(sf.stage_backbone(work), "cpu")  # CPU on both sides of the reload check
    feats = sf.embed(model, [core.load_gray(data_root / r["relpath"])[0] for r in ref], "cpu")
    logits = core.head_logits(feats, head["weight"], head["bias"])
    core.write_json(out / "reload_reference.json", {"device": "cpu", "tolerance": core.PARITY_TOLERANCE, "examples": [{"image_id": r["image_id"], "relpath": r["relpath"], "label": r["label"], "logits": [float(v) for v in logits[n]]} for n, r in enumerate(ref)]})
    print(core.canonical_json({"exported": written["selection"], "head_sha256": written["files"][core.ARTIFACT_WEIGHTS]["sha256"], "reference_examples": len(ref)}))


if __name__ == "__main__":
    core.run_main(main, sys.argv[1])
