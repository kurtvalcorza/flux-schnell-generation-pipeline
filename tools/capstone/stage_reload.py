"""Stage 9 (fresh lab process): rebuild inference from the exported files alone, check logit parity, then score new images.

Nothing from the training process is imported or unpickled: the head comes from safetensors, the backbone from its
digest-verified safetensors file, and preprocessing from the definition in sdi_core.
"""
# ruff: noqa: E501
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdi_core as core  # noqa: E402
import stage_features as sf  # noqa: E402

NEW_PER_CLASS = 2


def main(cfg_path: str) -> None:
    cfg = core.read_json(cfg_path)
    work, out = Path(cfg["work_dir"]), Path(cfg["out_dir"])
    weight, bias, manifest = core.load_artifact(out / "classifier")
    if manifest["backbone"]["sha256"] != sf.BACKBONE["sha256"]:
        raise core.ContractError("the artifact names a different backbone digest than the one this notebook pins")
    model = sf.build_backbone(sf.stage_backbone(work), "cpu")
    data_root = work / "data" / ("sdi" if cfg["data_source"] == "bosch" else "byod")
    reference = core.read_json(out / "reload_reference.json")
    feats = sf.embed(model, [core.load_gray(data_root / e["relpath"])[0] for e in reference["examples"]], "cpu")
    logits = core.head_logits(feats, weight, bias)
    expected = np.array([e["logits"] for e in reference["examples"]], dtype=np.float32)
    diff = float(np.abs(logits - expected).max())
    parity = {"examples": len(expected), "max_abs_logit_difference": diff, "tolerance": core.PARITY_TOLERANCE, "passed": diff <= core.PARITY_TOLERANCE, "same_predictions": bool((logits.argmax(1) == expected.argmax(1)).all())}
    core.write_json(out / "reload_parity.json", parity)
    print(f"reload parity: max |logit difference| = {diff:.2e} (tolerance {core.PARITY_TOLERANCE:.0e}) -> {'PASS' if parity['passed'] else 'FAIL'}")
    if not parity["passed"]:
        raise core.ContractError("the reloaded classifier does not reproduce the exported logits; the artifact is not trustworthy")

    # New-image inference: training-partition images never used for fitting, captioning or generation, plus any
    # images in NEW_IMAGE_DIR. Unknown labels stay empty.
    order = manifest["classes"]
    rows = core.read_split_manifest(out / "split_manifest.csv")
    unused = [r for r in rows if r["split"] == "train" and not r["selected"]]
    chosen = []
    for label in order:
        pool = sorted((r for r in unused if r["label"] == label), key=lambda r: core.sha256_bytes(f"new|{r['file_sha256']}".encode()))
        chosen += [(r["image_id"], data_root / r["relpath"], label, "unused training-partition image") for r in pool[:NEW_PER_CLASS]]
    new_dir = cfg.get("new_image_dir") or ""
    rejected = []
    if new_dir:
        for p in sorted(Path(new_dir).iterdir()):
            if p.suffix.lower() in core.IMAGE_SUFFIXES and p.is_file():
                chosen.append((p.stem, p, "", "NEW_IMAGE_DIR"))
    results = []
    for image_id, path, label, source in chosen:
        info = core.inspect_image(path)
        if not info["decode_ok"]:
            rejected.append({"image": str(path), "reason": info["error"]})
            continue
        image, mode = core.load_gray(path)
        feat = sf.embed(model, [image], "cpu")
        score = core.softmax(core.head_logits(feat, weight, bias))[0]
        results.append({"image_id": image_id, "source": source, "known_label": label, "input_mode": mode, "input_size": f"{info['width']}x{info['height']}", "predicted_label": order[int(score.argmax())], **{f"score_{c}": round(float(score[i]), 6) for i, c in enumerate(order)}})
    with open(out / "new_image_predictions.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    for r in results:
        print(f"{r['image_id']:<22} known={r['known_label'] or '?':<10} predicted={r['predicted_label']:<10} " + " ".join(f"{c}={r['score_' + c]:.3f}" for c in order))
    if rejected:
        print(f"rejected {len(rejected)} new image(s): {rejected}")
    core.write_json(out / "new_image_rejections.json", rejected)


if __name__ == "__main__":
    core.run_main(main, sys.argv[1])
