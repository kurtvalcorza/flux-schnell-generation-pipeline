"""Stage 9 (fresh lab process): rebuild inference from the exported files alone, check logit parity, then score new images.

Nothing from the training process is imported or unpickled: the head comes from safetensors, the backbone from its
digest-verified safetensors file, and preprocessing from the definition in sdi_core.
"""
# ruff: noqa: E501
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdi_core as core  # noqa: E402
import stage_features as sf  # noqa: E402

NEW_PER_CLASS = 2


def main(cfg_path: str) -> None:
    cfg = core.read_json(cfg_path)
    base, work, out = Path(cfg["base_dir"]), Path(cfg["work_dir"]), Path(cfg["out_dir"])
    weight, bias, manifest = core.load_artifact(out / "classifier")
    # Meaning before numbers: the artifact must name the frozen experiment, its class order and indices, decision
    # rule, preprocessing, backbone and selected head. A relabelled class list would leave every logit unchanged.
    record = core.verify_frozen(out / "experiment_config.json", base)
    core.check_artifact_semantics(manifest, record=record, preprocessing=core.PREPROCESSING, backbone=sf.BACKBONE)
    reference_path = out / "reload_reference.json"
    if core.sha256_file(reference_path) != manifest.get("reload_reference_sha256"):
        raise core.ContractError("reload_reference.json differs from the reference recorded in the artifact manifest")
    model = sf.build_backbone(sf.stage_backbone(work), "cpu")
    data_root = work / "data" / ("sdi" if cfg["data_source"] == "bosch" else "byod")
    rows = core.read_split_manifest(out / "split_manifest.csv")
    by_id = {r["image_id"]: r for r in rows}
    reference = core.read_json(reference_path)
    feats = sf.embed(model, [core.load_verified_gray(data_root / e["relpath"], by_id[e["image_id"]]) for e in reference["examples"]], "cpu")
    order = manifest["classes"]
    expected = np.array([e["logits"] for e in reference["examples"]], dtype=np.float32)
    parity = core.replay_check(core.head_logits(feats, weight, bias), expected, [e["expected_prediction"] for e in reference["examples"]], order)
    parity["semantic_checks"] = "classes, class_index, decision rule, preprocessing, backbone, experiment record and selected head match the frozen experiment"
    print(f"semantic checks: PASS ({parity['semantic_checks']})")
    print(f"reload parity: max |logit difference| = {parity['max_abs_logit_difference']:.2e} (tolerance {core.PARITY_TOLERANCE:.0e}); replayed decisions identical: {parity['same_predictions']} -> {'PASS' if parity['passed'] else 'FAIL'}")
    if not parity["passed"]:
        core.write_json(out / "reload_parity.json", parity)
        raise core.ContractError("the reloaded classifier does not reproduce the exported logits and decisions; the artifact is not trustworthy")

    # New-image inference: training-partition images never used for fitting, captioning or generation, plus any
    # images in NEW_IMAGE_DIR. Unknown labels stay empty. An empty cohort is recorded as not run, never as run.
    unused = [r for r in rows if r["split"] == "train" and not r["selected"]]
    chosen = []
    for label in order:
        pool = sorted((r for r in unused if r["label"] == label), key=lambda r: core.sha256_bytes(f"new|{r['file_sha256']}".encode()))
        chosen += [(r["image_id"], data_root / r["relpath"], label, "unused training-partition image", r) for r in pool[:NEW_PER_CLASS]]
    new_dir = cfg.get("new_image_dir") or ""
    rejected = []
    if new_dir:
        for p in sorted(Path(new_dir).iterdir()):
            if p.suffix.lower() in core.IMAGE_SUFFIXES and p.is_file():
                chosen.append((p.stem, p, "", "NEW_IMAGE_DIR", None))
    results = []
    for image_id, path, label, source, row in chosen:
        info = core.inspect_image(path)
        if not info["decode_ok"]:
            rejected.append({"image": str(path), "reason": info["error"]})
            continue
        image, mode = (core.load_verified_gray(path, row), info["mode"]) if row is not None else core.load_gray(path)
        score = core.softmax(core.head_logits(sf.embed(model, [image], "cpu"), weight, bias))[0]
        results.append({"image_id": image_id, "source": source, "known_label": label, "input_mode": mode, "input_size": f"{info['width']}x{info['height']}", "predicted_label": order[int(score.argmax())], **{f"score_{c}": round(float(score[i]), 6) for i, c in enumerate(order)}})
    core.write_new_predictions(out / "new_image_predictions.csv", results, order)
    status = "ran" if results else ("not_run_all_inputs_rejected" if chosen else "not_run_no_inputs")
    parity["new_image_inference"] = {"status": status, "scored": len(results), "candidates": len(chosen), "rejected": rejected}
    core.write_json(out / "reload_parity.json", parity)
    for r in results:
        print(f"{r['image_id']:<22} known={r['known_label'] or '?':<10} predicted={r['predicted_label']:<10} " + " ".join(f"{c}={r['score_' + c]:.3f}" for c in order))
    if rejected:
        print(f"rejected {len(rejected)} new image(s): {rejected}")
    if not results:
        print(f"new-image inference NOT RUN ({status}): no unused training-partition images and no usable NEW_IMAGE_DIR images. "
              "The export and reload checks above still stand; set NEW_IMAGE_DIR to score your own images.")


if __name__ == "__main__":
    core.run_main(main, sys.argv[1])
