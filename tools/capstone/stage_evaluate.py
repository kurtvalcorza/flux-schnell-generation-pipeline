"""Stage 7 (lab environment): verify the frozen experiment, then score every arm and seed once on the test partition."""
# ruff: noqa: E501
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdi_core as core  # noqa: E402
import stage_features as sf  # noqa: E402


def main(cfg_path: str) -> None:
    cfg = core.read_json(cfg_path)
    base, work, out = Path(cfg["base_dir"]), Path(cfg["work_dir"]), Path(cfg["out_dir"])
    started = time.time()
    record = core.verify_frozen(out / "experiment_config.json", base)
    print(f"frozen record verified ({len(record['frozen_files'])} inputs unchanged); scoring the test partition once")
    import torch
    from PIL import Image
    from safetensors.numpy import load_file

    order, seeds, arms = record["class_order"], record["seeds"], record["arms_run"]
    k = len(order)
    rows = core.read_split_manifest(out / "split_manifest.csv")
    data_root = work / "data" / ("sdi" if cfg["data_source"] == "bosch" else "byod")
    test = sorted((r for r in rows if r["split"] == "test"), key=lambda r: r["image_id"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = sf.build_backbone(sf.stage_backbone(work), device)
    x = sf.embed(model, [core.load_gray(data_root / r["relpath"])[0] for r in test], device)
    y = np.array([order.index(r["label"]) for r in test])
    groups = [r["group_id"] for r in test]

    predictions, scores, per_run = {}, {}, {}
    majority = order.index(record["majority_class"])
    for seed in seeds:
        predictions[("majority", seed)] = np.full_like(y, majority)
        scores[("majority", seed)] = np.eye(k, dtype=np.float32)[np.full_like(y, majority)]
        for arm in arms:
            head = load_file(str(base / record["validation"][arm][str(seed)]["head"]))
            logits = core.head_logits(x, head["weight"], head["bias"])
            predictions[(arm, seed)] = logits.argmax(1)
            scores[(arm, seed)] = core.softmax(logits)
    for (arm, seed), pred in predictions.items():
        per_run.setdefault(arm, {})[str(seed)] = core.metrics_from_confusion(core.confusion_matrix(y, pred, k), order)
    summary = {}
    for arm, runs in per_run.items():
        summary[arm] = {m: {"mean": float(np.mean([r[m] for r in runs.values()])), "min": float(np.min([r[m] for r in runs.values()])), "max": float(np.max([r[m] for r in runs.values()])), "per_seed": {s: r[m] for s, r in runs.items()}} for m in ("macro_f1", "balanced_accuracy", "accuracy")}
    contrasts = {}
    if "C" in arms:
        contrasts["C - B"] = core.paired_contrast(predictions, y, groups, arm="C", reference="B", seeds=seeds, k=k)
    contrasts["B - A"] = core.paired_contrast(predictions, y, groups, arm="B", reference="A", seeds=seeds, k=k)

    # Held-out overlap audit, run only now that everything is frozen. A consequential match is reported, not removed.
    sig = np.load(work / "audit" / "signatures.npz")
    split_of = {r["image_id"]: r["split"] for r in rows}
    held = np.array([split_of.get(i) in ("val", "test") for i in sig["image_id"]])
    generated = [r for r in core.read_jsonl(out / "generation_manifest.jsonl") if r["status"] == "eligible"]
    synth_sigs = np.stack([core.signature(Image.open(out / r["file"]).convert("L")) for r in generated]) if generated else np.zeros((0, core.SIGNATURE_SIDE**2), np.float32)
    best, where = core.max_similarity(synth_sigs, sig["signature"][held].astype(np.float32))
    held_ids = sig["image_id"][held]
    flagged = [{"candidate_id": generated[n]["candidate_id"], "closest_heldout_image": str(held_ids[where[n]]), "correlation": round(float(best[n]), 4)} for n in np.nonzero(best >= core.NEAR_DUPLICATE_THRESHOLD)[0]]
    pixel_split: dict = {}
    for r in rows:
        if r["split"] in ("train", "val", "test"):
            pixel_split.setdefault(r["pixel_sha256"], set()).add(r["split"])
    cross_pixels = sum(1 for s in pixel_split.values() if len(s) > 1)
    overlap = {"synthetic_vs_heldout": {"max_correlation": float(best.max()) if len(best) else None, "flagged": flagged}, "exact_pixel_duplicates_across_partitions": cross_pixels}
    valid = not flagged and cross_pixels == 0
    metrics = {
        "evidence": "tutorial evidence from one grouped split of product A and one generated pool; not factory-level or production performance",
        "test_counts": {c: int((y == i).sum()) for i, c in enumerate(order)},
        "arms": {a: core.ARM_NAMES[a] for a in per_run},
        "per_run": per_run, "summary": summary, "contrasts": contrasts,
        "interval_note": "approximate 95% percentile interval from 1,000 class-stratified group bootstrap resamples of the test partition, paired across arms; conditional on this split and generated pool; it does not include training or generation variability",
        "decision_rule": "argmax of the linear head's logits (no threshold tuning); scores are softmax of uncalibrated logits, not calibrated probabilities",
        "overlap_audit": overlap, "comparison_valid": valid,
        "validity_note": "" if valid else "a held-out overlap was found after freezing; the comparison is reported as invalid rather than repaired",
        "generation_complete": record["generation_status"]["complete"], "smoke_run": record["smoke_run"],
        "frozen_record_sha256": record["record_sha256"], "seconds": round(time.time() - started, 1),
    }
    core.write_json(out / "metrics.json", metrics)
    with open(out / "predictions.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["arm", "seed", "image_id", "group_id", "true_label", "predicted_label", *[f"score_{c}" for c in order]])
        for (arm, seed), pred in sorted(predictions.items(), key=lambda kv: (["majority", *core.ARMS].index(kv[0][0]), kv[0][1])):
            for n, r in enumerate(test):
                writer.writerow([arm, seed, r["image_id"], r["group_id"], r["label"], order[pred[n]], *[f"{v:.6f}" for v in scores[(arm, seed)][n]]])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = out / "figures"
    shown = ["majority", *arms]
    fig, axes = plt.subplots(len(shown), len(seeds), figsize=(3.1 * len(seeds), 2.9 * len(shown)), squeeze=False)
    for i, arm in enumerate(shown):
        for j, seed in enumerate(seeds):
            cm = np.array(per_run[arm][str(seed)]["confusion"])
            ax = axes[i][j]
            ax.imshow(np.log1p(cm), cmap="Blues")
            for (a, b), v in np.ndenumerate(cm):
                ax.text(b, a, str(v), ha="center", va="center", fontsize=9)
            ax.set_xticks(range(k), order, fontsize=7)
            ax.set_yticks(range(k), order, fontsize=7)
            ax.set_title(f"{core.ARM_NAMES[arm]}\nseed {seed}, macro-F1 {per_run[arm][str(seed)]['macro_f1']:.3f}", fontsize=8)
            ax.set_xlabel("predicted", fontsize=7)
            ax.set_ylabel("true", fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "confusion_matrices.png", dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for i, arm in enumerate(shown):
        values = [per_run[arm][str(s)]["macro_f1"] for s in seeds]
        ax.scatter([i] * len(values), values, color="0.2", zorder=3)
        ax.hlines(np.mean(values), i - 0.25, i + 0.25, color="tab:blue")
    ax.set_xticks(range(len(shown)), [core.ARM_NAMES[a] for a in shown], fontsize=7)
    ax.set_ylabel("test macro-F1")
    ax.set_title("Each dot is one seed; the bar is the mean", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "macro_f1_by_arm.png", dpi=110)
    plt.close(fig)
    # Error gallery for the canonical seed (test images may be viewed now that the experiment is frozen).
    canon = record["selection"]["seed"]
    items = []
    for arm in [a for a in ("B", "C") if a in arms]:
        wrong = [n for n in range(len(test)) if predictions[(arm, canon)][n] != y[n] and y[n] != majority][:6] + [n for n in range(len(test)) if predictions[(arm, canon)][n] != y[n] and y[n] == majority][:2]
        items += [(core.load_gray(data_root / test[n]["relpath"])[0], f"{arm}: true {order[y[n]]}\npred {order[predictions[(arm, canon)][n]]}") for n in wrong]
    if items:
        core.contact_sheet(items, title=f"Misclassified TEST images, seed {canon} (arm: true -> predicted)").save(figures / "error_gallery.png")
    print(core.canonical_json({"macro_f1_mean": {a: round(s["macro_f1"]["mean"], 4) for a, s in summary.items()}, "contrasts": {c: [round(v["mean_difference"], 4), [round(z, 4) for z in v["interval_95"]]] for c, v in contrasts.items()}, "comparison_valid": valid}))


if __name__ == "__main__":
    core.run_main(main, sys.argv[1])
