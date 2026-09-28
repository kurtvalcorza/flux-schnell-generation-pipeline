"""Stage 6 (lab environment): fit the linear heads for arms A/B/C on matched slot schedules, select epochs on
validation, and freeze the experiment. The test partition is not read here.

Modes: canonical (default); `--preview` (majority baseline and arm A on validation, before any generation);
`--synthetic-probability P --tag NAME` (change-one-thing exercise, validation only);
`--review CSV --tag NAME` (optional human-reviewed pool, validation only). Only the canonical mode freezes.
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdi_core as core  # noqa: E402


def load_features(work: Path, *, with_synthetic: bool = True) -> dict:
    f = work / "features"
    feats = {name: dict(np.load(f / f"{name}.npz")) for name in ("train", "val")}
    empty = {"candidate_id": np.array([], dtype=str), "label": np.array([], dtype=np.int64), "det": np.zeros((0, core.FEATURE_DIM), np.float32), "views": np.zeros((0, core.N_VIEWS, core.FEATURE_DIM), np.float32)}
    feats["synthetic"] = dict(np.load(f / "synthetic.npz")) if with_synthetic else empty
    return feats


def pools(labels: np.ndarray, k: int) -> tuple[list[int], list[int]]:
    sizes = [int((labels == c).sum()) for c in range(k)]
    offsets = [int(sum(sizes[:c])) for c in range(k)]
    if any(np.any(labels[offsets[c] : offsets[c] + sizes[c]] != c) for c in range(k)):
        raise core.ContractError("feature rows are not grouped by class")
    return sizes, offsets


def slot_features(feats: dict, sched: dict, src: dict, real_off: list[int], synth_off: list[int]) -> np.ndarray:
    cls, index, view, synthetic = sched["cls"], src["index"], src["view"], src["synthetic"]
    x = np.empty((len(cls), core.FEATURE_DIM), dtype=np.float32)
    real = ~synthetic
    rows = np.asarray(real_off)[cls] + index
    det = view < 0
    x[real & det] = feats["train"]["det"][rows[real & det]]
    x[real & ~det] = feats["train"]["views"][rows[real & ~det], view[real & ~det]]
    if synthetic.any():
        srows = np.asarray(synth_off)[cls[synthetic]] + index[synthetic]
        x[synthetic] = feats["synthetic"]["views"][srows, view[synthetic]]
    return x


def train_head(x: np.ndarray, y: np.ndarray, init: dict, val_x: np.ndarray, val_y: np.ndarray, k: int) -> dict:
    import torch

    torch.use_deterministic_algorithms(True)
    head = torch.nn.Linear(core.FEATURE_DIM, k)
    head.load_state_dict({n: t.clone() for n, t in init.items()})
    opt = torch.optim.AdamW(head.parameters(), lr=core.HEAD["learning_rate"], weight_decay=core.HEAD["weight_decay"])
    xt, yt, vx = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(val_x)
    batch, updates = core.HEAD["batch_size"], core.HEAD["updates_per_epoch"]
    curve, best = [], None
    for epoch in range(1, core.HEAD["epochs"] + 1):
        losses = []
        for u in range(updates):
            start = ((epoch - 1) * updates + u) * batch
            loss = torch.nn.functional.cross_entropy(head(xt[start : start + batch]), yt[start : start + batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        with torch.no_grad():
            pred = head(vx).argmax(1).numpy()
        m = core.metrics_from_confusion(core.confusion_matrix(val_y, pred, k), [str(i) for i in range(k)])
        curve.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 5), "val_macro_f1": m["macro_f1"], "val_balanced_accuracy": m["balanced_accuracy"]})
        if best is None or m["macro_f1"] > best["val_macro_f1"]:  # strict: the earlier epoch wins a tie
            best = {"epoch": epoch, "val_macro_f1": m["macro_f1"], "weight": head.weight.detach().numpy().copy(), "bias": head.bias.detach().numpy().copy()}
    return {**best, "curve": curve}


def read_review(path: Path, candidates: list[str]) -> set[str]:
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    known, accepted = set(candidates), set()
    for n, row in enumerate(rows, start=2):
        cid, decision = row.get("candidate_id", "").strip(), row.get("decision", "").strip().lower()
        if cid not in known:
            continue  # the template lists every attempt; rejected attempts have no features
        if decision not in ("accept", "reject", "uncertain"):
            raise core.ContractError(f"{path} line {n}: decision {decision!r} must be accept, reject or uncertain")
        if decision == "accept":
            accepted.add(cid)
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--synthetic-probability", type=float, default=None)
    parser.add_argument("--review", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    cfg = core.read_json(args.config)
    base, work, out = Path(cfg["base_dir"]), Path(cfg["work_dir"]), Path(cfg["out_dir"])
    canonical = args.synthetic_probability is None and args.review is None and not args.preview
    if args.preview:
        args.tag = "real_only_preview"
    started = time.time()
    import torch

    data = core.read_json(out / "data_manifest.json")
    order = data["class_order"]
    k = len(order)
    seeds = [int(s) for s in cfg["seeds"]]
    feats = load_features(work, with_synthetic=not args.preview)
    real_sizes, real_off = pools(feats["train"]["label"], k)
    gen_status = core.read_json(out / "generation_status.json") if not args.preview else {"complete": False, "shortfall": ["not generated yet (preview)"]}
    synth = feats["synthetic"]
    if args.review:
        accepted = read_review(Path(args.review), list(synth["candidate_id"]))
        keep = np.array([c in accepted for c in synth["candidate_id"]], dtype=bool)
        synth = {key: value[keep] for key, value in synth.items()}
        feats["synthetic"] = synth
        print(f"human-review extension: {int(keep.sum())} accepted candidates of {len(keep)}")
    synth_sizes = [int((synth["label"] == c).sum()) for c in range(k)]
    synth_off = [int(sum(synth_sizes[:c])) for c in range(k)]
    if len(synth["label"]) and np.any(np.diff(synth["label"]) < 0):
        raise core.ContractError("synthetic feature rows are not grouped by class")
    probability = core.SYNTHETIC_PROBABILITY if args.synthetic_probability is None else args.synthetic_probability
    run_c = gen_status["complete"] or not canonical
    arms = list(core.ARMS if canonical else ("A",) if args.preview else ("C",))
    if canonical and not run_c:
        arms.remove("C")
        print("generation shortfall: " + "; ".join(gen_status["shortfall"]) + " -> arm C is not run and no synthetic comparison is reported")
    val_x, val_y = feats["val"]["det"], feats["val"]["label"]
    results: dict = {"majority": {}}
    majority = int(np.argmax(real_sizes))
    majority_metrics = core.metrics_from_confusion(core.confusion_matrix(val_y, np.full_like(val_y, majority), k), order)
    heads_dir = work / "heads" / (args.tag or "canonical")
    heads_dir.mkdir(parents=True, exist_ok=True)
    from safetensors.numpy import save_file

    for seed in seeds:
        sched = core.slot_schedule(seed, real_sizes, synth_sizes)
        torch.manual_seed(seed)
        init = {n: t.detach().clone() for n, t in torch.nn.Linear(core.FEATURE_DIM, k).state_dict().items()}
        results["majority"][str(seed)] = {"class": order[majority], "val": majority_metrics}
        for arm in arms:
            src = core.arm_sources(sched, arm, synth_probability=probability, synth_sizes=synth_sizes)
            x = slot_features(feats, sched, src, real_off, synth_off)
            fit = train_head(x, sched["cls"], init, val_x, val_y, k)
            pred = (val_x @ fit["weight"].T + fit["bias"]).argmax(1)
            val_metrics = core.metrics_from_confusion(core.confusion_matrix(val_y, pred, k), order)
            path = heads_dir / f"{arm}_s{seed}.safetensors"
            save_file({"weight": fit["weight"].astype(np.float32), "bias": fit["bias"].astype(np.float32)}, str(path))
            results.setdefault(arm, {})[str(seed)] = {
                "selected_epoch": fit["epoch"], "val": val_metrics, "curve": fit["curve"], "head": str(path.relative_to(base)),
                "synthetic_slot_fraction": float(src["synthetic"].mean()), "synthetic_share_of_defect_slots": float(src["synthetic"][sched["cls"] > 0].mean()),
            }
            print(f"seed {seed} arm {arm}: epoch {fit['epoch']:>2} val macro-F1 {val_metrics['macro_f1']:.4f} (synthetic share of defect slots {results[arm][str(seed)]['synthetic_share_of_defect_slots']:.3f})")
    if not canonical:
        body = {"tag": args.tag, "arms": arms, "synthetic_probability": None if args.preview else probability, "review_csv": args.review, "synthetic_sizes": synth_sizes, "evidence": "validation only; exploratory; does not alter the canonical frozen results", "results": results}
        core.write_json(out / "extensions" / f"{args.tag}.json", body)
        print(core.canonical_json({"tag": args.tag, "val_macro_f1": {a: {s: v["val"]["macro_f1"] for s, v in results[a].items()} for a in arms}, "majority_val_macro_f1": majority_metrics["macro_f1"]}))
        return
    # Export choice: canonical seed, arm with the best validation macro-F1, ties to A, then B, then C.
    canon = str(core.CANONICAL_SEED if core.CANONICAL_SEED in seeds else seeds[0])
    ranked = sorted(arms, key=lambda a: (-results[a][canon]["val"]["macro_f1"], core.ARMS.index(a)))
    selection = {"seed": int(canon), "arm": ranked[0], "val_macro_f1": results[ranked[0]][canon]["val"]["macro_f1"], "epoch": results[ranked[0]][canon]["selected_epoch"], "rule": "canonical seed; highest validation macro-F1; ties favour A, then B, then C"}
    record = {
        "question": "Under a fixed real-data budget, classifier, class-sampling policy and training budget, does substituting synthetic defect examples into training improve real-test macro-F1 compared with conventional augmentation?",
        "primary_contrast": "C - B", "contextual_contrast": "B - A", "primary_metric": "macro-F1 on the test partition",
        "class_order": order, "budget": data["budget"], "split_version": data["split"]["version"], "split_sha256": data["split"]["manifest_sha256"],
        "seeds": seeds, "smoke_run": len(seeds) < 3, "head": core.HEAD, "synthetic_probability": probability,
        "slot_policy": "class-balanced batches (11/11/10 rotating); normal slots always real; in arm C each defect slot draws its synthetic pool with the stated probability; A uses deterministic features, B and C draw one of four fixed views",
        "arms_run": arms, "generation_status": gen_status, "real_pool_sizes": real_sizes, "synthetic_pool_sizes": synth_sizes,
        "preprocessing": core.PREPROCESSING, "augment": core.AUGMENT, "features": core.read_json(work / "state" / "features_stage.json"),
        "validation": results, "selection": selection, "majority_class": order[majority], "core_version": core.CORE_VERSION,
    }
    referenced = {name: out / name for name in ("data_manifest.json", "split_manifest.csv", "caption_records.jsonl", "prompts.json", "generation_manifest.jsonl", "generation_status.json", "synthetic_training_overlap.json", "augmentation_views.jsonl", "synthetic_augmentation_views.jsonl")}
    referenced.update({f"features/{n}": work / "features" / f"{n}.npz" for n in ("train", "synthetic", "val")})
    referenced.update({f"heads/{p.stem}": p for p in sorted(heads_dir.glob("*.safetensors"))})
    frozen = core.freeze_record(out / "experiment_config.json", record, referenced, base)
    print(f"experiment frozen: record SHA-256 {frozen['record_sha256'][:16]}..., {len(frozen['frozen_files'])} inputs pinned; export choice {selection} ({time.time() - started:.1f} s)")


if __name__ == "__main__":
    core.run_main(main)
