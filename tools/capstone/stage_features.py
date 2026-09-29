"""Stage 3 / 7 (lab environment): frozen ResNet-18 features.

`--part real` (before any generation): the selected training budget (deterministic + four fixed views) and validation.
`--part synthetic` (after generation): the eligible synthetic pool and the synthetic-to-training near-duplicate audit.
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdi_core as core  # noqa: E402

BACKBONE = {
    "id": "timm/resnet18.tv_in1k",
    "revision": "bbd144b3e5565108aad885f145491d11bc6ce807",
    "file": "model.safetensors",
    "bytes": 46_807_446,
    "sha256": "694f673df6520a3158624e8a89af086f59923ee4cd7436fe5bc3bc71d295ad81",
    "license": "bsd-3-clause",
    "architecture": "torchvision.models.resnet18 (the torchvision ImageNet-1k weights, repackaged as safetensors)",
    "feature": "512-d global-average-pooled output of layer4; the ImageNet classifier (fc) is replaced by identity",
}
AUG_SEED = 20260929
BATCH = 64


def stage_backbone(work: Path) -> Path:
    from huggingface_hub import hf_hub_download

    root = work / "weights" / "resnet18-tv-in1k"
    path = root / BACKBONE["file"]
    if not path.is_file() or path.stat().st_size != BACKBONE["bytes"]:
        hf_hub_download(BACKBONE["id"], BACKBONE["file"], revision=BACKBONE["revision"], local_dir=str(root))
    if core.sha256_file(path) != BACKBONE["sha256"]:
        raise core.ContractError(f"{path} fails its pinned digest; refusing to load it")
    return path


def build_backbone(weights: Path, device: str):
    import torch
    import torchvision
    from safetensors.torch import load_file

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = torchvision.models.resnet18(weights=None)
    model.load_state_dict(load_file(str(weights)), strict=True)
    model.fc = torch.nn.Identity()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval().to(device)


def embed(model, images, device: str) -> np.ndarray:
    import torch

    out = []
    for start in range(0, len(images), BATCH):
        batch = np.stack([core.to_model_array(im) for im in images[start : start + BATCH]])
        with torch.inference_mode():
            out.append(model(torch.from_numpy(batch).to(device)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, core.FEATURE_DIM), dtype=np.float32)


def features_with_views(model, items, device: str, log: list) -> tuple[np.ndarray, np.ndarray]:
    """items: (key, grayscale image). Deterministic features plus the fixed four-view bank."""
    det = embed(model, [im for _, im in items], device)
    views = np.zeros((len(items), core.N_VIEWS, core.FEATURE_DIM), dtype=np.float32)
    params = [core.view_params(key, AUG_SEED) for key, _ in items]
    for v in range(core.N_VIEWS):
        views[:, v] = embed(model, [core.apply_view(im, params[n][v]) for n, (_, im) in enumerate(items)], device)
    log.extend({"key": key, "views": p} for (key, _), p in zip(items, params, strict=True))
    return det, views


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--part", choices=("real", "synthetic"), required=True)
    args = parser.parse_args()
    cfg = core.read_json(args.config)
    work, out = Path(cfg["work_dir"]), Path(cfg["out_dir"])
    started = time.time()
    import torch
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_backbone(stage_backbone(work), device)
    order = core.read_json(out / "data_manifest.json")["class_order"]
    rows = core.read_split_manifest(out / "split_manifest.csv")
    data_root = work / "data" / ("sdi" if cfg["data_source"] == "bosch" else "byod")
    feat_dir = work / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    aug_log: list = []
    train = sorted((r for r in rows if r["selected"]), key=lambda r: (order.index(r["label"]), r["budget_rank"]))

    if args.part == "real":
        train_items = [(r["file_sha256"], core.load_verified_gray(data_root / r["relpath"], r)) for r in train]
        det, views = features_with_views(model, train_items, device, aug_log)
        np.savez(feat_dir / "train.npz", image_id=np.array([r["image_id"] for r in train]), label=np.array([order.index(r["label"]) for r in train]), det=det, views=views)
        print(f"training budget: {len(train)} images x (1 deterministic + {core.N_VIEWS} views)")
        val = sorted((r for r in rows if r["split"] == "val"), key=lambda r: r["image_id"])
        vdet = embed(model, [core.load_verified_gray(data_root / r["relpath"], r) for r in val], device)
        np.savez(feat_dir / "val.npz", image_id=np.array([r["image_id"] for r in val]), label=np.array([order.index(r["label"]) for r in val]), group=np.array([r["group_id"] for r in val]), det=vdet)
        print(f"validation: {len(val)} images (deterministic preprocessing only)")
        core.write_jsonl(out / "augmentation_views.jsonl", aug_log)
        sheet = []
        for label in order:
            r = next(x for x in train if x["label"] == label)
            image = core.load_verified_gray(data_root / r["relpath"], r)
            params = core.view_params(r["file_sha256"], AUG_SEED)
            sheet.append((image, f"{label} original"))
            sheet += [(core.apply_view(image, p), f"view {p['view']} flip={int(p['hflip'])}\nb{p['brightness']} c{p['contrast']}") for p in params]
        core.contact_sheet(sheet, columns=5, title="Conventional augmentation views of TRAINING images (label must survive every view)").save(figures / "augmentation_views.png")
        state = {"backbone": BACKBONE, "device": device, "aug_seed": AUG_SEED, "augment": core.AUGMENT, "n_views": core.N_VIEWS, "preprocessing": core.PREPROCESSING, "seconds": round(time.time() - started, 1), "torch": torch.__version__}
        core.write_json(work / "state" / "features_stage.json", state)
        print(core.canonical_json({"device": device, "train": len(train), "val": len(val), "seconds": state["seconds"]}))
        return

    generated = [r for r in core.read_jsonl(out / "generation_manifest.jsonl") if r["status"] == "eligible"]
    synth_items = [(r["pixel_sha256"], Image.open(out / r["file"]).convert("L")) for r in generated]
    sdet, sviews = features_with_views(model, synth_items, device, aug_log)
    np.savez(feat_dir / "synthetic.npz", candidate_id=np.array([r["candidate_id"] for r in generated]), label=np.array([order.index(r["intended_label"]) for r in generated], dtype=np.int64), det=sdet, views=sviews)
    core.write_jsonl(out / "synthetic_augmentation_views.jsonl", aug_log)
    print(f"synthetic pool: {len(generated)} eligible candidates (converted to grayscale, letterboxed from 512 px)")
    # Synthetic-to-training near-duplicate audit (before fitting): training partition only.
    sig = np.load(work / "audit" / "signatures.npz")
    split_of = {r["image_id"]: r["split"] for r in rows}
    keep = np.array([split_of.get(i) == "train" for i in sig["image_id"]])
    train_sigs, train_ids = sig["signature"][keep].astype(np.float32), sig["image_id"][keep]
    synth_sigs = np.stack([core.signature(im) for _, im in synth_items]) if synth_items else np.zeros((0, core.SIGNATURE_SIDE**2), np.float32)
    best, where = core.max_similarity(synth_sigs, train_sigs)
    audit = {
        "threshold": core.NEAR_DUPLICATE_THRESHOLD,
        "reference": "all training-partition real images",
        "max_correlation": float(best.max()) if len(best) else None,
        "flagged": [{"candidate_id": generated[n]["candidate_id"], "closest_training_image": str(train_ids[where[n]]), "correlation": round(float(best[n]), 4)} for n in np.nonzero(best >= core.NEAR_DUPLICATE_THRESHOLD)[0]],
        "per_candidate": {generated[n]["candidate_id"]: round(float(best[n]), 4) for n in range(len(generated))},
        "policy": "reported, not used to filter candidates",
    }
    core.write_json(out / "synthetic_training_overlap.json", audit)
    print(f"synthetic-to-training audit: max correlation {audit['max_correlation']}, {len(audit['flagged'])} flagged at >= {core.NEAR_DUPLICATE_THRESHOLD}")
    print(core.canonical_json({"device": device, "synthetic": len(generated), "seconds": round(time.time() - started, 1)}))


if __name__ == "__main__":
    core.run_main(main)
