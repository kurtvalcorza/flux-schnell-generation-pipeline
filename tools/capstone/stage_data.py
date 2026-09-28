"""Stage 1 (lab environment): acquire, verify, extract, inventory, audit, group, split and select the real-data budget."""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdi_core as core  # noqa: E402

# Digest of split_manifest.csv produced from the pinned archive when this notebook was written (Pillow 11.3.0,
# NumPy 2.5.3). A different digest means grouping or assignment changed; the run records it rather than hiding it.
REFERENCE_SPLIT_SHA256 = "8b5ef942a82bdacdabf98f2a0b3ab5d3b5adb60ce02dc57831f73147323e65aa"


def acquire_bosch(cfg: dict, work: Path) -> tuple[Path, dict]:
    root = work / "data" / "sdi"
    marker = root / ".extracted.json"
    if marker.is_file() and core.read_json(marker).get("archive_sha256") == core.ARCHIVE_SHA256:
        info = core.read_json(marker)
        info["reused_extraction"] = True
        print(f"reusing verified extraction at {root} ({info['files']} files)")
        return root, info
    zip_path = work / "downloads" / core.ARCHIVE_NAME
    print(f"downloading {core.ARCHIVE_URL}\n  expected {core.ARCHIVE_BYTES:,} bytes, SHA-256 {core.ARCHIVE_SHA256}")
    fetched = core.fetch_archive(core.ARCHIVE_URL, zip_path, expected_bytes=core.ARCHIVE_BYTES, expected_sha256=core.ARCHIVE_SHA256)
    print(f"verified: {fetched}")
    extracted = core.safe_extract(zip_path, root, allowed_suffixes=(".jpg",))
    info = {"archive_sha256": core.ARCHIVE_SHA256, "archive_bytes": core.ARCHIVE_BYTES, "download": fetched, **extracted}
    core.write_json(marker, info)
    if cfg["delete_archive_after_extract"]:
        zip_path.unlink()
        info["archive_deleted_after_extract"] = True
    print(f"extracted {extracted['files']:,} files, {extracted['bytes']:,} bytes (every entry checked before writing)")
    return root, info


def inventory_bosch(root: Path) -> list[dict]:
    records = []
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
    for n, path in enumerate(files, 1):
        rel = path.relative_to(root).as_posix()
        rec = {"image_id": path.stem, "relpath": rel, "path": path, **core.parse_bosch_path(rel), **core.inspect_image(path)}
        records.append(rec)
        if n % 5000 == 0:
            print(f"  inspected {n:,} / {len(files):,}")
    ids = [r["image_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise core.ContractError("image ids are not unique across the archive")
    return records


def acquire_byod(cfg: dict, work: Path) -> tuple[list[dict], dict]:
    source = Path(cfg["byod_zip"])
    if not source.is_file():
        raise core.ContractError(f"BYOD_ZIP_PATH {source} does not exist; upload a zip with manifest.csv and images")
    root = work / "data" / "byod"
    extracted = core.safe_extract(source, root, allowed_suffixes=core.IMAGE_SUFFIXES, allowed_names=(core.BYOD_MANIFEST,), max_entries=50_000, max_total=8_000_000_000, max_member=50_000_000)
    rows = core.load_byod_manifest(root)
    records = []
    for row in rows:
        rec = {**row, **core.inspect_image(row["path"])}
        rec["relpath"] = Path(row["path"]).relative_to(root.resolve()).as_posix()
        records.append(rec)
    return records, {"byod_zip_sha256": core.sha256_file(source), **extracted}


def check_byod(cfg: dict, work: Path, zip_path: str) -> None:
    """Dry run for a BYOD zip: the same extraction, manifest, image, grouping, split and budget rules, no models."""
    cfg = {**cfg, "byod_zip": zip_path}
    scratch = work / "byod_check"
    records, _ = acquire_byod(cfg, scratch)
    bad = [r["relpath"] for r in records if not r["decode_ok"]]
    ok = [r for r in records if r["decode_ok"]]
    grouping = core.assign_groups(ok, np.stack([r["signature"] for r in ok]).astype(np.float32))
    for r, gid in zip(ok, grouping["group_of"], strict=True):
        r["group_id"] = gid
    for r in records:
        if not r["decode_ok"]:
            r["group_id"] = "x_" + r["file_sha256"][:16]
    split = core.build_split_rows(records)
    counts: dict = {}
    for r in records:
        counts.setdefault(r["split"], {}).setdefault(r["label"], 0)
        counts[r["split"]][r["label"]] += 1
    shutil.rmtree(scratch, ignore_errors=True)
    print(core.canonical_json({"byod_check": "PASS", "images": len(records), "undecodable": bad, "groups": grouping["groups"], "class_order": split["class_order"], "split_counts": counts}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--check-byod", default=None, help="validate a BYOD zip without running the experiment")
    args = parser.parse_args()
    cfg = core.read_json(args.config)
    if args.check_byod:
        check_byod(cfg, Path(cfg["work_dir"]), args.check_byod)
        return
    work, out = Path(cfg["work_dir"]), Path(cfg["out_dir"])
    started = time.time()
    if cfg["data_source"] == "bosch":
        root, acquisition = acquire_bosch(cfg, work)
        records = inventory_bosch(root)
        measured: dict = {}
        for r in records:
            measured.setdefault(r["product"], {}).setdefault(r["label"], 0)
            measured[r["product"]][r["label"]] += 1
        official: dict = {}
        for r in records:
            if r["official_split"]:
                key = f"{r['product']}/{r['label']}/{r['official_split']}"
                official[key] = official.get(key, 0) + 1
        inventory = {
            "entries": len(records),
            "measured_counts": measured,
            "documented_counts": core.DOCUMENTED_COUNTS,
            "differences": {p: {c: measured.get(p, {}).get(c, 0) - n for c, n in cls.items()} for p, cls in core.DOCUMENTED_COUNTS.items()},
            "official_split_counts": official,
            "official_split_note": "the source assigns only defect images to train/val/test; normal images carry no official split",
            "modes": sorted({f"{r['mode']} {r['width']}x{r['height']}" for r in records if r["decode_ok"]}),
            "undecodable": [r["relpath"] for r in records if not r["decode_ok"]],
            "specimen_or_session_metadata": "none supplied by the source (unknown, not inferred)",
        }
        experiment = [r for r in records if r["product"] == "A"]
    else:
        experiment, acquisition = acquire_byod(cfg, work)
        inventory = {"entries": len(experiment), "undecodable": [r["relpath"] for r in experiment if not r["decode_ok"]], "declared_groups": sum(1 for r in experiment if r.get("declared_group"))}
    print(f"inventory: {inventory['entries']:,} images, {len(inventory['undecodable'])} undecodable")

    ok = [r for r in experiment if r["decode_ok"]]
    signatures = np.stack([r["signature"] for r in ok]).astype(np.float32)
    grouping = core.assign_groups(ok, signatures)
    for r, gid in zip(ok, grouping["group_of"], strict=True):
        r["group_id"] = gid
    for r in experiment:
        if not r["decode_ok"]:
            r["group_id"] = "x_" + r["file_sha256"][:16]
            r.update(width=0, height=0, mode="", pixel_sha256="")
    split = core.build_split_rows(experiment)
    split_sha = core.write_split_manifest(out / "split_manifest.csv", experiment)
    by_split: dict = {}
    for r in experiment:
        by_split.setdefault(r["split"], {}).setdefault(r["label"], 0)
        by_split[r["split"]][r["label"]] += 1
    selected = {c: sum(1 for r in experiment if r["selected"] and r["label"] == c) for c in split["class_order"]}
    pairs = grouping["near_duplicate_pairs"]
    ids = [r["image_id"] for r in ok]
    conflict_members = sorted(r["image_id"] for r in experiment if r["exclusion_reason"] == "label-conflict-duplicate")
    split_of = [r["split"] for r in ok]
    label_of = [r["label"] for r in ok]
    similar = core.near_duplicate_pairs(signatures, core.AUDIT_SIMILARITY)
    crossing: dict = {}
    for i, j, _ in similar:
        if split_of[i] != split_of[j] and "excluded" not in (split_of[i], split_of[j]):
            key = "/".join(sorted((label_of[i], label_of[j])))
            crossing[key] = crossing.get(key, 0) + 1
    below = sum(1 for _, _, s in similar if core.NEAR_DUPLICATE_THRESHOLD - 0.005 <= s < core.NEAR_DUPLICATE_THRESHOLD)
    official_vs_custom: dict = {}
    for r in experiment:
        if r["official_split"]:
            key = f"official {r['official_split']} -> {r['split']}"
            official_vs_custom[key] = official_vs_custom.get(key, 0) + 1
    manifest = {
        "data_source": cfg["data_source"],
        "source": {"repository": core.SOURCE_REPOSITORY, "commit": core.SOURCE_COMMIT, "archive": core.ARCHIVE_NAME, "url": core.ARCHIVE_URL, "bytes": core.ARCHIVE_BYTES, "sha256": core.ARCHIVE_SHA256, "license": core.DATASET_LICENSE, "notes": core.DATASET_NOTES_URL} if cfg["data_source"] == "bosch" else {"byod": True},
        "acquisition": acquisition,
        "inventory": inventory,
        "experiment_scope": "product A, classes normal / scratches / spots" if cfg["data_source"] == "bosch" else "user-supplied manifest",
        "grouping": {
            "rule": "union of exact decoded-pixel duplicates, 32x32 signature correlation >= threshold, and declared groups",
            "threshold": core.NEAR_DUPLICATE_THRESHOLD,
            "groups": grouping["groups"],
            "exact_duplicate_links": grouping["exact_duplicate_links"],
            "near_duplicate_pairs": len(pairs),
            "example_pairs": [[ids[i], ids[j], s] for i, j, s in pairs[:25]],
            "label_conflict_groups": len(split["label_conflict_groups"]),
            "label_conflict_images": conflict_members,
            "pairs_just_below_threshold": {"range": [core.NEAR_DUPLICATE_THRESHOLD - 0.005, core.NEAR_DUPLICATE_THRESHOLD], "count": below},
            "similar_pairs_crossing_partitions": {"correlation_at_least": core.AUDIT_SIMILARITY, "by_label_pair": crossing},
            "interpretation": "exploratory image-group benchmark: duplicate groups approximate shared capture; they do not prove independent physical specimens, and similar images below the threshold can sit in different partitions",
        },
        "split": {
            "version": core.SPLIT_VERSION,
            "design": "deterministic, approximately class-stratified 60/20/20 group assignment (custom; not the source's split)",
            "fractions": dict(core.SPLIT_FRACTIONS),
            "counts": by_split,
            "official_vs_custom": official_vs_custom,
            "manifest_sha256": split_sha,
            "reference_sha256": REFERENCE_SPLIT_SHA256 if cfg["data_source"] == "bosch" else None,
            "matches_reference": (split_sha == REFERENCE_SPLIT_SHA256) if cfg["data_source"] == "bosch" else None,
        },
        "class_order": split["class_order"],
        "budget": split["budget"],
        "selected": selected,
        "unused_training_images": sum(1 for r in experiment if r["split"] == "train" and not r["selected"]),
        "new_image_candidates": {c: sum(1 for r in experiment if r["split"] == "train" and not r["selected"] and r["label"] == c) for c in split["class_order"]},
        "exemplars": sorted((r["label"], r["budget_rank"], r["image_id"]) for r in experiment if r["exemplar"]),
        "core_version": core.CORE_VERSION,
        "seconds": round(time.time() - started, 1),
    }
    core.write_json(out / "data_manifest.json", manifest)
    if not all(manifest["new_image_candidates"].values()):
        print(f"NOTE: unused training images per class {manifest['new_image_candidates']}: Section 11's new-image preview needs spare images or NEW_IMAGE_DIR; without them it is recorded as not run")
    audit = work / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    np.savez(audit / "signatures.npz", image_id=np.array(ids), signature=signatures.astype(np.float16))
    # Training-only views: selected budget images, and near-duplicate pairs whose members are both training images.
    from PIL import Image

    items = []
    for label in split["class_order"]:
        chosen = sorted((r for r in experiment if r["selected"] and r["label"] == label), key=lambda r: r["budget_rank"])[:8]
        items += [(Image.open(r["path"]), f"{label}\n{r['image_id']}") for r in chosen]
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    core.contact_sheet(items, title="Selected TRAINING images only (held-out images are never previewed)").save(figures / "training_examples.png")
    by_id = {r["image_id"]: r for r in experiment}
    train_pairs = [(ids[i], ids[j], s) for i, j, s in pairs if by_id[ids[i]]["split"] == "train" and by_id[ids[j]]["split"] == "train"][:4]
    pair_items = []
    for a, b, s in train_pairs:
        pair_items += [(Image.open(by_id[a]["path"]), f"{a}\ncorr {s}"), (Image.open(by_id[b]["path"]), f"{b}\nsame group")]
    if pair_items:
        core.contact_sheet(pair_items, columns=4, title="Near-duplicate training pairs grouped together").save(figures / "near_duplicate_pairs.png")
    print(core.canonical_json({"split_counts": by_split, "selected": selected, "split_sha256": split_sha, "matches_reference": manifest["split"]["matches_reference"], "label_conflict_images": len(conflict_members), "near_duplicate_pairs": len(pairs), "similar_pairs_crossing": crossing, "just_below_threshold": below, "seconds": manifest["seconds"]}))


if __name__ == "__main__":
    core.run_main(main)
