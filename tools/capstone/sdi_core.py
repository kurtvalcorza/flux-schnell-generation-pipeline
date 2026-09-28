"""Shared helpers for the Bosch synthetic-defect capstone notebook (standard library + NumPy + Pillow only).

The notebook writes this file into its working directory and every stage script imports it, so acquisition,
validation, grouping, splitting, prompt compilation, slot schedules, metrics, the bootstrap and the classifier
artifact are implemented once and used identically by every stage. Nothing here imports a model library.
"""
# ruff: noqa: E501
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import stat
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

# %% [section] Source, constants and errors

CORE_VERSION = "sdi-core/1"
SOURCE_REPOSITORY = "https://github.com/boschresearch/The-Surface-Defect-Inspection-Dataset"
SOURCE_COMMIT = "c6e0afe66e9dbd9d99326c986ea884ddf2aad9f8"
ARCHIVE_NAME = "SDI_DATASET_v1.zip"
ARCHIVE_URL = f"https://media.githubusercontent.com/media/boschresearch/The-Surface-Defect-Inspection-Dataset/{SOURCE_COMMIT}/{ARCHIVE_NAME}"
ARCHIVE_BYTES = 1_647_097_112
ARCHIVE_SHA256 = "d33ea340151cdf909f3807a37e00d335c66eb8960b0c2eb568fd9fc75d96fa7c"
DATASET_LICENSE = "CC BY-SA 4.0"
DATASET_NOTES_URL = f"{SOURCE_REPOSITORY}/blob/{SOURCE_COMMIT}/DT_GAN_BMVC_SDI_dataset.pdf"
# Counts published in the dataset notes: expectations, not measurements. The notebook measures the archive.
DOCUMENTED_COUNTS = {
    "A": {"normal": 6250, "scratches": 340, "spots": 108},
    "B": {"normal": 6250, "scratches": 167, "spots": 670},
    "C": {"normal": 6250, "scratches": 121, "spots": 258},
}
# Extraction ceilings set from the inspected archive (18,841 entries, 1,644,493,674 expanded bytes,
# largest member 232,910 bytes) with a small margin; anything larger is refused rather than extracted.
MAX_ARCHIVE_ENTRIES = 19_000
MAX_EXPANDED_BYTES = 1_700_000_000
MAX_MEMBER_BYTES = 4_000_000
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
MIN_IMAGE_SIDE = 64
MAX_IMAGE_SIDE = 4096
SIGNATURE_SIDE = 32
# Perceptual grouping threshold: Pearson correlation of 32x32 grayscale thumbnails. In product A the exact-pixel
# duplicates score >= 0.99 and the 18 pairs between 0.98 and 0.99 are consecutive capture frames, but the distribution
# has no gap here: hundreds of pairs sit between 0.97 and 0.98 and ~32,000 exceed 0.95 because the surfaces share
# one texture. 0.98 is a declared cut, not a natural boundary; the data stage reports how many pairs sit just below
# it and how many similar pairs cross partitions.
NEAR_DUPLICATE_THRESHOLD = 0.98
AUDIT_SIMILARITY = 0.95
SPLIT_VERSION = "sdi-a-grouped-602020-v1"
SPLIT_FRACTIONS = (("train", 0.6), ("val", 0.2), ("test", 0.2))
BUDGET_BY_ROLE = (128, 64, 32)  # normal, more frequent defect, less frequent defect
MIN_HELDOUT_PER_CLASS = 5
EXEMPLARS_PER_CLASS = 3
NORMAL = "normal"
CLASS_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
IMAGE_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class ContractError(ValueError):
    """A validation failure that names the broken contract and the corrective action."""


def run_main(entry: Callable[..., None], *args: Any) -> None:
    """Run a stage entry point; a ContractError is printed as one actionable line and exits with status 2."""
    try:
        entry(*args)
    except ContractError as exc:
        print(f"\nCONTRACT FAILURE: {exc}", flush=True)
        raise SystemExit(2) from None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# %% [section] Acquisition and archive safety


def fetch_archive(
    url: str,
    dest: str | Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    timeout: float = 60.0,
    retries: int = 4,
    chunk: int = 8 << 20,
    log: Callable[[str], None] = print,
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Stream `url` to `dest` with timeouts, bounded resumable retries and atomic completion, then verify size and
    SHA-256 before the file is given its final name. A mismatch is refused; nothing substitutes another dataset."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == expected_bytes and sha256_file(dest) == expected_sha256:
        return {"reused": True, "downloaded_bytes": 0, "attempts": 0, "sha256": expected_sha256, "bytes": expected_bytes}
    part = dest.with_name(dest.name + ".part")
    urlopen = opener or urllib.request.urlopen
    downloaded, attempts, complete = 0, 0, False
    started = time.time()
    for attempt in range(1, retries + 2):
        attempts = attempt
        have = part.stat().st_size if part.exists() else 0
        if have > expected_bytes:
            part.unlink()
            have = 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
                resumed = bool(have) and getattr(response, "status", 200) == 206
                mode = "ab" if resumed else "wb"
                have = have if resumed else 0
                next_report = have + (256 << 20)
                with open(part, mode) as handle:
                    while block := response.read(chunk):
                        handle.write(block)
                        downloaded += len(block)
                        have += len(block)
                        if have >= next_report:
                            log(f"  {have / 1e9:.2f} / {expected_bytes / 1e9:.2f} GB")
                            next_report += 256 << 20
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            wait = min(2**attempt, 30)
            log(f"  attempt {attempt} interrupted ({type(exc).__name__}: {exc}); retrying in {wait} s")
            time.sleep(wait)
            continue
        size = part.stat().st_size
        if size == expected_bytes:
            complete = True
            break
        if size < 1024 and expected_bytes > 1024:
            text = part.read_bytes()[:200].decode("utf-8", "replace")
            if "git-lfs" in text:
                part.unlink()
                raise ContractError(f"{url} returned a Git LFS pointer, not the archive; the LFS media host must serve the object")
        log(f"  attempt {attempt} ended at {size} of {expected_bytes} bytes; resuming")
    if not complete:
        raise ContractError(
            f"download of {url} did not complete after {attempts} attempts. A timeout is an access problem, not "
            "evidence of a corrupt dataset: re-run this cell (completed bytes are kept and resumed)."
        )
    digest = sha256_file(part)
    if digest != expected_sha256:
        part.unlink()
        raise ContractError(f"archive SHA-256 {digest} differs from the pinned {expected_sha256}; refusing to extract it")
    os.replace(part, dest)
    return {"reused": False, "downloaded_bytes": downloaded, "attempts": attempts, "sha256": digest, "bytes": expected_bytes, "seconds": round(time.time() - started, 1)}


def check_zip_members(
    archive: zipfile.ZipFile,
    *,
    allowed_suffixes: Sequence[str],
    allowed_names: Sequence[str] = (),
    max_entries: int = MAX_ARCHIVE_ENTRIES,
    max_total: int = MAX_EXPANDED_BYTES,
    max_member: int = MAX_MEMBER_BYTES,
) -> list[zipfile.ZipInfo]:
    """Refuse absolute paths, traversal, symlinks, encrypted or duplicate entries, unsupported file types and
    archives whose declared expanded size exceeds the ceiling. Returns the file entries to extract."""
    infos = archive.infolist()
    if len(infos) > max_entries:
        raise ContractError(f"archive has {len(infos)} entries; the ceiling is {max_entries}")
    seen: set[str] = set()
    files: list[zipfile.ZipInfo] = []
    total = 0
    for info in infos:
        name = info.filename
        if stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF):
            raise ContractError(f"symlink entry refused: {name!r}")
        if name.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", name) or "\\" in name or "\x00" in name:
            raise ContractError(f"absolute or non-POSIX path refused: {name!r}")
        if any(part == ".." for part in name.split("/")):
            raise ContractError(f"path traversal refused: {name!r}")
        if name in seen:
            raise ContractError(f"duplicate archive entry refused: {name!r}")
        seen.add(name)
        if info.is_dir():
            continue
        if info.flag_bits & 0x1:
            raise ContractError(f"encrypted entry refused: {name!r}")
        base = name.rsplit("/", 1)[-1]
        if base not in allowed_names and Path(base).suffix.lower() not in allowed_suffixes:
            raise ContractError(f"unsupported file refused: {name!r}; allowed: {sorted(allowed_suffixes) + sorted(allowed_names)} (remove OS metadata such as __MACOSX/ or .DS_Store and re-zip)")
        if info.file_size > max_member:
            raise ContractError(f"entry {name!r} expands to {info.file_size} bytes; the per-file ceiling is {max_member}")
        total += info.file_size
        if total > max_total:
            raise ContractError(f"archive expands beyond the {max_total}-byte ceiling")
        files.append(info)
    return files


def safe_extract(zip_path: str | Path, root: str | Path, **limits: Any) -> dict[str, Any]:
    """Extract into `<root>.partial` with every member checked, count the bytes actually written against the
    ceilings, and rename to `root` only when everything succeeded."""
    root = Path(root)
    staging = root.with_name(root.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    base = staging.resolve()
    max_total = limits.get("max_total", MAX_EXPANDED_BYTES)
    written = 0
    with zipfile.ZipFile(zip_path) as archive:
        files = check_zip_members(archive, **limits)
        for info in files:
            target = (staging / info.filename).resolve()
            if not target.is_relative_to(base):
                raise ContractError(f"entry {info.filename!r} resolves outside the extraction root")
            target.parent.mkdir(parents=True, exist_ok=True)
            count = 0
            with archive.open(info) as source, open(target, "wb") as sink:
                while block := source.read(1 << 20):
                    count += len(block)
                    if count > info.file_size or written + count > max_total:
                        raise ContractError(f"entry {info.filename!r} expanded beyond its declared size")
                    sink.write(block)
            written += count
    shutil.rmtree(root, ignore_errors=True)
    os.replace(staging, root)
    return {"files": len(files), "bytes": written}


# %% [section] Inventory and image validation

BOSCH_OK = re.compile(r"^SDI_DATASET_v1/([ABC])_ok/([ABC])_ok_(\d{5})\.jpg$")
BOSCH_NOK = re.compile(r"^SDI_DATASET_v1/([ABC])_nok/(train|val|test)/(scratches|spots)/([ABC])_(sc|sp)_(train|val|test)_(\d{5})\.jpg$")


def parse_bosch_path(relpath: str) -> dict[str, str]:
    """Map an archive path to product, label and the source's own split assignment (defects only)."""
    match = BOSCH_OK.match(relpath)
    if match and match.group(1) == match.group(2):
        return {"product": match.group(1), "label": NORMAL, "official_split": ""}
    match = BOSCH_NOK.match(relpath)
    if match:
        product, split, folder, product2, abbrev, split2, _ = match.groups()
        if product == product2 and split == split2 and abbrev == folder[:2]:
            return {"product": product, "label": folder, "official_split": split}
    raise ContractError(f"unexpected archive path {relpath!r}; the documented layout is <P>_ok/*.jpg and <P>_nok/<split>/<class>/*.jpg")


def load_gray(path: str | Path) -> tuple[Image.Image, str]:
    """Decode an image and return it as 8-bit grayscale plus its original mode (conversions are reported)."""
    with Image.open(path) as image:
        image.load()
        mode = image.mode
        if mode == "L":
            return image.copy(), mode
        if mode in ("RGB", "RGBA", "P", "LA", "I;16", "I", "F"):
            if mode in ("I;16", "I", "F"):
                array = np.asarray(image, dtype=np.float64)
                if not np.isfinite(array).all():
                    raise ContractError(f"{path}: non-finite pixel values")
                return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)), mode
            return image.convert("L"), mode
    raise ContractError(f"{path}: unsupported image mode {mode!r}")


def signature(image: Image.Image) -> np.ndarray:
    """32x32 grayscale thumbnail, zero mean, unit norm: the dot product of two signatures is their Pearson correlation."""
    thumb = np.asarray(image.convert("L").resize((SIGNATURE_SIDE, SIGNATURE_SIDE), Image.BILINEAR), dtype=np.float32).ravel()
    thumb = thumb - thumb.mean()
    norm = float(np.linalg.norm(thumb))
    return thumb / norm if norm > 1e-6 else np.zeros_like(thumb)


def inspect_image(path: str | Path) -> dict[str, Any]:
    """File digest, decodability, size, mode, decoded-pixel digest and signature. Never raises for a bad image:
    failures are recorded so the inventory can report them."""
    path = Path(path)
    record: dict[str, Any] = {"bytes": path.stat().st_size, "file_sha256": sha256_file(path)}
    try:
        image, mode = load_gray(path)
        array = np.asarray(image)
        if not (min(image.size) >= MIN_IMAGE_SIDE and max(image.size) <= MAX_IMAGE_SIDE):
            raise ContractError(f"side {image.size} outside {MIN_IMAGE_SIDE}..{MAX_IMAGE_SIDE}")
        record.update(
            decode_ok=True,
            width=image.size[0],
            height=image.size[1],
            mode=mode,
            pixel_sha256=sha256_bytes(f"{array.shape}|".encode() + array.tobytes()),
            signature=signature(image),
            error="",
        )
    except Exception as exc:  # corrupt files are data to report, not a crash
        record.update(decode_ok=False, width=0, height=0, mode="", pixel_sha256="", signature=None, error=f"{type(exc).__name__}: {exc}"[:200])
    return record


# %% [section] Duplicate grouping, split and budget


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def near_duplicate_pairs(signatures: np.ndarray, threshold: float = NEAR_DUPLICATE_THRESHOLD, block: int = 2048) -> list[tuple[int, int, float]]:
    """All pairs i < j whose signature correlation is >= threshold (blockwise, so memory stays bounded)."""
    pairs: list[tuple[int, int, float]] = []
    n = len(signatures)
    for start in range(0, n, block):
        sims = signatures[start : start + block] @ signatures.T
        rows, cols = np.nonzero(sims >= threshold)
        for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
            i = start + r
            if i < c:
                pairs.append((i, c, round(float(sims[r, c]), 5)))
    return pairs


def max_similarity(queries: np.ndarray, references: np.ndarray, block: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """For each query signature, the highest correlation with any reference signature and its index."""
    if len(references) == 0 or len(queries) == 0:
        return np.full(len(queries), -1.0, dtype=np.float32), np.full(len(queries), -1)
    best = np.empty(len(queries), dtype=np.float32)
    where = np.empty(len(queries), dtype=np.int64)
    for start in range(0, len(queries), block):
        sims = queries[start : start + block] @ references.T
        where[start : start + block] = sims.argmax(axis=1)
        best[start : start + block] = sims.max(axis=1)
    return best, where


def assign_groups(records: Sequence[Mapping[str, Any]], signatures: np.ndarray, *, threshold: float = NEAR_DUPLICATE_THRESHOLD) -> dict[str, Any]:
    """Union exact decoded-pixel duplicates, perceptual near duplicates and any user-declared groups. Returns the
    group id of every record plus the evidence. Groups approximate shared origin; they do not prove independent
    physical specimens."""
    n = len(records)
    uf = _UnionFind(n)
    by_pixels: dict[str, int] = {}
    exact = 0
    for i, rec in enumerate(records):
        key = rec["pixel_sha256"]
        if key in by_pixels:
            uf.union(by_pixels[key], i)
            exact += 1
        else:
            by_pixels[key] = i
    declared: dict[str, int] = {}
    for i, rec in enumerate(records):
        user = rec.get("declared_group") or ""
        if user:
            if user in declared:
                uf.union(declared[user], i)
            else:
                declared[user] = i
    pairs = near_duplicate_pairs(signatures, threshold)
    for i, j, _ in pairs:
        uf.union(i, j)
    members: dict[int, list[int]] = {}
    for i in range(n):
        members.setdefault(uf.find(i), []).append(i)
    group_of = [""] * n
    for idx in members.values():
        gid = "g_" + min(records[i]["pixel_sha256"] for i in idx)[:16]
        for i in idx:
            group_of[i] = gid
    return {"group_of": group_of, "exact_duplicate_links": exact, "near_duplicate_pairs": pairs, "groups": len(members)}


def _rank(*parts: Any) -> str:
    return sha256_bytes("|".join(str(p) for p in parts).encode())


def assign_split(groups: Mapping[str, Mapping[str, Any]], *, version: str = SPLIT_VERSION, fractions: Sequence[tuple[str, float]] = SPLIT_FRACTIONS) -> dict[str, str]:
    """Deterministic, approximately class-stratified group assignment: within each class, groups are visited in a
    hash order and each goes to the partition with the largest relative shortfall against its target."""
    out: dict[str, str] = {}
    labels = sorted({g["label"] for g in groups.values()})
    for label in labels:
        gids = sorted((gid for gid, g in groups.items() if g["label"] == label), key=lambda gid: _rank(version, "split", gid))
        total = sum(groups[gid]["size"] for gid in gids)
        counts = {name: 0 for name, _ in fractions}
        for gid in gids:
            name = max(fractions, key=lambda item: ((item[1] * total - counts[item[0]]) / (item[1] * total), -[f[0] for f in fractions].index(item[0])))[0]
            out[gid] = name
            counts[name] += groups[gid]["size"]
    return out


def check_group_leakage(rows: Sequence[Mapping[str, Any]]) -> None:
    """Every group must sit in exactly one partition; raises naming the groups that cross."""
    seen: dict[str, set[str]] = {}
    for row in rows:
        if row["split"] in ("train", "val", "test"):
            seen.setdefault(row["group_id"], set()).add(row["split"])
    crossing = sorted(gid for gid, splits in seen.items() if len(splits) > 1)
    if crossing:
        raise ContractError(f"{len(crossing)} group(s) cross partitions, e.g. {crossing[:5]}; the split is invalid")


def class_order_from_counts(counts: Mapping[str, int]) -> list[str]:
    """normal first, then defect classes by descending count (ties by name)."""
    if NORMAL not in counts:
        raise ContractError(f"a class named {NORMAL!r} is required; found {sorted(counts)}")
    defects = sorted((c for c in counts if c != NORMAL), key=lambda c: (-counts[c], c))
    if len(defects) != 2:
        raise ContractError(f"exactly two defect classes are supported besides {NORMAL!r}; found {defects}")
    return [NORMAL, *defects]


def build_split_rows(records: Sequence[dict[str, Any]], *, version: str = SPLIT_VERSION, budgets: Sequence[int] = BUDGET_BY_ROLE) -> dict[str, Any]:
    """Exclude undecodable images and label-conflict groups, assign groups to partitions, check leakage, then
    select the fixed real-data training budget (one image per group) and the captioning exemplars."""
    groups: dict[str, dict[str, Any]] = {}
    for rec in records:
        if not rec["decode_ok"]:
            rec["split"], rec["exclusion_reason"] = "excluded", "decode-failure"
            continue
        g = groups.setdefault(rec["group_id"], {"labels": set(), "size": 0, "members": []})
        g["labels"].add(rec["label"])
        g["size"] += 1
        g["members"].append(rec)
    conflicts = sorted(gid for gid, g in groups.items() if len(g["labels"]) > 1)
    usable = {}
    for gid, g in groups.items():
        if gid in conflicts:
            for rec in g["members"]:
                rec["split"], rec["exclusion_reason"] = "excluded", "label-conflict-duplicate"
        else:
            usable[gid] = {"label": next(iter(g["labels"])), "size": g["size"], "members": g["members"]}
    assignment = assign_split(usable, version=version)
    for gid, g in usable.items():
        for rec in g["members"]:
            rec["split"], rec["exclusion_reason"] = assignment[gid], ""
    check_group_leakage(records)
    counts = {}
    for rec in records:
        if rec["split"] != "excluded":
            counts[rec["label"]] = counts.get(rec["label"], 0) + 1
    order = class_order_from_counts(counts)
    budget = dict(zip(order, budgets, strict=True))
    for rec in records:
        rec["selected"] = False
        rec["exemplar"] = False
        rec["budget_rank"] = -1
    shortfalls = []
    for label in order:
        train_groups = sorted((gid for gid, g in usable.items() if g["label"] == label and assignment[gid] == "train"), key=lambda gid: _rank(version, "budget", gid))
        if len(train_groups) < budget[label]:
            shortfalls.append(f"{label}: {len(train_groups)} training groups for a budget of {budget[label]}")
            continue
        for rank, gid in enumerate(train_groups[: budget[label]]):
            rep = min(usable[gid]["members"], key=lambda r: _rank(version, "representative", r["file_sha256"]))
            rep["selected"], rep["budget_rank"], rep["exemplar"] = True, rank, rank < EXEMPLARS_PER_CLASS
        for split in ("val", "test"):
            n = sum(1 for r in records if r["label"] == label and r["split"] == split)
            if n < MIN_HELDOUT_PER_CLASS:
                shortfalls.append(f"{label}: {n} {split} images (minimum {MIN_HELDOUT_PER_CLASS})")
    if shortfalls:
        raise ContractError("insufficient independent examples under the grouped split: " + "; ".join(shortfalls) + ". Revise the versioned split design explicitly; held-out images are never borrowed and grouping is never relaxed silently.")
    return {"class_order": order, "budget": budget, "label_conflict_groups": conflicts, "counts": counts}


SPLIT_COLUMNS = ("image_id", "relpath", "product", "label", "official_split", "group_id", "split", "selected", "budget_rank", "exemplar", "exclusion_reason", "file_sha256", "pixel_sha256", "width", "height", "mode")


def write_split_manifest(path: str | Path, records: Sequence[Mapping[str, Any]]) -> str:
    """Deterministic CSV (sorted by image id, fixed columns, LF endings); returns its SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(SPLIT_COLUMNS)
        for rec in sorted(records, key=lambda r: r["image_id"]):
            writer.writerow([int(rec[c]) if isinstance(rec[c], bool) else rec[c] for c in SPLIT_COLUMNS])
    return sha256_file(path)


def read_split_manifest(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["selected"] = row["selected"] == "1"
        row["exemplar"] = row["exemplar"] == "1"
        row["budget_rank"] = int(row["budget_rank"])
    return rows


# %% [section] Bring-your-own-data manifest

BYOD_MANIFEST = "manifest.csv"
BYOD_REQUIRED = ("image_id", "path", "class")


def load_byod_manifest(root: str | Path) -> list[dict[str, Any]]:
    """Validate a BYOD folder: manifest.csv with image_id, path, class and an optional group column; safe relative
    paths to existing images; exactly one `normal` class and two defect classes; groups that never span classes."""
    root = Path(root)
    manifest = root / BYOD_MANIFEST
    if not manifest.is_file():
        candidates = sorted(root.glob(f"*/{BYOD_MANIFEST}"))
        if len(candidates) == 1:
            root, manifest = candidates[0].parent, candidates[0]
        else:
            raise ContractError(f"{BYOD_MANIFEST} not found at the top of the BYOD folder (or one level down)")
    with open(manifest, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = [h.strip() for h in (reader.fieldnames or [])]
        rows = list(reader)
    if len(set(header)) != len(header):
        raise ContractError(f"{BYOD_MANIFEST} has duplicate column names: {header}")
    missing = [c for c in BYOD_REQUIRED if c not in header]
    if missing:
        raise ContractError(f"{BYOD_MANIFEST} is missing required column(s) {missing}; expected {list(BYOD_REQUIRED)} plus optional 'group'")
    out, ids, group_labels = [], set(), {}
    base = root.resolve()
    for n, raw in enumerate(rows, start=2):
        row = {k.strip(): (v or "").strip() for k, v in raw.items() if k}
        image_id, rel, label, group = row["image_id"], row["path"], row["class"], row.get("group", "")
        if not IMAGE_ID_RE.match(image_id):
            raise ContractError(f"line {n}: image_id {image_id!r} must match {IMAGE_ID_RE.pattern}")
        if image_id in ids:
            raise ContractError(f"line {n}: duplicate image_id {image_id!r}")
        ids.add(image_id)
        if not CLASS_NAME_RE.match(label):
            raise ContractError(f"line {n}: class {label!r} must match {CLASS_NAME_RE.pattern}")
        if rel.startswith(("/", "\\")) or ".." in rel.split("/") or "\\" in rel:
            raise ContractError(f"line {n}: path {rel!r} must be relative to the manifest without '..'")
        target = (root / rel).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            raise ContractError(f"line {n}: image {rel!r} not found inside the BYOD folder")
        if target.suffix.lower() not in IMAGE_SUFFIXES:
            raise ContractError(f"line {n}: {rel!r} is not one of {IMAGE_SUFFIXES}")
        if group:
            if not IMAGE_ID_RE.match(group):
                raise ContractError(f"line {n}: group {group!r} must match {IMAGE_ID_RE.pattern}")
            if group_labels.setdefault(group, label) != label:
                raise ContractError(f"line {n}: group {group!r} spans classes {group_labels[group]!r} and {label!r}; a specimen/session group must carry one label")
        out.append({"image_id": image_id, "path": target, "relpath": rel, "label": label, "declared_group": group, "product": "byod", "official_split": ""})
    if not out:
        raise ContractError(f"{BYOD_MANIFEST} has no rows")
    counts: dict[str, int] = {}
    for row in out:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    class_order_from_counts(counts)
    return out


# %% [section] Descriptions to bounded prompts

VOCABULARY = {
    "texture": ("horizontal lines", "horizontal stripes", "vertical lines", "parallel lines", "striped", "grooves", "grainy", "granular", "fine texture", "rough", "smooth", "brushed", "uniform", "noisy"),
    "illumination": ("even lighting", "uniform lighting", "low contrast", "high contrast", "bright", "dark", "shadow", "reflective", "glossy", "matte"),
    "shape": ("thin", "thick", "long", "short", "straight", "curved", "diagonal", "linear", "elongated", "small", "large", "round", "circular", "oval", "irregular", "blob", "dot", "speck", "cluster", "streak", "line", "patch"),
}
TERM_LIMITS = {"texture": 2, "illumination": 1, "shape": 3}
MAX_DESCRIPTION_CHARS = 1500
MAX_PROMPT_CHARS = 300
PROMPT_BASE = "close-up grayscale industrial inspection photograph of a flat manufactured surface"
SINGULAR = {"scratches": "scratch", "spots": "spot"}
PROMPT_CHARSET = re.compile(r"^[a-z0-9 ,.\-]+$")


def extract_terms(text: str) -> dict[str, list[str]]:
    """Whitelisted descriptor terms found in a description, in vocabulary order, capped per category. Only these
    terms can reach a prompt: model output is treated as evidence to match, never as text to paste or execute."""
    lowered = text.lower()
    found: dict[str, list[str]] = {}
    for category, terms in VOCABULARY.items():
        hits = [t for t in terms if re.search(rf"(?<![a-z]){re.escape(t)}(?![a-z])", lowered)]
        found[category] = hits[: TERM_LIMITS[category]]
    return found


def description_problems(text: Any, *, need: Sequence[str]) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return ["empty description"]
    problems = []
    if len(text) > MAX_DESCRIPTION_CHARS:
        problems.append(f"longer than {MAX_DESCRIPTION_CHARS} characters")
    terms = extract_terms(text)
    if not any(terms[c] for c in need):
        problems.append(f"no whitelisted {'/'.join(need)} term")
    return problems


def defect_word(label: str) -> str:
    return SINGULAR.get(label, label.replace("_", " "))


def compile_prompts(caption_records: Sequence[Mapping[str, Any]], class_order: Sequence[str]) -> list[dict[str, Any]]:
    """One bounded prompt per (defect class, exemplar k): background terms from normal exemplar k, mark terms from
    defect exemplar k. A description that fails validation falls back to the neutral class template, recorded as
    such and never presented as Phi-4-guided."""
    by_class: dict[str, list[Mapping[str, Any]]] = {}
    for rec in sorted(caption_records, key=lambda r: (r["label"], r["exemplar_rank"])):
        by_class.setdefault(rec["label"], []).append(rec)
    normals = by_class.get(class_order[0], [])
    prompts = []
    for label in class_order[1:]:
        defects = by_class.get(label, [])
        if not defects:
            raise ContractError(f"no caption records for class {label!r}")
        for k, defect in enumerate(defects):
            normal = normals[k] if k < len(normals) else None
            defect_text, normal_text = defect.get("response", ""), (normal or {}).get("response", "")
            problems = description_problems(defect_text, need=("shape", "texture"))
            background = extract_terms(normal_text) if normal and not description_problems(normal_text, need=("texture", "illumination")) else {"texture": [], "illumination": []}
            mark = extract_terms(defect_text) if not problems else {"texture": [], "illumination": [], "shape": []}
            texture = background["texture"] or mark["texture"]
            illumination = background["illumination"] or mark["illumination"]
            if problems:
                text = f"{PROMPT_BASE}, showing one {defect_word(label)} defect, sharp focus, no text"
            else:
                parts = [PROMPT_BASE, *texture, *illumination]
                shape = " ".join(mark["shape"])
                parts.append(f"showing one {defect_word(label)} defect" + (f" as a {shape} mark" if shape else ""))
                parts += ["sharp focus", "no text"]
                text = ", ".join(parts)
            if len(text) > MAX_PROMPT_CHARS or not PROMPT_CHARSET.match(text):
                raise ContractError(f"compiled prompt violates the bound: {text!r}")
            prompts.append({
                "prompt_id": f"{label}-{k}",
                "label": label,
                "exemplar_rank": k,
                "text": text,
                "phi4_guided": not problems,
                "fallback_reason": "; ".join(problems),
                "terms": {"texture": texture, "illumination": illumination, "shape": mark["shape"]},
                "source_image_ids": [defect["image_id"]] + ([normal["image_id"]] if normal else []),
            })
    return prompts


# %% [section] Synthetic candidate eligibility

GENERATION_CANDIDATES = 32
MIN_USABLE_CANDIDATES = 24


def pixel_digest(array: np.ndarray) -> str:
    return sha256_bytes(f"{array.shape}|".encode() + np.ascontiguousarray(array).tobytes())


def candidate_eligibility(pixels: np.ndarray | None, *, side: int, seen: set[str], real: set[str]) -> tuple[str, str, str]:
    """Default eligibility checks only decoding, dimensions, finite values and exact duplicates: never classifier
    confidence, CLIP score, test performance or model approval. Returns (status, reason, digest of the uint8 image)."""
    if pixels is None:
        return "rejected", "decode-failure", ""
    if pixels.shape[:2] != (side, side) or pixels.ndim not in (2, 3):
        return "rejected", f"dimensions {tuple(pixels.shape)}", ""
    if not np.isfinite(pixels).all():
        return "rejected", "non-finite pixel values", ""
    gray = np.asarray(Image.fromarray(to_uint8(pixels)).convert("L"))
    digest = pixel_digest(gray)
    if digest in real:
        return "rejected", "exact duplicate of a real image", digest
    if digest in seen:
        return "rejected", "exact duplicate of an earlier candidate", digest
    return "eligible", "", digest


def to_uint8(pixels: np.ndarray) -> np.ndarray:
    """[-1, 1] float (VAE output) or already-uint8 pixels to uint8."""
    if pixels.dtype == np.uint8:
        return pixels
    return np.clip((pixels.astype(np.float32) + 1.0) * 127.5, 0, 255).round().astype(np.uint8)


def generation_status(rows: Sequence[Mapping[str, Any]], defect_classes: Sequence[str], *, expected: int = GENERATION_CANDIDATES, minimum: int = MIN_USABLE_CANDIDATES) -> dict[str, Any]:
    usable = {c: sum(1 for r in rows if r["intended_label"] == c and r["status"] == "eligible") for c in defect_classes}
    attempted = {c: sum(1 for r in rows if r["intended_label"] == c) for c in defect_classes}
    shortfall = [f"{c}: {usable[c]} usable of {attempted[c]} attempts (minimum {minimum})" for c in defect_classes if usable[c] < minimum or attempted[c] != expected]
    return {"complete": not shortfall, "usable": usable, "attempted": attempted, "shortfall": shortfall}


# %% [section] Preprocessing and fixed augmentation views

INPUT_SIDE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
N_VIEWS = 4
AUGMENT = {"hflip_probability": 0.5, "brightness": (0.9, 1.1), "contrast": (0.9, 1.1)}
PREPROCESSING = {
    "input": "8-bit grayscale (RGB inputs converted with PIL 'L')",
    "letterbox": f"resize the longer side to {INPUT_SIDE} px (bilinear, antialiased), centre on a {INPUT_SIDE}x{INPUT_SIDE} black canvas",
    "channels": "grayscale replicated to 3 channels",
    "normalisation": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD, "scale": "pixel / 255"},
}


def letterbox(image: Image.Image, side: int = INPUT_SIDE) -> Image.Image:
    image = image.convert("L")
    scale = side / max(image.size)
    size = (max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale)))
    resized = image.resize(size, Image.BILINEAR, reducing_gap=None)
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(resized, ((side - size[0]) // 2, (side - size[1]) // 2))
    return canvas


def to_model_array(image: Image.Image) -> np.ndarray:
    """(3, 224, 224) float32 normalised array from a letterboxed grayscale image."""
    gray = np.asarray(letterbox(image), dtype=np.float32) / 255.0
    return np.stack([(gray - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)]).astype(np.float32)


def view_params(key: str, aug_seed: int, n_views: int = N_VIEWS) -> list[dict[str, Any]]:
    """Fixed, recorded label-preserving transforms for one image: horizontal flip and mild brightness/contrast.
    No crop, so a defect cannot be cut out of the frame."""
    rng = np.random.default_rng([aug_seed, int(key[:15], 16)])
    params = []
    for view in range(n_views):
        params.append({
            "view": view,
            "hflip": bool(rng.random() < AUGMENT["hflip_probability"]),
            "brightness": round(float(rng.uniform(*AUGMENT["brightness"])), 4),
            "contrast": round(float(rng.uniform(*AUGMENT["contrast"])), 4),
        })
    return params


def apply_view(image: Image.Image, params: Mapping[str, Any]) -> Image.Image:
    out = image.convert("L")
    if params["hflip"]:
        out = ImageOps.mirror(out)
    out = ImageEnhance.Brightness(out).enhance(params["brightness"])
    return ImageEnhance.Contrast(out).enhance(params["contrast"])


# %% [section] Matched training schedule

HEAD = {"optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 32, "epochs": 30, "updates_per_epoch": 32}
SEEDS = (17, 29, 43)
CANONICAL_SEED = 17
SYNTHETIC_PROBABILITY = 0.5
ARMS = ("A", "B", "C")
ARM_NAMES = {"majority": "majority baseline", "A": "A: real-only", "B": "B: conventional augmentation", "C": "C: synthetic augmentation"}


def slot_schedule(seed: int, class_sizes: Sequence[int], synth_sizes: Sequence[int], *, n_views: int = N_VIEWS, epochs: int = HEAD["epochs"], updates: int = HEAD["updates_per_epoch"], batch: int = HEAD["batch_size"]) -> dict[str, np.ndarray]:
    """Every random draw the three arms need, made once per seed so the arms differ only by where a slot's image
    comes from. Each batch is class-balanced (11/11/10, rotating which class gets 10)."""
    k = len(class_sizes)
    rng = np.random.default_rng([seed, 5107])
    total_updates = epochs * updates
    base, extra = divmod(batch, k)
    cls = np.empty((total_updates, batch), dtype=np.int64)
    for u in range(total_updates):
        per_class = [base + (1 if ((c - u) % k) < extra else 0) for c in range(k)]
        cls[u] = np.repeat(np.arange(k), per_class)
    cls = cls.ravel()
    uniform = rng.random(cls.shape)
    real_idx = np.floor(uniform * np.asarray(class_sizes)[cls]).astype(np.int64)
    view = rng.integers(0, n_views, size=cls.shape)
    coin = rng.random(cls.shape)
    synth_uniform = rng.random(cls.shape)
    synth_idx = np.floor(synth_uniform * np.maximum(np.asarray(synth_sizes), 1)[cls]).astype(np.int64)
    return {"cls": cls, "real_idx": real_idx, "view": view, "coin": coin, "synth_idx": synth_idx, "updates": np.int64(total_updates), "batch": np.int64(batch)}


def arm_sources(schedule: Mapping[str, np.ndarray], arm: str, *, synth_probability: float = SYNTHETIC_PROBABILITY, synth_sizes: Sequence[int] | None = None) -> dict[str, np.ndarray]:
    """Resolve each slot to (is_synthetic, index, view); view -1 means the deterministic (unaugmented) features.
    Normal slots (class 0) always draw real images; a defect slot in arm C draws its synthetic pool with probability
    `synth_probability`."""
    cls = schedule["cls"]
    if arm == "A":
        return {"synthetic": np.zeros_like(cls, dtype=bool), "index": schedule["real_idx"], "view": np.full_like(cls, -1)}
    if arm == "B":
        return {"synthetic": np.zeros_like(cls, dtype=bool), "index": schedule["real_idx"], "view": schedule["view"]}
    if arm == "C":
        available = np.asarray(synth_sizes if synth_sizes is not None else [0] * (int(cls.max()) + 1)) > 0
        synthetic = (cls > 0) & available[cls] & (schedule["coin"] < synth_probability)
        index = np.where(synthetic, schedule["synth_idx"], schedule["real_idx"])
        return {"synthetic": synthetic, "index": index, "view": schedule["view"]}
    raise ValueError(f"unknown arm {arm!r}")


# %% [section] Metrics and the paired group bootstrap


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(np.asarray(y_true) * k + np.asarray(y_pred), minlength=k * k).reshape(k, k)


def metrics_from_confusion(cm: np.ndarray, class_names: Sequence[str]) -> dict[str, Any]:
    """Per-class precision/recall/F1/support with undefined values reported as None, plus accuracy, balanced
    accuracy and macro-F1 over the classes present in the evaluation set (missing classes are listed)."""
    cm = np.asarray(cm, dtype=np.int64)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    per_class, f1s, recalls = {}, [], []
    for i, name in enumerate(class_names):
        precision = float(tp[i] / predicted[i]) if predicted[i] else None
        recall = float(tp[i] / support[i]) if support[i] else None
        denominator = 2 * tp[i] + (predicted[i] - tp[i]) + (support[i] - tp[i])
        f1 = float(2 * tp[i] / denominator) if support[i] else None
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(support[i]), "predicted": int(predicted[i])}
        if support[i]:
            f1s.append(f1 if denominator else 0.0)
            recalls.append(recall)
    total = int(cm.sum())
    return {
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "accuracy": float(tp.sum() / total) if total else None,
        "per_class": per_class,
        "missing_classes": [n for i, n in enumerate(class_names) if not support[i]],
        "undefined": [f"{n}.precision" for n, v in per_class.items() if v["precision"] is None],
        "confusion": cm.tolist(),
        "n": total,
    }


def macro_f1_batch(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    return metrics_from_confusion(confusion_matrix(y_true, y_pred, k), [str(i) for i in range(k)])["macro_f1"]


def group_bootstrap_indices(labels: np.ndarray, groups: Sequence[str], *, n_resamples: int, seed: int) -> list[np.ndarray]:
    """Class-stratified group bootstrap: within each class stratum, resample its groups with replacement (same
    group count) and take all their members. Every arm is scored on the same resamples."""
    labels = np.asarray(labels)
    strata: dict[int, dict[str, list[int]]] = {}
    for i, (label, gid) in enumerate(zip(labels.tolist(), groups, strict=True)):
        strata.setdefault(label, {}).setdefault(gid, []).append(i)
    rng = np.random.default_rng([seed, 9091])
    ordered = [(label, [np.asarray(v) for _, v in sorted(g.items())]) for label, g in sorted(strata.items())]
    out = []
    for _ in range(n_resamples):
        parts = []
        for _, members in ordered:
            picks = rng.integers(0, len(members), size=len(members))
            parts.extend(members[p] for p in picks)
        out.append(np.concatenate(parts))
    return out


def paired_contrast(predictions: Mapping[tuple[str, int], np.ndarray], y_true: np.ndarray, groups: Sequence[str], *, arm: str, reference: str, seeds: Sequence[int], k: int, n_resamples: int = 1000, seed: int = 2026) -> dict[str, Any]:
    """Mean over seeds of macro-F1(arm) - macro-F1(reference), with an approximate percentile interval from the
    paired, class-stratified group bootstrap. Conditional on this split and this generated pool."""
    per_seed = {s: macro_f1_batch(y_true, predictions[(arm, s)], k) - macro_f1_batch(y_true, predictions[(reference, s)], k) for s in seeds}
    observed = float(np.mean(list(per_seed.values())))
    draws = []
    for idx in group_bootstrap_indices(y_true, groups, n_resamples=n_resamples, seed=seed):
        yt = y_true[idx]
        draws.append(np.mean([macro_f1_batch(yt, predictions[(arm, s)][idx], k) - macro_f1_batch(yt, predictions[(reference, s)][idx], k) for s in seeds]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"contrast": f"{arm} - {reference}", "mean_difference": observed, "per_seed": {str(s): float(v) for s, v in per_seed.items()}, "interval_95": [float(lo), float(hi)], "resamples": n_resamples, "bootstrap_seed": seed, "fraction_above_zero": float(np.mean(np.asarray(draws) > 0))}


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# %% [section] Frozen experiment record and classifier artifact

ARTIFACT_FORMAT = "org.valcorza.bosch-sdi-capstone.linear-head.v1"
ARTIFACT_WEIGHTS = "head.safetensors"
ARTIFACT_MANIFEST = "manifest.json"
FEATURE_DIM = 512
PARITY_TOLERANCE = 1e-5


def freeze_record(path: str | Path, record: Mapping[str, Any], referenced: Mapping[str, str | Path], base: str | Path) -> dict[str, Any]:
    """Write the experiment record with the SHA-256 of every file it depends on and of itself. Nothing downstream
    may change after this point without `verify_frozen` failing."""
    base = Path(base)
    files = {}
    for name, p in sorted(referenced.items()):
        p = Path(p)
        files[name] = {"path": str(p.relative_to(base)), "sha256": sha256_file(p), "bytes": p.stat().st_size}
    body = dict(record)
    body["frozen_files"] = files
    body["record_sha256"] = sha256_bytes(canonical_json(body).encode())
    write_json(path, body)
    return body


def verify_frozen(path: str | Path, base: str | Path) -> dict[str, Any]:
    body = read_json(path)
    claimed = body.pop("record_sha256", None)
    if claimed != sha256_bytes(canonical_json(body).encode()):
        raise ContractError(f"{path}: the frozen experiment record was modified after freezing")
    for name, entry in body["frozen_files"].items():
        p = Path(base) / entry["path"]
        if not p.is_file():
            raise ContractError(f"frozen input {name} ({entry['path']}) is missing")
        if sha256_file(p) != entry["sha256"]:
            raise ContractError(f"frozen input {name} ({entry['path']}) changed after the experiment was frozen")
    body["record_sha256"] = claimed
    return body


def write_artifact(out_dir: str | Path, weight: np.ndarray, bias: np.ndarray, manifest: Mapping[str, Any]) -> dict[str, Any]:
    from safetensors.numpy import save_file

    out = Path(out_dir)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    save_file({"weight": np.ascontiguousarray(weight, dtype=np.float32), "bias": np.ascontiguousarray(bias, dtype=np.float32)}, str(out / ARTIFACT_WEIGHTS))
    body = dict(manifest)
    body["format"] = ARTIFACT_FORMAT
    body["format_version"] = "1.0"
    body["files"] = {ARTIFACT_WEIGHTS: {"sha256": sha256_file(out / ARTIFACT_WEIGHTS), "bytes": (out / ARTIFACT_WEIGHTS).stat().st_size}}
    write_json(out / ARTIFACT_MANIFEST, body)
    return body


def load_artifact(artifact_dir: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reconstruct the head from files only: manifest format, exact file set, byte counts, digests, tensor names,
    shapes, dtype and finiteness are all checked before the tensors are used. safetensors holds plain arrays, so
    nothing is unpickled."""
    from safetensors.numpy import load_file

    root = Path(artifact_dir)
    manifest_path = root / ARTIFACT_MANIFEST
    if not manifest_path.is_file():
        raise ContractError(f"{manifest_path} is missing")
    manifest = read_json(manifest_path)
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ContractError(f"artifact format {manifest.get('format')!r} is not {ARTIFACT_FORMAT!r}")
    present = sorted(p.name for p in root.iterdir())
    expected = sorted([ARTIFACT_MANIFEST, *manifest.get("files", {})])
    if present != expected:
        raise ContractError(f"artifact files {present} differ from the manifest's {expected}")
    for name, entry in manifest["files"].items():
        p = root / name
        if p.stat().st_size != entry["bytes"] or sha256_file(p) != entry["sha256"]:
            raise ContractError(f"{name} does not match its manifest digest; the artifact was modified or truncated")
    tensors = load_file(str(root / ARTIFACT_WEIGHTS))
    if sorted(tensors) != ["bias", "weight"]:
        raise ContractError(f"{ARTIFACT_WEIGHTS} holds {sorted(tensors)}; expected ['bias', 'weight']")
    k = len(manifest["classes"])
    weight, bias = tensors["weight"], tensors["bias"]
    if weight.shape != (k, FEATURE_DIM) or bias.shape != (k,) or weight.dtype != np.float32 or bias.dtype != np.float32:
        raise ContractError(f"head tensors have shapes {weight.shape}/{bias.shape} and dtypes {weight.dtype}/{bias.dtype}; expected ({k}, {FEATURE_DIM})/({k},) float32")
    if not (np.isfinite(weight).all() and np.isfinite(bias).all()):
        raise ContractError("head tensors contain non-finite values")
    return weight, bias, manifest


def head_logits(features: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (np.asarray(features, dtype=np.float32) @ weight.T + bias).astype(np.float32)


# %% [section] Contact sheets


def contact_sheet(items: Sequence[tuple[Image.Image, str]], *, columns: int = 8, thumb: int = 160, title: str = "") -> Image.Image:
    """Grid of labelled thumbnails (each caption drawn under its image) for quick visual inspection."""
    rows = max(1, -(-len(items) // columns))
    caption_h, title_h = 28, (24 if title else 0)
    sheet = Image.new("RGB", (columns * thumb, title_h + rows * (thumb + caption_h)), "white")
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((6, 5), title, fill="black")
    for n, (image, caption) in enumerate(items):
        x, y = (n % columns) * thumb, title_h + (n // columns) * (thumb + caption_h)
        sheet.paste(image.convert("RGB").resize((thumb - 4, thumb - 4), Image.BILINEAR), (x + 2, y + 2))
        for line_no, line in enumerate(caption.split("\n")[:2]):
            draw.text((x + 3, y + thumb + 1 + 12 * line_no), line[:26], fill="black")
    return sheet
