"""Regression tests for the flux_schnell_generation_colab review fixes (FS-M1..M4, FS-m1..m5).

They run within CI's install budget (pytest, NumPy, Pillow, safetensors; no torch, no model library, no weights). The
notebook's own cells are executed with the carried package functions and stand-ins (a stub pipeline, a NumPy-backed
CLIP scorer, a fake `google.colab`); none of this is model or clean-runtime evidence.
"""
# ruff: noqa: E501  -- test cases quote notebook source lines and refusal messages in full

from __future__ import annotations

import ast
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import types
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

# eo-notebook-test (Windows conda) trap: a NumPy matmul before torch's first import breaks every later torch import in
# the process (WinError 127). This module multiplies NumPy matrices, so torch is imported first when it exists.
with contextlib.suppress(ImportError):
    import torch  # noqa: F401

import flux_schnell_generation_pipeline as package
from conftest import synthetic_image
from flux_schnell_generation_pipeline import (
    MIN_BYOD_IMAGES,
    REAL_PHOTO_REFERENCE_KIND,
    FluxSchnellPipeline,
    duplicate_images,
    load_byod_dataset,
    real_photo_baseline,
    real_photo_reference,
    split_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "flux_schnell_generation_colab.ipynb"
PKG = ROOT / "src" / "flux_schnell_generation_pipeline"


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(f"_fs_{name}", ROOT / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    return "".join(cell["source"])


def _code_cells(nb: dict) -> list[dict]:
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _cell(nb: dict, marker: str) -> str:
    found = [_src(c) for c in _code_cells(nb) if marker in _src(c)]
    assert len(found) == 1, marker
    return found[0]


def _markdown(nb: dict) -> str:
    return "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")


def _functions(source: str, names: set[str], namespace: dict[str, Any]) -> dict[str, Any]:
    """Execute only the named top-level function definitions of a notebook cell in `namespace`."""
    tree = ast.parse(source)
    defs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {d.name for d in defs} == names
    exec(compile(ast.Module(body=defs, type_ignores=[]), "notebook-cell", "exec"), namespace)
    return namespace


# --- FS-M1: isolated runtime, no in-kernel install ----------------------------------------------------------------


def test_exactly_two_kernel_cells_and_a_hash_locked_isolated_install(nb: dict) -> None:
    kernel = [_src(c) for c in _code_cells(nb) if "# dimer: kernel cell" in _src(c)]
    assert len(kernel) == 2
    install, router = kernel
    for needed in ('"--managed-python"', '"--require-hashes"', '"--only-binary"', "UV_SHA256", "LOCK_SHA256", 'platform.machine() != "x86_64"'):
        assert needed in install
    assert "_ip.input_transformers_cleanup.append(_route_to_isolated_runtime)" in router
    build = _load_tool("build_notebook")
    lock = (ROOT / "tutorials" / "requirements-colab.lock.txt").read_text(encoding="utf-8")
    build.check_lock(build._pins(ROOT), lock)  # every pyproject pin is in the lock at the same version, every entry hashed
    for pin in ("torch==2.14.0", "diffusers==0.40.0", "transformers==5.17.0", "peft==0.21.0", "bitsandbytes==0.50.2"):
        assert pin in lock


@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(nb: dict, monkeypatch: pytest.MonkeyPatch, real_google: bool) -> None:
    """find_spec("google.colab") (accelerate does this) must not raise on the worker's stubs (fleet Colab failure)."""
    namespace: dict[str, Any] = {"SKIP_INSTALL": True, "__name__": "__main__"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(_src(_code_cells(nb)[1]), "router", "exec"), namespace)
    worker = namespace["_WORKER_SOURCE"]
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    shim = worker[start : worker.index('_main = types.ModuleType("__main__")', start)]
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    for name in ("google", "google.colab", "google.colab.files"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "google", fake_google if real_google else None)
    monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
    shim_globals = {"os": os, "sys": sys, "types": types, "_send": None, "_recv": None}
    try:
        exec(compile(shim, "worker-colab-shim", "exec"), shim_globals)
        for name in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(name)
            assert spec is not None and spec.name == name
        assert sys.modules["google.colab"].__path__ == [] and callable(sys.modules["google.colab.files"].upload)
    finally:
        for name in ("google.colab", "google.colab.files"):
            sys.modules.pop(name, None)


def test_release_record_no_longer_counts_the_restarted_run_as_one_pass(nb: dict) -> None:
    text = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "an interpreter restart after the install is expected" not in text
    assert text.count("**PASSED in 2 passes — not a one-pass `Run all`**") == 2
    assert "Current status: **Candidate" in (ROOT / "STATUS.md").read_text(encoding="utf-8")
    for name in ("README.md", "STATUS.md", "tutorials/README.md"):
        assert "**Release-grade** —" not in (ROOT / name).read_text(encoding="utf-8"), name
    assert "Restart the runtime, then rerun" not in _markdown(nb)


# --- FS-M2: the real photographs are a leave-one-out reference line, not a ceiling -------------------------------


class _T(np.ndarray):
    """The few torch-tensor methods score_generations / real_photo_reference use, on NumPy."""

    def argmax(self, dim=None, **_):  # type: ignore[override]
        return np.asarray(self).argmax(axis=dim).view(_T)

    def mean(self, dim=None, **_):  # type: ignore[override]
        return np.asarray(self).mean(axis=dim).view(_T)

    def norm(self):
        return float(np.linalg.norm(np.asarray(self)))


class _Scorer:
    """CLIP stand-in: fixed unit vectors per image key and per caption."""

    identity = {"id": "stand-in", "revision": "0"}

    def __init__(self, images: dict[str, np.ndarray], texts: dict[str, np.ndarray]) -> None:
        self.images, self.texts = images, texts

    @staticmethod
    def _stack(rows):
        return np.stack([r / np.linalg.norm(r) for r in rows]).astype(np.float64).view(_T)

    def image_embeddings(self, images):
        return self._stack([self.images[k] for k in images])

    def text_embeddings(self, texts):
        return self._stack([self.texts[t] for t in texts])


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def test_real_photo_reference_excludes_each_photo_from_its_own_reference() -> None:
    a, b, c = _unit([1, 0.2, 0, 0]), _unit([0.6, 0.8, 0.1, 0]), _unit([0, 0, 1, 0.3])
    scorer = _Scorer({"a": a, "b": b, "c": c}, {"A": _unit([1, 0.5, 0, 0]), "C": _unit([0, 0, 1, 0])})
    records = [{"image": "a", "caption": "A"}, {"image": "b", "caption": "A"}, {"image": "c", "caption": "C"}]
    report = real_photo_reference(scorer, records)
    cos_ab = float(a @ b) * 100
    rows = report["per_image"]
    # two photos of a caption: each one's reference similarity is cos(a, b), not sqrt((1 + cos(a, b)) / 2)
    assert rows[0]["reference_similarity"] == rows[1]["reference_similarity"] == round(cos_ab, 3)
    assert rows[0]["reference_similarity"] != round(np.sqrt((1 + cos_ab / 100) / 2) * 100, 3)
    assert "reference_similarity" not in rows[2] and report["n_without_reference"] == 1
    assert report["reference_similarity"] == round(cos_ab, 3)
    assert report["reference_kind"] == REAL_PHOTO_REFERENCE_KIND == "leave-one-out real-photo reference"
    assert set(report["reading"]) == {"clip_prompt_similarity", "label_accuracy", "reference_similarity"}
    assert "not a ceiling" in report["note"] and "excludes the photo itself" in report["note"]
    assert real_photo_baseline is real_photo_reference  # the old name gives the corrected measure


def test_recorded_two_photo_values_reproduce_the_reviews_leave_one_out_mean() -> None:
    """With two photos per caption the self-inclusive value s and the leave-one-out cos(a, b) satisfy cos = 2 s² − 1;
    the 2026-09-20 record's six self-inclusive values give the review's 57.7."""
    recorded = [95.419, 95.779, 82.961, 86.968, 85.623, 85.212]
    loo = [(2 * (s / 100) ** 2 - 1) * 100 for s in recorded]
    assert round(sum(loo) / len(loo), 1) == 57.7
    a, b = _unit([1, 0.3, 0.2]), _unit([0.4, 1, 0])
    mean = (a + b) / np.linalg.norm(a + b)
    assert abs(float(a @ mean) - np.sqrt((1 + float(a @ b)) / 2)) < 1e-12


def test_no_learner_text_calls_the_real_photographs_a_ceiling(nb: dict) -> None:
    md = _markdown(nb)
    for phrase in ("real-photo ceiling", "the ceiling these numbers could reach", "the ceiling."):
        assert phrase not in md
    assert md.count("not a ceiling") >= 4 and "leave-one-out" in md
    code = "\n".join(_src(c) for c in _code_cells(nb) if not c["metadata"].get("dimer"))
    assert "real_reference = real_photo_reference(scorer, test_records)" in code
    assert "real_photo_baseline" not in code and "real_ceiling" not in code
    for name in ("README.md", "MODEL_CARD.md", "tutorials/README.md", "STATUS.md"):
        assert "real-photo ceiling" not in (ROOT / name).read_text(encoding="utf-8"), name


# --- FS-M3: nothing stays resident for a BYOD re-run -------------------------------------------------------------


class _StubPipe:
    def __init__(self, *, transformer=None, adapter=None) -> None:
        self.transformer, self.adapter, self.released = transformer, adapter, 0

    def release_transformer(self) -> bool:
        had = self.transformer is not None
        self.transformer, self.adapter = None, None
        self.released += 1
        return had


def _torch_stub() -> types.SimpleNamespace:
    return types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None, memory_allocated=lambda: 0))


def test_section_5_releases_every_gpu_resident_before_the_encoders_load(nb: dict) -> None:
    source = _cell(nb, "encode_report = pipe.encode_prompts(all_prompts)")
    assert source.index("released_before_encoding = release_gpu_residents()") < source.index("pipe.encode_prompts(all_prompts)")
    import gc

    pipe = _StubPipe(transformer="adapted")
    reloaded = _StubPipe(transformer="reloaded 4-bit")
    namespace: dict[str, Any] = {"pipe": pipe, "reloaded": reloaded, "scorer": object(), "torch": _torch_stub(), "gc": gc, "FluxSchnellPipeline": _StubPipe}
    _functions(source, {"gpu_memory_gb", "release_gpu_residents", "reset_to_pretrained"}, namespace)
    released = namespace["release_gpu_residents"]()
    assert released == {"pipe.transformer": True, "reloaded": True, "scorer": True}
    assert pipe.transformer is None and reloaded.transformer is None
    assert "reloaded" not in namespace and "scorer" not in namespace and namespace["pipe"] is pipe
    assert namespace["release_gpu_residents"]() == {}  # a second call finds nothing


def test_section_9_releases_the_reloaded_pipeline_after_the_parity_check(nb: dict) -> None:
    source = _cell(nb, "reloaded = FluxSchnellPipeline.from_artifact(")
    parity = source.index("assert parity['flow_matching_mse_diff'] < 1e-5")
    assert parity < source.index("reloaded.release_transformer()") < source.index("del reloaded") < source.index("result_payload = {")
    assert "reloaded_best_epoch = reloaded.adapter['best_epoch']" in source


def test_model_loads_stream_to_the_gpu() -> None:
    """Every GPU model load in the package passes device_map, so a 12.7 GiB Colab VM never holds a converted model
    (the capstone's verified host-RAM fix, 0cef303)."""
    tree = ast.parse((PKG / "pipeline.py").read_text(encoding="utf-8"))
    loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {"FluxTransformer2DModel", "AutoencoderKL", "CLIPTextModel", "T5EncoderModel"}
    ]
    assert len(loads) == 4
    assert all(any(k.arg == "device_map" for k in node.keywords) for node in loads)


# --- FS-M4: guided layer and infrastructure labels ------------------------------------------------------------------


def test_guided_layer_and_collapsed_infrastructure(nb: dict) -> None:
    md = _markdown(nb)
    for marker, least in (
        ("**Who this is for.**", 1),
        ("**Input → Model → Output.**", 1),
        ("**How to use this notebook.**", 1),
        ("**Roadmap:**", 1),
        ("**Predict before running:**", 6),
        ("**What to notice (Section", 7),
        ("<summary>Check your reasoning</summary>", 7),
        ("**Predict → Change one thing → Run → Observe → Explain.**", 1),
        ("## Troubleshooting", 1),
        ("## Glossary", 1),
        ("## Conclusion (your notes)", 1),
        ("> **Infrastructure.**", 3),
    ):
        assert md.count(marker) >= least, marker
    titled = [c for c in _code_cells(nb) if _src(c).startswith("# @title Infrastructure:")]
    assert len(titled) == 7 and all(c["metadata"].get("cellView") == "form" for c in titled)
    activity = _cell(nb, "RUN_ACTIVITY = False  # @param")
    assert "outputs/activity" in activity and "reset_to_pretrained()" in activity
    assert "{{" not in md and "}}" not in md


# --- FS-m1: BYOD contract -------------------------------------------------------------------------------------------


def _zip(path: Path, rows: list[tuple[str, str, str]], files: dict[str, bytes], *, table_at: str = "captions.csv") -> Path:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "file", "caption"])
    writer.writerows(rows)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(table_at, buffer.getvalue())
        for name, data in files.items():
            archive.writestr(name, data)
    return path


def _jpeg(seed: int) -> bytes:
    out = io.BytesIO()
    synthetic_image(width=320, height=300, seed=seed).save(out, format="PNG")
    return out.getvalue()


def _one_caption_zip(tmp_path: Path, n: int, *, prefix: str = "images/", name: str = "byod.zip") -> Path:
    files = {f"{prefix}img{i}.png": _jpeg(i) for i in range(n)}
    rows = [(f"r{i}", f"{prefix}img{i}.png", "a photo of my bird") for i in range(n)]
    return _zip(tmp_path / name, rows, files)


def test_stated_minimum_is_accepted_and_one_less_is_refused_with_the_rule(tmp_path: Path) -> None:
    assert MIN_BYOD_IMAGES == 6
    splits = split_dataset(load_byod_dataset(_one_caption_zip(tmp_path, 6)), seed=0)
    assert {k: len(v) for k, v in splits.items()} == {"test": 1, "validation": 1, "train": 4}
    with pytest.raises(ValueError, match=r"split leaves 3 training records .* at least 6 distinct images remain, for example six of one caption"):
        split_dataset(load_byod_dataset(_one_caption_zip(tmp_path, 5, name="five.zip")), seed=0)
    one_each = [{"id": f"u{i}", "image": synthetic_image(seed=40 + i), "caption": f"caption {i}"} for i in range(6)]
    with pytest.raises(ValueError, match="no test record .* three or more images"):
        split_dataset(one_each, seed=0)


def test_subfolder_paths_folder_zips_and_unique_file_names_resolve(tmp_path: Path) -> None:
    assert len(load_byod_dataset(_one_caption_zip(tmp_path, 6))) == 6  # images/imgN.png as listed
    files = {f"dataset/photos/img{i}.png": _jpeg(i) for i in range(6)}
    nested = _zip(tmp_path / "nested.zip", [(f"r{i}", f"photos/img{i}.png", "c") for i in range(6)], files, table_at="dataset/captions.csv")
    assert len(load_byod_dataset(nested)) == 6  # relative to the folder holding captions.csv
    by_name = _zip(tmp_path / "names.zip", [(f"r{i}", f"img{i}.png", "c") for i in range(6)], files, table_at="dataset/captions.csv")
    assert len(load_byod_dataset(by_name)) == 6  # unique file names


@pytest.mark.parametrize(
    ("rows", "files", "message"),
    [
        ([("r0", "missing.png", "c")], {}, "captions.csv file 'missing.png' is not in the zip"),
        ([("r0", "notes.png", "c")], {"notes.png": b"not an image"}, "notes.png: not a readable JPEG or PNG image"),
        ([("r0", "../escape.png", "c")], {}, "give a path inside the zip"),
        ([("r0", "/abs.png", "c")], {}, "give a path inside the zip"),
    ],
)
def test_byod_refusals_name_the_file_and_the_fix(tmp_path: Path, rows, files, message) -> None:
    with pytest.raises(ValueError, match=message.replace(".", r"\.").replace("(", r"\(")):
        load_byod_dataset(_zip(tmp_path / "bad.zip", rows, files))


def test_not_a_zip_and_missing_table_are_named(tmp_path: Path) -> None:
    junk = tmp_path / "photos.zip"
    junk.write_bytes(b"plain text, not a zip")
    with pytest.raises(ValueError, match="photos.zip is not a zip archive"):
        load_byod_dataset(junk)
    with zipfile.ZipFile(tmp_path / "empty.zip", "w") as archive:
        archive.writestr("img.png", _jpeg(0))
    with pytest.raises(ValueError, match="exactly one captions.csv"):
        load_byod_dataset(tmp_path / "empty.zip")


def test_duplicates_are_reported_by_id_and_dropped_by_the_split() -> None:
    records = [{"id": f"r{i}", "image": synthetic_image(seed=i % 6), "caption": "c"} for i in range(9)]
    assert duplicate_images(records) == [{"dropped": "r6", "kept": "r0"}, {"dropped": "r7", "kept": "r1"}, {"dropped": "r8", "kept": "r2"}]
    assert sum(len(v) for v in split_dataset(records, seed=0).values()) == 6


def _section_4(nb: dict, *, use_byod: bool, byod_path: str = "") -> str:
    source = _cell(nb, "USE_BYOD = False  # @param")
    source = source.replace("USE_BYOD = False  # @param", f"USE_BYOD = {use_byod}  # @param")
    return source.replace("BYOD_PATH = ''  # @param", f"BYOD_PATH = {byod_path!r}  # @param")


def _package_namespace() -> dict[str, Any]:
    return {name: getattr(package, name) for name in package.__all__} | {"__name__": "__main__"}


def test_section_4_cancelled_upload_is_named(nb: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    colab = types.ModuleType("google.colab")
    colab.__spec__ = importlib.machinery.ModuleSpec("google.colab", None, is_package=True)
    colab.__path__ = []
    colab.files = types.SimpleNamespace(upload=lambda: {})
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(ValueError, match=r"Upload exactly one \.zip .* \(got 0 files; a cancelled upload gives 0\)"):
        exec(compile(_section_4(nb, use_byod=True), "section-4", "exec"), _package_namespace())


def test_section_4_byod_path_runs_the_cell_and_writes_byod_captions(nb: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    archive = _one_caption_zip(tmp_path, 7)
    with zipfile.ZipFile(archive, "a") as handle:  # one pixel-identical copy under another name
        handle.writestr("images/copy.png", _jpeg(0))
    rows = list(csv.reader(io.StringIO(zipfile.ZipFile(archive).read("captions.csv").decode())))
    monkeypatch.chdir(tmp_path)
    namespace = _package_namespace()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(_section_4(nb, use_byod=True, byod_path=str(archive)), "section-4", "exec"), namespace)
    assert len(rows) == 8
    assert namespace["data_source"] == "BYOD (byod.zip)"
    assert namespace["captions_csv"] == "outputs/flux_schnell_generation_byod_captions.csv" and Path(namespace["captions_csv"]).is_file()
    assert {k: len(namespace[k]) for k in ("train_records", "val_records", "test_records")} == {"train_records": 5, "val_records": 1, "test_records": 1}
    assert "'byod_records': 7" in out.getvalue() and "'duplicates_dropped': 0" in out.getvalue()


def test_section_6_caps_and_warns_on_many_generation_prompts(nb: dict) -> None:
    source = _cell(nb, "MAX_GENERATION_PROMPTS = 12  # @param")
    assert "if len(prompts) > MAX_GENERATION_PROMPTS:" in source and "'warning':" in source
    assert "generation_prompts = [p for p in prompts[:MAX_GENERATION_PROMPTS] for _ in range(IMAGES_PER_PROMPT)]" in source
    assert "stratified within each caption" in _markdown(nb) and "split by caption" not in _markdown(nb)


# --- FS-m2: the exported revision contains the carried modules ------------------------------------------------------


def test_notebook_source_carries_per_module_digests(nb: dict) -> None:
    generated = nb["metadata"]["dimer"]["generated_from"]
    expected = {f"src/flux_schnell_generation_pipeline/{m}": hashlib.sha256((PKG / m).read_text(encoding="utf-8").encode("utf-8")).hexdigest() for m in ("pipeline.py", "samples.py", "metrics.py")}
    assert generated["module_sha256_per_file"] == expected
    runtime = _cell(nb, "NOTEBOOK_SOURCE = {")
    carried = re.search(r"'module_sha256_per_file': (\{[^}]*\})", runtime)
    assert carried and ast.literal_eval(carried.group(1)) == expected
    assert not generated["revision"].endswith("+working-tree")


def test_generator_labels_a_dirty_module_tree_as_working_tree(tmp_path: Path) -> None:
    git = subprocess.run(["git", "--version"], capture_output=True, check=False)
    if git.returncode:
        pytest.skip("git is not available")
    build = _load_tool("build_notebook")
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, env=env)
    (tmp_path / "m.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "m.py"], check=True, env=env)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "c"], check=True, env=env)
    head = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    assert build._head_revision(tmp_path, {"m.py": "x = 1\n"}) == head
    assert build._head_revision(tmp_path, {"m.py": "x = 2\n"}) == head + build.WORKING_TREE_SUFFIX


def test_exported_revision_reproduces_the_carried_module_digests(nb: dict) -> None:
    """The review's acceptance check, where the labelled commit is in the local history (CI's shallow checkout skips)."""
    generated = nb["metadata"]["dimer"]["generated_from"]
    revision = generated["revision"]
    for rel, digest in generated["module_sha256_per_file"].items():
        shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{revision}:{rel}"], capture_output=True, check=False)
        if shown.returncode:
            pytest.skip(f"revision {revision[:12]} is not in this checkout's history")
        text = shown.stdout.decode("utf-8").replace("\r\n", "\n")
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == digest, rel


# --- FS-m3: experiments never score an adapted model as frozen --------------------------------------------------------


def test_adapt_refuses_to_continue_from_a_trained_adapter() -> None:
    pipe = FluxSchnellPipeline(
        vae=None, scheduler_config={}, tokenizer=None, tokenizer_2=None, device="cuda", dtype=None,
        compute_dtype="float16", weights_dir=Path("."), source="stub", use_lora=True,
    )
    pipe.adapter = {"best_epoch": 3}
    with pytest.raises(ValueError, match="already carries a trained adapter.*release_transformer"):
        pipe.adapt([], [])


def test_sections_6_7_and_10_reset_to_the_pretrained_base_and_8_9_require_an_adapter(nb: dict) -> None:
    for marker, call in (
        ("load_report = pipe.load_transformer()", "load_report = pipe.load_transformer()"),
        ("adapt_result = pipe.adapt(", "adapt_result = pipe.adapt("),
        ("RUN_ACTIVITY = False  # @param", "activity_load = pipe.load_transformer()"),
    ):
        source = _cell(nb, marker)
        assert source.index("reset_to_pretrained()") < source.index(call), marker
    for marker in ("adapted_val = pipe.evaluate(", "pipe.save_artifact(artifact_dir"):
        source = _cell(nb, marker)
        assert source.lstrip().startswith(("if pipe.adapter is None:", "import platform")) and "if pipe.adapter is None:\n    raise RuntimeError(" in source


def test_reset_to_pretrained_releases_only_an_adapted_transformer(nb: dict) -> None:
    source = _cell(nb, "encode_report = pipe.encode_prompts(all_prompts)")
    frozen = _StubPipe(transformer="frozen base")
    namespace: dict[str, Any] = {"pipe": frozen}
    _functions(source, {"reset_to_pretrained"}, namespace)
    namespace["reset_to_pretrained"]()
    assert frozen.released == 0 and frozen.transformer == "frozen base"
    adapted = _StubPipe(transformer="adapted", adapter={"best_epoch": 3})
    namespace["pipe"] = adapted
    with contextlib.redirect_stdout(io.StringIO()):
        namespace["reset_to_pretrained"]()
    assert adapted.released == 1 and adapted.transformer is None and adapted.adapter is None


# --- FS-m4 / FS-m5: expectations, overlap and variability statements; images shown inline ---------------------------


def test_expectations_match_the_record_and_the_conclusion_is_conditional(nb: dict) -> None:
    md = _markdown(nb)
    assert "label accuracy at or below the real photographs'" in md
    assert "below the real photographs' 1.0" not in md and "about an hour" not in md
    assert "Read your own Section 8 before concluding" in md
    assert "**Pretraining overlap.**" in md and "**Run-to-run variability.**" in md
    assert "Chipping Sparrow 70.24 → 68.58" in md


def test_generated_images_are_displayed_inline(nb: dict) -> None:
    assert "display(Image.open(frozen_grid))" in _cell(nb, "frozen_generation = pipe.generate(")
    assert "display(Image.open(pair_grid))" in _cell(nb, "adapted_generation = pipe.generate(")
    assert "display(Image.open(grid([g['image'] for g in new_generation['images']]" in _cell(nb, "new_generation = pipe.generate(")
    assert Image is not None  # PIL is what the notebook displays with
