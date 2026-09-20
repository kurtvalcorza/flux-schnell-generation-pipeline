"""Offline tests for the two snapshot manifests (the model snapshot staged from the mirror, the scorer), staging, the
dataset contract, latent packing and the artifact-manifest rejections. No model library is imported."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from conftest import synthetic_image, synthetic_records
from flux_schnell_generation_pipeline import (
    INPUT_SCHEMA,
    LORA_ALPHA,
    LORA_RANK,
    LORA_TARGETS,
    MAX_IMAGE_SIDE,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    QUANTIZATION,
    RESOLUTION,
    SCORER_REVISION,
    STAGING_ID,
    STAGING_REVISION,
    FluxSchnellPipeline,
    dataset_digest,
    preprocess_image,
    stage_missing_files,
    stage_missing_scorer_files,
    validate_dataset,
    validate_inputs,
    validate_prompts,
    verify_scorer_snapshot,
    verify_snapshot,
)
from flux_schnell_generation_pipeline import pipeline as pl

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_FILE = "transformer/diffusion_pytorch_model-00001-of-00003.safetensors"


def _write_snapshot(root: Path, model_id: str, revision: str, files: dict[str, bytes], *, staging: bool = True) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
        entries.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": "k",
        "modelId": model_id,
        "revision": revision,
        "files": entries,
        "totalBytes": sum(e["bytes"] for e in entries),
    }
    if staging and model_id == MODEL_ID:
        manifest["staging"] = {"repo": STAGING_ID, "revision": STAGING_REVISION}
    (root / pl.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


# --- identity and committed manifests -------------------------------------------------------------------


def test_identity_is_immutable_and_committed_manifests_agree():
    assert len(MODEL_REVISION) == len(STAGING_REVISION) == len(SCORER_REVISION) == 40
    for key, model_id, revision in ((pl.MODEL_KEY, MODEL_ID, MODEL_REVISION), (pl.SCORER_KEY, pl.SCORER_ID, SCORER_REVISION)):
        manifest = json.loads((ROOT / "weights" / key / pl.MANIFEST_NAME).read_text(encoding="utf-8"))
        assert (manifest["modelId"], manifest["revision"], manifest["modelKey"]) == (model_id, revision, key)
        assert manifest["totalBytes"] == sum(e["bytes"] for e in manifest["files"])
        assert all(len(e["sha256"]) == 64 for e in manifest["files"])
        assert not any(e["path"].endswith((".bin", ".pt", ".pth", ".ckpt", ".pickle", ".py")) for e in manifest["files"])
    snapshot = json.loads((ROOT / "weights" / pl.MODEL_KEY / pl.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert snapshot["staging"]["repo"] == STAGING_ID and snapshot["staging"]["revision"] == STAGING_REVISION
    assert snapshot["license"] == pl.MODEL_LICENSE == "apache-2.0"
    paths = {e["path"] for e in snapshot["files"]}
    assert len(paths) == 23 and snapshot["totalBytes"] == 33_725_923_002
    shards = {p for p in paths if p.endswith(".safetensors")}
    assert shards == {
        WEIGHTS_FILE,
        "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
        "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
        "text_encoder/model.safetensors",
        "text_encoder_2/model-00001-of-00002.safetensors",
        "text_encoder_2/model-00002-of-00002.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    }
    # the BFL single-file checkpoints, the mirror's README/NOTICE and the grid image are not part of the snapshot
    assert not any(p.startswith(("flux1-schnell", "ae.", "README", "NOTICE", "schnell_grid")) for p in paths)
    # every small (non-LFS) file is committed so a fresh clone only stages the seven safetensors and spiece.model
    committed = {e["path"] for e in snapshot["files"] if not e["path"].endswith((".safetensors", ".model"))}
    assert all((ROOT / "weights" / pl.MODEL_KEY / p).is_file() for p in committed)
    config = json.loads((ROOT / "weights" / pl.MODEL_KEY / "transformer" / "config.json").read_text(encoding="utf-8"))
    assert (config["num_layers"], config["num_single_layers"], config["guidance_embeds"]) == (19, 38, False)
    assert config["num_attention_heads"] * config["attention_head_dim"] == pl.HIDDEN_SIZE
    scheduler = json.loads((ROOT / "weights" / pl.MODEL_KEY / "scheduler" / "scheduler_config.json").read_text(encoding="utf-8"))
    assert scheduler["_class_name"] == "FlowMatchEulerDiscreteScheduler" and scheduler["shift"] == 1.0
    assert not scheduler["use_dynamic_shifting"]
    vae = json.loads((ROOT / "weights" / pl.MODEL_KEY / "vae" / "config.json").read_text(encoding="utf-8"))
    assert vae["latent_channels"] == pl.LATENT_CHANNELS and vae["force_upcast"] is True
    index_path = ROOT / "weights" / pl.MODEL_KEY / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert len(index["weight_map"]) == pl.TRANSFORMER_TENSORS
    assert pl.LORA_MODULES == 190 and pl.LORA_TENSORS == 380 and pl.LORA_PARAMETERS == 9_338_880


def test_verify_snapshot_requires_the_pinned_staging_source(tmp_path, forbid_model_imports):
    root = tmp_path / "snap"
    _write_snapshot(root, MODEL_ID, MODEL_REVISION, {"a.json": b"{}"}, staging=False)
    with pytest.raises(ValueError, match="staging"):
        verify_snapshot(root)
    manifest = json.loads((root / pl.MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["staging"] = {"repo": STAGING_ID, "revision": "0" * 40}
    (root / pl.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="staging"):
        verify_snapshot(root)


# --- snapshot verification and staging --------------------------------------------------------------------


def test_verify_snapshot_refuses_mismatches(tmp_path, forbid_model_imports):
    root = tmp_path / "snap"
    manifest = _write_snapshot(root, MODEL_ID, MODEL_REVISION, {"transformer/config.json": b"{}", WEIGHTS_FILE: b"tensors"})
    assert verify_snapshot(root)["files"] == manifest["files"]
    (root / WEIGHTS_FILE).write_bytes(b"tensorz")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(root)
    (root / WEIGHTS_FILE).write_bytes(b"tensors-longer")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(root)
    (root / WEIGHTS_FILE).unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        verify_snapshot(root)
    _write_snapshot(root, "someone/else", MODEL_REVISION, {"a.json": b"{}"})
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(root)
    _write_snapshot(root, MODEL_ID, "0" * 40, {"a.json": b"{}"})
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(root)
    with pytest.raises(FileNotFoundError, match="manifest"):
        verify_snapshot(tmp_path / "nowhere")


def test_verify_snapshot_refuses_executable_file_types(tmp_path, forbid_model_imports):
    root = tmp_path / "snap"
    _write_snapshot(root, MODEL_ID, MODEL_REVISION, {"transformer/diffusion_pytorch_model.bin": b"pickle"})
    with pytest.raises(ValueError, match="unexpected file type"):
        verify_snapshot(root)


def test_each_snapshot_has_its_own_identity(tmp_path, forbid_model_imports):
    scorer = tmp_path / "scorer"
    _write_snapshot(scorer, pl.SCORER_ID, SCORER_REVISION, {"config.json": b"{}"})
    assert verify_scorer_snapshot(scorer)["modelId"] == pl.SCORER_ID
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(scorer)
    snap = tmp_path / "snap"
    _write_snapshot(snap, MODEL_ID, MODEL_REVISION, {"vae/config.json": b"{}"})
    with pytest.raises(ValueError, match="modelId"):
        verify_scorer_snapshot(snap)


def test_default_downloader_targets_the_mirror(tmp_path, forbid_model_imports, monkeypatch):
    """`stage_missing_files` fetches from STAGING_ID@STAGING_REVISION, never from the gated upstream repository."""
    root = tmp_path / "snap"
    _write_snapshot(root, MODEL_ID, MODEL_REVISION, {WEIGHTS_FILE: b"tensors"})
    (root / WEIGHTS_FILE).unlink()
    seen = []

    def fake_hub_download(rel, dst, repo_id, revision):
        seen.append((rel, repo_id, revision))
        (dst / rel).write_bytes(b"tensors")

    monkeypatch.setattr(pl, "_hub_download", fake_hub_download)
    assert stage_missing_files(root, allow_download=True) == [WEIGHTS_FILE]
    assert seen == [(WEIGHTS_FILE, STAGING_ID, STAGING_REVISION)]
    scorer = tmp_path / "scorer"
    _write_snapshot(scorer, pl.SCORER_ID, SCORER_REVISION, {"model.safetensors": b"x"})
    (scorer / "model.safetensors").unlink()
    seen.clear()
    stage_missing_scorer_files(scorer, allow_download=True)
    assert seen == [("model.safetensors", pl.SCORER_ID, SCORER_REVISION)]


def test_stage_missing_files_fetches_only_absent_entries(tmp_path, forbid_model_imports):
    root = tmp_path / "snap"
    _write_snapshot(root, MODEL_ID, MODEL_REVISION, {"transformer/config.json": b"{}", WEIGHTS_FILE: b"tensors"})
    (root / WEIGHTS_FILE).unlink()
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(root)
    calls = []

    def downloader(rel, dst):
        calls.append(rel)
        (dst / rel).write_bytes(b"tensors")

    assert stage_missing_files(root, allow_download=True, downloader=downloader) == [WEIGHTS_FILE]
    assert calls == [WEIGHTS_FILE]
    assert stage_missing_files(root, allow_download=True, downloader=downloader) == []
    assert verify_snapshot(root)["revision"] == MODEL_REVISION


def test_stage_refuses_a_manifest_naming_another_model(tmp_path, forbid_model_imports):
    root = tmp_path / "snap"
    _write_snapshot(root, "someone/else", MODEL_REVISION, {"a.json": b"{}"})
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(root, allow_download=True, downloader=lambda rel, dst: None)
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_scorer_files(root, allow_download=True, downloader=lambda rel, dst: None)


# --- dataset contract -------------------------------------------------------------------------------------


def test_validate_dataset_reports_shape_and_digest(forbid_model_imports):
    records = synthetic_records(6)
    report = validate_dataset(records)
    assert report["n_records"] == 6 and report["n_captions"] == 2
    assert report["shorter_side"] == {"min": 480, "max": 480} and report["centre_cropped"] == 6
    assert report["resolution"] == RESOLUTION and len(report["digest"]) == 64
    assert report["digest"] == dataset_digest(records)
    assert INPUT_SCHEMA["image_side"] == [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE]


def test_validate_dataset_refusals_name_the_rule(forbid_model_imports):
    records = synthetic_records(6)
    with pytest.raises(ValueError, match="records must be a list"):
        validate_dataset({"id": "x"})
    with pytest.raises(ValueError, match="4..2000 are required"):
        validate_dataset(records[:3])
    with pytest.raises(ValueError, match="missing 'caption'"):
        validate_dataset([{"id": "a", "image": records[0]["image"]}, *records[1:]])
    with pytest.raises(ValueError, match="caption must be a non-empty string"):
        validate_dataset([{**records[0], "caption": "   "}, *records[1:]])
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset([records[0], *records])
    with pytest.raises(ValueError, match="image sides must be within"):
        validate_dataset([{**records[0], "image": synthetic_image(width=200, height=200)}, *records[1:]])
    with pytest.raises(ValueError, match="image must be a PIL"):
        validate_dataset([{**records[0], "image": b"bytes"}, *records[1:]])
    with pytest.raises(ValueError, match="image file not found"):
        validate_dataset([{**records[0], "image": "nope.jpg"}, *records[1:]])


def test_validate_inputs_reports_the_crop(tmp_path, forbid_model_imports):
    image = synthetic_image(width=800, height=600)
    path = tmp_path / "photo.jpg"
    image.save(path)
    report = validate_inputs({"id": "p", "image": str(path), "caption": "a photo"})
    assert report["size"] == (800, 600) and report["resized_to"] == (683, 512) and report["centre_crop"] == (512, 512)
    out = preprocess_image(image)
    assert out.size == (RESOLUTION, RESOLUTION) and out.mode == "RGB"
    assert preprocess_image(Image.new("RGB", (512, 512), (10, 20, 30))).getpixel((0, 0)) == (10, 20, 30)


def test_validate_prompts(forbid_model_imports):
    assert validate_prompts(["  a bird ", "a bird"]) == ["a bird", "a bird"]
    with pytest.raises(ValueError, match="non-empty list"):
        validate_prompts([])
    with pytest.raises(ValueError, match="prompts\\[1\\]"):
        validate_prompts(["ok", ""])


# --- artifact manifest (static checks, no weights) -------------------------------------------------------


def _good_manifest(root: Path) -> dict:
    (root / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    return {
        "format": pl.ARTIFACT_FORMAT,
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "base_model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "key": pl.MODEL_KEY,
            "quantization": QUANTIZATION,
        },
        "adapter": {"rank": LORA_RANK, "alpha": LORA_ALPHA, "targets": list(LORA_TARGETS)},
        "tensors": ["transformer_blocks.0.attn.to_q.lora_A.default.weight"],
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest()}],
    }


def test_artifact_manifest_static_checks(tmp_path, forbid_model_imports):
    good = _good_manifest(tmp_path)
    assert FluxSchnellPipeline.check_artifact_manifest(tmp_path, good) == (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).resolve()
    cases = {
        "format": ({**good, "format": "other"}, "artifact format"),
        "version": ({**good, "format_version": "2.0"}, "format_version"),
        "base": ({**good, "base_model": {**good["base_model"], "revision": "0" * 40}}, "different base model"),
        "quantization": ({**good, "base_model": {**good["base_model"], "quantization": "fp8"}}, "trained over a 'fp8' base"),
        "two files": ({**good, "files": good["files"] * 2}, "exactly one weights file"),
        "other name": ({**good, "files": [{**good["files"][0], "path": "weights.safetensors"}]}, "must be named"),
        "traversal": ({**good, "files": [{**good["files"][0], "path": "../adapter.safetensors"}]}, "must be named"),
        "rank": ({**good, "adapter": {**good["adapter"], "rank": 16}}, "must declare rank"),
        "targets": ({**good, "adapter": {**good["adapter"], "targets": ["to_q"]}}, "must declare rank"),
        "tensors": ({**good, "tensors": "all"}, "list its tensors"),
    }
    for name, (manifest, message) in cases.items():
        with pytest.raises(ValueError, match=message):
            FluxSchnellPipeline.check_artifact_manifest(tmp_path, manifest)
        del name


# --- latent packing (pure tensor reshapes; torch is imported only here) -----------------------------------


def test_pack_and_unpack_latents_round_trip_and_ids():
    torch = pytest.importorskip("torch")
    latents = torch.arange(2 * 16 * 8 * 8, dtype=torch.float32).reshape(2, 16, 8, 8)
    tokens = pl.pack_latents(latents)
    assert tokens.shape == (2, 16, 64)
    # token 0 holds the 2×2 patch at the top-left corner of every channel, channel-major then row then column
    assert tokens[0, 0, :4].tolist() == [latents[0, 0, 0, 0], latents[0, 0, 0, 1], latents[0, 0, 1, 0], latents[0, 0, 1, 1]]
    assert torch.equal(pl.unpack_latents(tokens, 8, 8), latents)
    ids = pl.latent_image_ids(8, 8, "cpu", torch.float32)
    assert ids.shape == (16, 3) and ids[:, 0].abs().sum() == 0
    assert ids[5].tolist() == [0.0, 1.0, 1.0] and ids[-1].tolist() == [0.0, 3.0, 3.0]
