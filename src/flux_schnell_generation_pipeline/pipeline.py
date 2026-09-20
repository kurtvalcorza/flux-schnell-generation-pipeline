"""FLUX.1 [schnell] (`black-forest-labs/FLUX.1-schnell`) DIMER pipeline: a verified snapshot, few-step text-to-image
generation, held-out flow-matching-loss evaluation, and bounded QLoRA fine-tuning of the 4-bit diffusion transformer to
a user's captioned images with a portable adapter.

FLUX.1 [schnell] (Black Forest Labs, 2024) is a 12 B-parameter rectified-flow transformer distilled to generate in one to
four sampling steps without classifier-free guidance. A CLIP-L encoder gives one pooled vector, a T5-XXL encoder gives
256 token embeddings, 19 double-stream and 38 single-stream transformer blocks predict the flow velocity of a 16-channel
latent (the FLUX VAE's 8× downsampled image, packed 2×2 into 64-channel tokens), and the VAE decodes the latent to
pixels. Everything comes from **one pinned snapshot** in the diffusers layout (23 files, 33.7 GB, bfloat16 as shipped):

* the identity of record is the upstream repository `MODEL_ID` at `MODEL_REVISION` (Apache-2.0), whose Hub page is
  gated behind a click-through; the files are **staged from the ungated mirror** `STAGING_ID` at `STAGING_REVISION`,
  whose 23 diffusers-layout files match the upstream tree file for file and byte for byte in size (upstream LFS
  digests are hidden behind the gate; see `docs/WEIGHTS.md` for the state of the byte-identity proof);
* the transformer (23.8 GB) is too large for a 16 GB GPU in any 16-bit format, so it is **always loaded 4-bit**
  (bitsandbytes NF4 with double quantisation; about 6.6 GB) with `COMPUTE_DTYPE` compute — a GPU is required;
* the text encoders (9.7 GB in 16-bit) are loaded only to encode prompts and released before the transformer loads;
* the **scorer** used only by evaluation (`SCORER_ID`, an MIT-licensed CLIP ViT-B/32, 605 MB) is a second snapshot.

Every file is safetensors or plain JSON/text: nothing is unpickled and no Hub-hosted code is executed (the model
classes come from `diffusers` and `transformers` on PyPI).

The adaptation contract is LoRA (rank 8) on the image-stream query/key/value/output projections of all 57 blocks
(380 tensors, 9,338,880 parameters) over the frozen 4-bit base — QLoRA; the VAE and text encoders stay frozen.
Training minimises the rectified-flow velocity MSE on the user's images; the held-out metric is the same MSE at fixed
noise levels and fixed noise, so the frozen and adapted models are compared on identical inputs. Everything
model-related is imported lazily so that snapshot verification and input validation run (and can refuse) before
`torch`, `diffusers`, `transformers` or `bitsandbytes` are imported (fleet RTM-001).
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import math
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
MODEL_REVISION = "741f7c3ce8b383c54771c7003378a50191e9efe9"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "flux1-schnell"
# The upstream repository is gated (click-through); the files are fetched from this unmodified, ungated mirror.
STAGING_ID = "unsloth/FLUX.1-schnell"
STAGING_REVISION = "9df3faa7ae3b6ddf0b2b69bb78616372897cc65c"
ARTIFACT_FORMAT = "org.valcorza.flux-schnell-generation.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"
_WEIGHTS_ROOT = Path(__file__).resolve().parents[2] / "weights"
DEFAULT_WEIGHTS_DIR = _WEIGHTS_ROOT / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
# Evaluation-only scorer (never trained, never part of generation): a CLIP ViT-B/32 served as safetensors.
SCORER_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
SCORER_REVISION = "1a25a446712ba5ee05982a381eed697ef9b435cf"
SCORER_LICENSE = "mit"
SCORER_KEY = "clip-vit-b-32-laion2b"
SCORER_WEIGHTS_DIR = _WEIGHTS_ROOT / SCORER_KEY

# Architecture and contract facts (transformer/config.json and the safetensors headers of the pinned snapshot).
TRANSFORMER_PARAMETERS = 11_891_178_560
TRANSFORMER_TENSORS = 1_156
NUM_DOUBLE_BLOCKS = 19
NUM_SINGLE_BLOCKS = 38
HIDDEN_SIZE = 3072
T5_HIDDEN = 4096
POOLED_DIM = 768
RESOLUTION = 512  # generation and training resolution here (the model was trained at up to 2 MP); centre-cropped
LATENT_CHANNELS = 16
VAE_SCALE = 8
PACKED_CHANNELS = LATENT_CHANNELS * 4  # 2×2 latent patches -> 64-channel tokens
MAX_PROMPT_TOKENS = 256  # the T5 sequence length FLUX.1 [schnell] was distilled with
MAX_CAPTION_CHARS = 1_000
MIN_IMAGE_SIDE = 256
MAX_IMAGE_SIDE = 4_096
MIN_TRAIN_RECORDS = 4
MAX_RECORDS = 2_000
DEFAULT_STEPS = 4
MAX_STEPS = 8  # schnell is distilled for 1-4 steps; more is allowed for inspection only
GUIDANCE_SCALE = 0.0  # guidance-distilled: `guidance_embeds` is false and classifier-free guidance is not used
EVAL_SIGMAS: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)  # fixed noise levels of the held-out flow-matching loss
QUANTIZATION = "nf4"  # bitsandbytes 4-bit NormalFloat, double quantisation; the only way a 12 B transformer fits 16 GB
COMPUTE_DTYPES: tuple[str, ...] = ("float16", "bfloat16")
COMPUTE_DTYPE = "float16"  # default compute/activation dtype; the T4 has no native bfloat16 (see docs/WEIGHTS.md)
LORA_RANK = 8
LORA_ALPHA = 8
LORA_TARGETS: tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0")
LORA_MODULES = 3 * (NUM_DOUBLE_BLOCKS + NUM_SINGLE_BLOCKS) + NUM_DOUBLE_BLOCKS  # single blocks have no to_out.0
LORA_TENSORS = 2 * LORA_MODULES  # 380: (A, B) per module
LORA_PARAMETERS = LORA_MODULES * 2 * LORA_RANK * HIDDEN_SIZE  # 9,338,880
SNAPSHOT_FILE_SUFFIXES = (".safetensors", ".json", ".model", ".txt", ".md")  # the scorer snapshot carries its README


# --------------------------------------------------------------------------------------------------
# manifests and staging (the model snapshot, staged from the mirror, and the scorer snapshot)
# --------------------------------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(root: Path, model_id: str, revision: str) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no snapshot manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("modelId") != model_id:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {model_id!r}")
    if manifest.get("revision") != revision:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {revision!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256_file(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
        if not entry["path"].endswith(SNAPSHOT_FILE_SUFFIXES):
            raise ValueError(f"{entry['path']}: unexpected file type in a code-free snapshot")
    return manifest


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the model snapshot against its DIMER manifest (size + SHA-256 of every listed file). The manifest names
    the upstream identity (`MODEL_ID`@`MODEL_REVISION`) and the mirror the bytes were staged from."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _verify_manifest(root, MODEL_ID, MODEL_REVISION)
    staging = manifest.get("staging", {})
    if (staging.get("repo"), staging.get("revision")) != (STAGING_ID, STAGING_REVISION):
        raise ValueError(f"manifest staging {staging!r} != {STAGING_ID}@{STAGING_REVISION}")
    return manifest


def verify_scorer_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the CLIP scorer snapshot against its own manifest."""
    root = Path(path) if path is not None else SCORER_WEIGHTS_DIR
    return _verify_manifest(root, SCORER_ID, SCORER_REVISION)


def _hub_download(relative_path: str, root: Path, repo_id: str, revision: str) -> None:
    """Fetch one manifest-listed file at the pinned revision straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(repo_id, relative_path, revision=revision, local_dir=str(root))


def _stage_missing(
    root: Path,
    model_id: str,
    revision: str,
    allow_download: bool,
    downloader: Callable[[str, Path], None] | None,
    *,
    source: tuple[str, str] | None = None,
) -> list[str]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != model_id or manifest.get("revision") != revision:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {model_id}@{revision}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; pass allow_download=True to fetch them at {revision}"
        )
    repo_id, repo_revision = source or (model_id, revision)
    fetch = downloader or (lambda rel, dst: _hub_download(rel, dst, repo_id, repo_revision))
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest entries that are absent locally from the ungated mirror `STAGING_ID`@`STAGING_REVISION` (a
    fresh clone commits the manifest and the small JSON/tokenizer files and git-ignores the 33.6 GB of safetensors).
    `verify_snapshot` then checks every byte against the manifest, so which host served a file cannot change what
    is loaded."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    return _stage_missing(
        root, MODEL_ID, MODEL_REVISION, allow_download, downloader, source=(STAGING_ID, STAGING_REVISION)
    )


def stage_missing_scorer_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Same for the CLIP scorer snapshot at SCORER_REVISION."""
    root = Path(path) if path is not None else SCORER_WEIGHTS_DIR
    return _stage_missing(root, SCORER_ID, SCORER_REVISION, allow_download, downloader)


# --------------------------------------------------------------------------------------------------
# captioned-image records and validation (no model import)
# --------------------------------------------------------------------------------------------------

INPUT_SCHEMA: dict[str, Any] = {
    "record": "{id, image, caption}: a PIL image (or a path to one) and the caption used to generate it",
    "image_side": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "resolution": RESOLUTION,
    "preprocessing": (
        f"each image is resized so its shorter side is {RESOLUTION} px and centre-cropped to {RESOLUTION}×{RESOLUTION}; "
        "the crop is reported per record (VAL7). Nothing else is changed"
    ),
    "caption_chars": [1, MAX_CAPTION_CHARS],
    "prompt_tokens": MAX_PROMPT_TOKENS,
    "records": [MIN_TRAIN_RECORDS, MAX_RECORDS],
    "generation": {"steps": [1, MAX_STEPS], "guidance_scale": GUIDANCE_SCALE, "size": RESOLUTION},
    "validation": (
        "record shape, image decodability and side limits, caption length and duplicate ids only. Nothing checks "
        "that a caption describes its image, that the images are photographs, or that the prompt is one the model "
        "can render -- any RGB image with any string is accepted"
    ),
}


def _check_record(record: Any, index: int) -> dict[str, Any]:
    from PIL import Image

    label = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be a mapping with id/image/caption")
    for key in ("id", "image", "caption"):
        if key not in record:
            raise ValueError(f"{label} is missing {key!r}")
    rid, image, caption = record["id"], record["image"], record["caption"]
    if not isinstance(rid, str) or not rid or len(rid) > 64:
        raise ValueError(f"{label}: id must be a non-empty string of at most 64 characters")
    if isinstance(image, str | Path):
        path = Path(image)
        if not path.is_file():
            raise ValueError(f"{label}: image file not found: {path}")
        image = Image.open(path)
        image.load()
    if not isinstance(image, Image.Image):
        raise ValueError(f"{label}: image must be a PIL.Image.Image or a file path")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE or max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"{label}: image sides must be within {MIN_IMAGE_SIDE}..{MAX_IMAGE_SIDE} px, got {image.size}")
    if not isinstance(caption, str) or not caption.strip() or len(caption) > MAX_CAPTION_CHARS:
        raise ValueError(f"{label}: caption must be a non-empty string of at most {MAX_CAPTION_CHARS} characters")
    item = {"id": rid, "image": image.convert("RGB"), "caption": caption.strip()}
    for key in ("label", "common_name", "scientific_name", "observer", "inat_photo_id", "inat_observation_url", "source_id"):
        if key in record:
            item[key] = record[key]
    return item


def image_digest(image: Any) -> str:
    """SHA-256 of the decoded RGB pixels (size + bytes), so a re-encoded copy of the same photo matches."""
    rgb = image.convert("RGB")
    return hashlib.sha256(f"{rgb.size[0]}x{rgb.size[1]}:".encode() + rgb.tobytes()).hexdigest()


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [[r["id"], image_digest(r["image"]), r["caption"]] for r in records]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_dataset(
    records: Sequence[Mapping[str, Any]], *, min_records: int = MIN_TRAIN_RECORDS, max_records: int = MAX_RECORDS
) -> dict[str, Any]:
    """Structural validation of a captioned-image dataset; raises ValueError before any model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ValueError("records must be a list of {id, image, caption} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked = []
    ids: set[str] = set()
    crops = 0
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        width, height = item["image"].size
        if width != height:
            crops += 1
        checked.append(item)
    sides = [min(r["image"].size) for r in checked]
    return {
        "records": checked,
        "n_records": len(checked),
        "n_captions": len({r["caption"] for r in checked}),
        "shorter_side": {"min": min(sides), "max": max(sides)},
        "centre_cropped": crops,
        "resolution": RESOLUTION,
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def validate_inputs(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one record; returns its id, size, the crop it will get and the caption length."""
    item = _check_record(record, 0)
    width, height = item["image"].size
    short = min(width, height)
    scale = RESOLUTION / short
    return {
        "id": item["id"],
        "size": (width, height),
        "resized_to": (round(width * scale), round(height * scale)),
        "centre_crop": (RESOLUTION, RESOLUTION),
        "caption_chars": len(item["caption"]),
    }


def validate_prompts(prompts: Sequence[str]) -> list[str]:
    """Generation prompts: non-empty strings within the caption limit; duplicates are allowed."""
    if isinstance(prompts, str) or not isinstance(prompts, Sequence) or not prompts:
        raise ValueError("prompts must be a non-empty list of strings")
    out = []
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_CAPTION_CHARS:
            raise ValueError(f"prompts[{index}] must be a non-empty string of at most {MAX_CAPTION_CHARS} characters")
        out.append(prompt.strip())
    return out


def preprocess_image(image: Any) -> Any:
    """Resize the shorter side to RESOLUTION and centre-crop; returns a PIL RGB image of RESOLUTION²."""
    from PIL import Image

    rgb = image.convert("RGB")
    width, height = rgb.size
    scale = RESOLUTION / min(width, height)
    new = (max(RESOLUTION, round(width * scale)), max(RESOLUTION, round(height * scale)))
    resized = rgb.resize(new, Image.Resampling.BICUBIC)
    left = (new[0] - RESOLUTION) // 2
    top = (new[1] - RESOLUTION) // 2
    return resized.crop((left, top, left + RESOLUTION, top + RESOLUTION))


# --------------------------------------------------------------------------------------------------
# model construction
# --------------------------------------------------------------------------------------------------


def _preload_nvidia_libs() -> None:
    """Preload the pip-bundled NVIDIA runtime libraries so bitsandbytes resolves libnvJitLink etc. on hosted images
    whose system CUDA is older than the one torch was built against (fleet trap F8)."""
    import ctypes
    import os
    import sys

    for site_pkg in sys.path:
        nvidia_dir = Path(site_pkg) / "nvidia"
        if nvidia_dir.is_dir():
            libs = [str(p) for p in nvidia_dir.glob("*/lib")]
            if libs:
                existing = os.environ.get("LD_LIBRARY_PATH", "")
                prefix = ":".join(libs)
                os.environ["LD_LIBRARY_PATH"] = f"{prefix}:{existing}" if existing else prefix
            for so in nvidia_dir.rglob("lib*.so*"):
                with contextlib.suppress(Exception):
                    ctypes.CDLL(str(so), mode=getattr(ctypes, "RTLD_GLOBAL", 0))


def _torch_dtype(name: str) -> Any:
    import torch

    if name not in COMPUTE_DTYPES:
        raise ValueError(f"compute_dtype must be one of {COMPUTE_DTYPES}")
    return getattr(torch, name)


def _lora_config() -> Any:
    from peft import LoraConfig

    return LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, init_lora_weights="gaussian", target_modules=list(LORA_TARGETS))


def lora_parameter_names(transformer: Any) -> list[str]:
    """The exact tensor set the adaptation contract may change on a transformer built with the adapter."""
    return sorted(name for name, _param in transformer.named_parameters() if ".lora_A." in name or ".lora_B." in name)


def count_parameters(model: Any) -> int:
    """Parameters as the checkpoint counts them: a 4-bit packed tensor counts the elements of its original shape."""
    total = 0
    for param in model.parameters():
        quant_state = getattr(param, "quant_state", None)
        total += math.prod(quant_state.shape) if quant_state is not None else param.numel()
    return total


def build_transformer(weights_dir: Path, *, dtype: Any, use_lora: bool, device: str = "cuda") -> Any:
    """Load the pinned transformer 4-bit (NF4, double quantisation, `dtype` compute) from the verified snapshot onto
    `device`, check the parameter count against the checkpoint, freeze it, and optionally attach the (untrained) LoRA."""
    _preload_nvidia_libs()
    import torch
    from diffusers import BitsAndBytesConfig, FluxTransformer2DModel

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type=QUANTIZATION, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = FluxTransformer2DModel.from_pretrained(
            str(weights_dir), subfolder="transformer", quantization_config=quant, torch_dtype=dtype
        ).to(device)
    n_params = count_parameters(model)
    if n_params != TRANSFORMER_PARAMETERS:
        raise ValueError(f"transformer has {n_params} parameters; expected {TRANSFORMER_PARAMETERS}")
    for param in model.parameters():
        param.requires_grad_(False)
    if use_lora:
        from peft import inject_adapter_in_model

        inject_adapter_in_model(_lora_config(), model, adapter_name="default")
        names = lora_parameter_names(model)
        if len(names) != LORA_TENSORS:
            raise ValueError(f"adapter attached {len(names)} LoRA tensors, expected {LORA_TENSORS}")
        name_set = set(names)
        for name, param in model.named_parameters():
            if name in name_set:
                param.data = param.data.to(torch.float32)  # trained in float32 under autocast
                param.requires_grad_(False)
    model.eval()
    return model


def _select_device(device: str | None) -> str:
    import torch

    if device is None:
        if not torch.cuda.is_available():
            raise ValueError(
                "FLUX.1 [schnell] is loaded 4-bit with bitsandbytes and needs a CUDA GPU; validation, staging and "
                "verification work without one"
            )
        return "cuda"
    if not device.startswith("cuda"):
        raise ValueError("only CUDA devices are supported: the 12 B transformer is loaded 4-bit with bitsandbytes")
    if not torch.cuda.is_available():
        raise ValueError("device='cuda' requested but CUDA is not available")
    return device


def pack_latents(latents: Any) -> Any:
    """(B, 16, H, W) latents -> (B, H/2 · W/2, 64) tokens: each 2×2 latent patch becomes one token."""
    batch, channels, height, width = latents.shape
    x = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    return x.permute(0, 2, 4, 1, 3, 5).reshape(batch, (height // 2) * (width // 2), channels * 4)


def unpack_latents(tokens: Any, height: int, width: int) -> Any:
    """Inverse of `pack_latents` for a (height, width) latent grid."""
    batch = tokens.shape[0]
    x = tokens.view(batch, height // 2, width // 2, LATENT_CHANNELS, 2, 2)
    return x.permute(0, 3, 1, 4, 2, 5).reshape(batch, LATENT_CHANNELS, height, width)


def latent_image_ids(height: int, width: int, device: Any, dtype: Any) -> Any:
    """Rotary position ids of the packed latent tokens: (H/2 · W/2, 3) rows of (0, row, column)."""
    import torch

    ids = torch.zeros(height // 2, width // 2, 3)
    ids[..., 1] += torch.arange(height // 2)[:, None]
    ids[..., 2] += torch.arange(width // 2)[None, :]
    return ids.reshape(-1, 3).to(device=device, dtype=dtype)


@dataclass
class FluxSchnellPipeline:
    """Few-step text-to-image generation and bounded QLoRA fine-tuning on top of the verified FLUX.1 [schnell]
    snapshot. The transformer (4-bit) and the text encoders (16-bit) do not fit a 16 GB GPU together, so prompts
    are encoded first (`encode_prompts`, which loads CLIP-L and T5-XXL) and the transformer is loaded on first use
    by `generate` / `evaluate` / `adapt`, which release the text encoders if they are still resident."""

    vae: Any
    scheduler_config: dict[str, Any]
    tokenizer: Any
    tokenizer_2: Any
    device: str
    dtype: Any
    compute_dtype: str
    weights_dir: Path
    source: str
    use_lora: bool
    transformer: Any = None
    adapter: dict[str, Any] | None = None
    _text_encoder: Any = field(default=None, repr=False)
    _text_encoder_2: Any = field(default=None, repr=False)
    _prompt_cache: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        use_lora: bool = False,
        compute_dtype: str = COMPUTE_DTYPE,
        load_transformer: bool = False,
    ) -> FluxSchnellPipeline:
        """Stage (when allowed) and verify the snapshot, then load the VAE, both tokenizers and the scheduler
        config. The transformer loads lazily (or now, with `load_transformer=True`) and the text encoders only
        inside `encode_prompts`, because 6.6 GB (4-bit transformer) + 9.7 GB (16-bit encoders) exceed 16 GB."""
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        stage_missing_files(root, allow_download=allow_download)
        verify_snapshot(root)
        import torch
        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
        from transformers import AutoTokenizer

        chosen = _select_device(device)
        dtype = _torch_dtype(compute_dtype)
        # The FLUX VAE declares `force_upcast`; it is small (168 MB) and kept in float32 throughout.
        vae = AutoencoderKL.from_pretrained(str(root), subfolder="vae", torch_dtype=torch.float32).to(chosen).eval()
        for param in vae.parameters():
            param.requires_grad_(False)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(root), subfolder="scheduler")
        pipeline = cls(
            vae=vae,
            scheduler_config=dict(scheduler.config),
            tokenizer=AutoTokenizer.from_pretrained(str(root / "tokenizer")),
            tokenizer_2=AutoTokenizer.from_pretrained(str(root / "tokenizer_2")),
            device=chosen,
            dtype=dtype,
            compute_dtype=compute_dtype,
            weights_dir=root,
            source=f"local-snapshot (manifest verified; safetensors only; staged from {STAGING_ID})",
            use_lora=use_lora,
        )
        if load_transformer:
            pipeline.load_transformer()
        return pipeline

    # ---- residency --------------------------------------------------------------------------------------

    def load_transformer(self) -> dict[str, Any]:
        """Load the 4-bit transformer onto the device, releasing the text encoders first if they are resident."""
        released = self.release_text_encoder()
        if self.transformer is not None:
            return {"loaded": False, "released_text_encoders": released}
        started = time.perf_counter()
        self.transformer = build_transformer(self.weights_dir, dtype=self.dtype, use_lora=self.use_lora, device=self.device)
        return {"loaded": True, "released_text_encoders": released, "load_seconds": round(time.perf_counter() - started, 1)}

    def release_transformer(self) -> bool:
        """Drop the transformer (and any adapter state it carries) so the text encoders can be loaded again."""
        import torch

        had = self.transformer is not None
        self.transformer = None
        self.adapter = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return had

    def _require_transformer(self) -> Any:
        if self.transformer is None:
            self.load_transformer()
        return self.transformer

    # ---- prompts ---------------------------------------------------------------------------------------

    def encode_prompts(self, prompts: Sequence[str]) -> dict[str, Any]:
        """Encode every distinct prompt with CLIP-L (pooled vector) and T5-XXL (256 token embeddings) into the
        pipeline's prompt cache. The encoders are loaded on first use in the compute dtype and stay loaded until
        `release_text_encoder` (or until the transformer loads). Refused while the transformer is resident: the
        two do not fit one 16 GB GPU together — encode every prompt first, or `release_transformer()`."""
        import torch
        from transformers import CLIPTextModel, T5EncoderModel

        wanted = [p for p in dict.fromkeys(validate_prompts(list(prompts))) if p not in self._prompt_cache]
        if not wanted:
            return {"encoded": 0, "cached": len(self._prompt_cache)}
        if self.transformer is not None:
            raise ValueError(
                "the transformer is resident; encode all prompts before generate/evaluate/adapt or call "
                "release_transformer() first (text encoders and the 4-bit transformer do not fit 16 GB together)"
            )
        if self._text_encoder is None:
            started = time.perf_counter()
            self._text_encoder = (
                CLIPTextModel.from_pretrained(str(self.weights_dir), subfolder="text_encoder", torch_dtype=self.dtype)
                .to(self.device)
                .eval()
            )
            self._text_encoder_2 = (
                T5EncoderModel.from_pretrained(str(self.weights_dir), subfolder="text_encoder_2", torch_dtype=self.dtype)
                .to(self.device)
                .eval()
            )
            load_seconds = round(time.perf_counter() - started, 1)
        else:
            load_seconds = 0.0
        started = time.perf_counter()
        with torch.inference_mode():
            for prompt in wanted:
                clip_tokens = self.tokenizer(
                    prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
                )
                pooled = self._text_encoder(clip_tokens.input_ids.to(self.device), output_hidden_states=False).pooler_output
                t5_tokens = self.tokenizer_2(
                    prompt,
                    padding="max_length",
                    max_length=MAX_PROMPT_TOKENS,
                    truncation=True,
                    return_length=False,
                    return_overflowing_tokens=False,
                    return_tensors="pt",
                )
                n_tokens = int(t5_tokens.attention_mask.sum())
                embeds = self._text_encoder_2(t5_tokens.input_ids.to(self.device), output_hidden_states=False)[0]
                self._prompt_cache[prompt] = {
                    "embeds": embeds[0].to("cpu", self.dtype),
                    "pooled": pooled[0].to("cpu", self.dtype),
                    "tokens": n_tokens,
                }
        return {
            "encoded": len(wanted),
            "cached": len(self._prompt_cache),
            "encoder_dtype": self.compute_dtype,
            "load_seconds": load_seconds,
            "encode_seconds": round(time.perf_counter() - started, 1),
            "truncated": [p[:40] for p in wanted if self._prompt_cache[p]["tokens"] >= MAX_PROMPT_TOKENS],
        }

    def export_prompt_cache(self) -> dict[str, Any]:
        """The encoded prompts (CPU tensors keyed by prompt), so a second pipeline can reuse them without loading
        the 9.7 GB text encoders again (the notebook's fresh-reload step)."""
        return {
            k: {"embeds": v["embeds"].clone(), "pooled": v["pooled"].clone(), "tokens": v["tokens"]}
            for k, v in self._prompt_cache.items()
        }

    def import_prompt_cache(self, cache: Mapping[str, Mapping[str, Any]]) -> int:
        """Adopt prompt embeddings exported by `export_prompt_cache` from a pipeline of the same identity."""
        for prompt, entry in cache.items():
            if tuple(entry["embeds"].shape) != (MAX_PROMPT_TOKENS, T5_HIDDEN) or tuple(entry["pooled"].shape) != (POOLED_DIM,):
                raise ValueError(f"prompt cache entry for {prompt[:40]!r} has an unexpected shape")
            self._prompt_cache[prompt] = {
                "embeds": entry["embeds"].to(self.dtype),
                "pooled": entry["pooled"].to(self.dtype),
                "tokens": int(entry["tokens"]),
            }
        return len(self._prompt_cache)

    def release_text_encoder(self) -> bool:
        """Drop both text encoders (the cached embeddings remain). Returns whether anything was released."""
        import torch

        had = self._text_encoder is not None or self._text_encoder_2 is not None
        self._text_encoder = None
        self._text_encoder_2 = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return had

    def _embeds(self, prompt: str) -> tuple[Any, Any]:
        if prompt not in self._prompt_cache:
            raise ValueError(f"prompt not encoded; call encode_prompts([...]) first: {prompt[:60]!r}")
        entry = self._prompt_cache[prompt]
        return entry["embeds"].to(self.device, self.dtype), entry["pooled"].to(self.device, self.dtype)

    def _batch_embeds(self, prompts: Sequence[str]) -> tuple[Any, Any]:
        import torch

        pairs = [self._embeds(p) for p in prompts]
        return torch.stack([e for e, _ in pairs]), torch.stack([p for _, p in pairs])

    # ---- generation ------------------------------------------------------------------------------------

    def _diffusers_pipeline(self) -> Any:
        from diffusers import FlowMatchEulerDiscreteScheduler
        from diffusers import FluxPipeline as _Upstream

        return _Upstream(
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(self.scheduler_config),
            vae=self.vae,
            text_encoder=None,
            tokenizer=self.tokenizer,
            text_encoder_2=None,
            tokenizer_2=self.tokenizer_2,
            transformer=self._require_transformer(),
        )

    def generate(self, prompts: Sequence[str], *, seed: int = 0, steps: int = DEFAULT_STEPS) -> dict[str, Any]:
        """Generate one RESOLUTION² image per prompt in `steps` Euler steps without guidance; image i uses seed + i.
        Every prompt must already be encoded (`encode_prompts`)."""
        import numpy as np
        import torch

        prompts = validate_prompts(prompts)
        if not isinstance(steps, int) or not 1 <= steps <= MAX_STEPS:
            raise ValueError(f"steps must be an int in 1..{MAX_STEPS}")
        for prompt in prompts:
            self._embeds(prompt)
        pipe = self._diffusers_pipeline()
        pipe.set_progress_bar_config(disable=True)
        started = time.perf_counter()
        images = []
        for index, prompt in enumerate(prompts):
            embeds, pooled = self._embeds(prompt)
            generator = torch.Generator(device="cpu").manual_seed(seed + index)
            with torch.inference_mode():
                out = pipe(
                    prompt=None,
                    prompt_embeds=embeds[None],
                    pooled_prompt_embeds=pooled[None],
                    num_inference_steps=steps,
                    guidance_scale=GUIDANCE_SCALE,
                    height=RESOLUTION,
                    width=RESOLUTION,
                    max_sequence_length=MAX_PROMPT_TOKENS,
                    generator=generator,
                    output_type="latent",
                )
                image = self._decode(out.images)[0]
            array = np.asarray(image)
            images.append({"prompt": prompt, "seed": seed + index, "image": image, "pixel_mean": round(float(array.mean()), 3)})
        return {
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY, "adapted": self.adapter is not None},
            "steps": steps,
            "guidance_scale": GUIDANCE_SCALE,
            "size": (RESOLUTION, RESOLUTION),
            "scheduler": "FlowMatchEulerDiscreteScheduler (upstream config)",
            "precision": f"{QUANTIZATION} weights, {self.compute_dtype} compute",
            "images": images,
            "seconds": round(time.perf_counter() - started, 2),
        }

    # ---- latents and the flow-matching loss -------------------------------------------------------------

    def _decode(self, tokens: Any) -> list[Any]:
        """Packed latent tokens -> PIL images through the float32 VAE (undoing the shift/scale of `_latents`)."""
        import numpy as np
        import torch
        from PIL import Image

        grid = RESOLUTION // VAE_SCALE
        latents = unpack_latents(tokens.float(), grid, grid) / self.vae.config.scaling_factor + self.vae.config.shift_factor
        pixels = self.vae.decode(latents, return_dict=False)[0]
        arrays = ((pixels.float().clamp(-1, 1) + 1) * 127.5).round().permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
        return [Image.fromarray(np.ascontiguousarray(a)) for a in arrays]

    def _latents(self, records: Sequence[Mapping[str, Any]], *, seed: int) -> Any:
        """VAE-encode preprocessed images to shifted, scaled latents (B, 16, 64, 64); the posterior sample is seeded."""
        import numpy as np
        import torch

        arrays = [np.asarray(preprocess_image(r["image"]), dtype=np.float32) / 127.5 - 1.0 for r in records]
        pixels = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(self.device, torch.float32)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            posterior = self.vae.encode(pixels).latent_dist
            latents = (posterior.sample(generator=generator) - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return latents.to(self.dtype)

    def _autocast(self) -> Any:
        """Mixed precision for the compute dtype (a no-op for the float32 stubs of the offline tests)."""
        import torch

        return torch.autocast(device_type=self.device.split(":")[0], dtype=self.dtype, enabled=self.dtype != torch.float32)

    def _predict_velocity(self, noisy_tokens: Any, sigmas: Any, embeds: Any, pooled: Any) -> Any:
        """One transformer call on packed tokens; `sigmas` in (0, 1) is the model's timestep conditioning."""
        import torch

        transformer = self._require_transformer()
        grid = RESOLUTION // VAE_SCALE
        img_ids = latent_image_ids(grid, grid, self.device, self.dtype)
        txt_ids = torch.zeros(embeds.shape[1], 3, device=self.device, dtype=self.dtype)
        return transformer(
            hidden_states=noisy_tokens,
            timestep=sigmas,
            guidance=None,
            pooled_projections=pooled,
            encoder_hidden_states=embeds,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]

    def evaluate(self, records: Sequence[Mapping[str, Any]], *, seed: int = 0, batch_size: int = 2) -> dict[str, Any]:
        """Held-out flow-matching MSE: every record is VAE-encoded, mixed with a seeded noise tensor at each of
        EVAL_SIGMAS (x_σ = (1 − σ)·x₀ + σ·ε), and the transformer's velocity prediction is scored against the
        rectified-flow target ε − x₀. The same seed gives the same latents, noise and noise levels for the frozen
        and the adapted model, so the numbers are paired."""
        import torch

        checked = validate_dataset(records, min_records=1)["records"]
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 16:
            raise ValueError("batch_size must be an int in 1..16")
        for record in checked:
            self._embeds(record["caption"])
        transformer = self._require_transformer()
        started = time.perf_counter()
        per_sigma: dict[float, list[float]] = {s: [] for s in EVAL_SIGMAS}
        per_record: dict[str, float] = {}
        transformer.eval()
        for start in range(0, len(checked), batch_size):
            batch = checked[start : start + batch_size]
            latents = self._latents(batch, seed=seed + start)
            tokens = pack_latents(latents)
            embeds, pooled = self._batch_embeds([r["caption"] for r in batch])
            record_losses = [0.0] * len(batch)
            for sigma in EVAL_SIGMAS:
                generator = torch.Generator(device="cpu").manual_seed(seed * 1_000 + int(sigma * 1_000) + start)
                noise = torch.randn(tokens.shape, generator=generator).to(self.device, self.dtype)
                sigmas = torch.full((len(batch),), sigma, device=self.device, dtype=self.dtype)
                noisy = ((1.0 - sigma) * tokens.float() + sigma * noise.float()).to(self.dtype)
                with torch.inference_mode(), self._autocast():
                    pred = self._predict_velocity(noisy, sigmas, embeds, pooled)
                target = noise.float() - tokens.float()
                loss = ((pred.float() - target) ** 2).mean(dim=(1, 2))
                for i, value in enumerate(loss.tolist()):
                    per_sigma[sigma].append(value)
                    record_losses[i] += value / len(EVAL_SIGMAS)
            for record, value in zip(batch, record_losses, strict=True):
                per_record[record["id"]] = round(value, 6)
        by_sigma = {str(s): round(sum(v) / len(v), 6) for s, v in per_sigma.items()}
        mean = sum(per_record.values()) / len(per_record)
        return {
            "metric": "flow_matching_mse (velocity-prediction MSE over the packed latent, mean over records and EVAL_SIGMAS)",
            "n_records": len(checked),
            "sigmas": list(EVAL_SIGMAS),
            "seed": seed,
            "flow_matching_mse": round(mean, 6),
            "by_sigma": by_sigma,
            "per_record": per_record,
            "adapted": self.adapter is not None,
            "seconds": round(time.perf_counter() - started, 2),
        }

    # ---- adaptation ------------------------------------------------------------------------------------

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 4,
        lr: float = 1e-4,
        batch_size: int = 1,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded QLoRA fine-tuning on the rectified-flow objective: every step draws one noise level per image
        uniformly from (0, 1) and one noise tensor (both seeded), mixes x_σ = (1 − σ)·x₀ + σ·ε, and minimises the
        MSE between the transformer's velocity prediction and ε − x₀. AdamW at a fixed learning rate on the 380 LoRA
        tensors only, over the frozen 4-bit base, with gradient checkpointing and autocast (loss scaling in
        float16). Epoch 0 records the frozen model; the epoch with the lowest validation loss is kept."""
        if not self.use_lora:
            raise ValueError("adapt() needs a pipeline built with use_lora=True")
        if not isinstance(epochs, int) or not 1 <= epochs <= 50:
            raise ValueError("epochs must be an int in 1..50")
        if not (0.0 < lr <= 1e-2):
            raise ValueError("lr must be in (0, 1e-2]")
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 4:
            raise ValueError("batch_size must be an int in 1..4")
        train_checked = validate_dataset(train)["records"]
        val_checked = validate_dataset(val, min_records=1)["records"] if val is not None else None
        for record in train_checked + (val_checked or []):
            self._embeds(record["caption"])
        import torch

        model = self._require_transformer()
        torch.manual_seed(seed)
        started = time.perf_counter()
        names = lora_parameter_names(model)
        name_set = set(names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in name_set)
        params = [p for n, p in model.named_parameters() if n in name_set]
        n_trainable = sum(p.numel() for p in params)
        if n_trainable != LORA_PARAMETERS:
            raise ValueError(f"{n_trainable} trainable parameters, expected {LORA_PARAMETERS}")
        initial_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        use_scaler = self.dtype == torch.float16
        scaler = torch.amp.GradScaler(self.device.split(":")[0], enabled=use_scaler)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        # Latents are encoded once (the VAE posterior sample is seeded), so epochs differ only in noise and σ.
        tokens_by_id = {}
        for start in range(0, len(train_checked), 4):
            batch = train_checked[start : start + 4]
            encoded = pack_latents(self._latents(batch, seed=seed + 10_000 + start))
            for record, tokens in zip(batch, encoded, strict=True):
                tokens_by_id[record["id"]] = tokens
        model.enable_gradient_checkpointing()
        try:
            history: list[dict[str, Any]] = []
            entry: dict[str, Any] = {"epoch": 0, "train_loss": None, "note": "frozen model (LoRA at initialisation: B = 0)"}
            entry["val_loss"] = self.evaluate(val_checked, seed=seed)["flow_matching_mse"] if val_checked else None
            history.append(entry)
            if progress:
                progress(entry)
            best_val = entry["val_loss"] if entry["val_loss"] is not None else math.inf
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
            best_epoch = 0
            n_steps = 0
            for epoch in range(1, epochs + 1):
                model.train()
                order = torch.randperm(len(train_checked), generator=generator).tolist()
                losses = []
                for start in range(0, len(order), batch_size):
                    batch = [train_checked[i] for i in order[start : start + batch_size]]
                    tokens = torch.stack([tokens_by_id[r["id"]] for r in batch])
                    embeds, pooled = self._batch_embeds([r["caption"] for r in batch])
                    noise = torch.randn(tokens.shape, generator=generator).to(self.device, self.dtype)
                    sigmas = torch.rand((len(batch),), generator=generator).to(self.device, self.dtype)
                    mix = sigmas.float().view(-1, 1, 1)
                    noisy = ((1.0 - mix) * tokens.float() + mix * noise.float()).to(self.dtype)
                    with self._autocast():
                        pred = self._predict_velocity(noisy, sigmas, embeds, pooled)
                    loss = torch.nn.functional.mse_loss(pred.float(), noise.float() - tokens.float())
                    optimiser.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimiser)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(optimiser)
                    scaler.update()
                    losses.append(float(loss.detach()))
                    n_steps += 1
                model.eval()
                entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses)}
                entry["val_loss"] = self.evaluate(val_checked, seed=seed)["flow_matching_mse"] if val_checked else None
                history.append(entry)
                if progress:
                    progress(entry)
                if entry["val_loss"] is None or entry["val_loss"] < best_val:
                    best_val = entry["val_loss"] if entry["val_loss"] is not None else best_val
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
                    best_epoch = epoch
        except BaseException:
            # Transactional: any failure leaves the transformer as it was before adapt() — LoRA restored to its
            # initial values, everything frozen, no adapter attached.
            self._write_lora_state(model, initial_state)
            model.disable_gradient_checkpointing()
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self.adapter = None
            raise
        self._write_lora_state(model, best_state)
        model.disable_gradient_checkpointing()
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "method": "QLoRA (peft LoRA over a bitsandbytes NF4 base)",
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "targets": list(LORA_TARGETS),
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": count_parameters(model),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "lr": lr,
            "batch_size": batch_size,
            "optimizer": "AdamW (weight_decay 0, grad-norm clip 1.0)",
            "precision": (
                f"{QUANTIZATION} base, {self.compute_dtype} autocast"
                + (" + GradScaler" if use_scaler else "")
                + ", float32 LoRA masters"
            ),
            "objective": "rectified-flow velocity MSE, uniform σ in (0, 1)",
            "quantization": QUANTIZATION,
            "compute_dtype": self.compute_dtype,
            "n_train_records": len(train_checked),
            "n_steps": n_steps,
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    @staticmethod
    def _write_lora_state(model: Any, values: Mapping[str, Any]) -> None:
        """Overwrite exactly the LoRA tensors in place (a full `load_state_dict` would try to re-quantise the base)."""
        import torch

        params = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in values.items():
                params[name].copy_(value.to(params[name].device, params[name].dtype))

    # ---- artifacts -------------------------------------------------------------------------------------

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the trained LoRA tensors as safetensors with a manifest."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        tensors = {
            k: v.detach().to("cpu", dtype=v.dtype).contiguous() for k, v in self.transformer.named_parameters() if k in names
        }
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "key": MODEL_KEY,
                "license": MODEL_LICENSE,
                "staging": {"repo": STAGING_ID, "revision": STAGING_REVISION},
                "quantization": QUANTIZATION,
            },
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {"path": ARTIFACT_WEIGHTS_NAME, "bytes": weights_path.stat().st_size, "sha256": _sha256_file(weights_path)}
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return out

    @staticmethod
    def check_artifact_manifest(root: Path, manifest: Mapping[str, Any]) -> Path:
        """Static checks before any weights work: format and version, the pinned base snapshot and quantisation,
        exactly one weights entry named `adapter.safetensors` inside the artifact directory, and the pinned LoRA
        configuration. Returns the weights path."""
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"artifact format_version {manifest.get('format_version')!r} is not supported "
                f"(expected {ARTIFACT_FORMAT_VERSION!r})"
            )
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision")) != (MODEL_ID, MODEL_REVISION):
            raise ValueError("artifact was adapted from a different base model or revision")
        if base.get("quantization") != QUANTIZATION:
            raise ValueError(
                f"artifact was trained over a {base.get('quantization')!r} base, this pipeline uses {QUANTIZATION!r}"
            )
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != 1:
            raise ValueError("artifact manifest must list exactly one weights file")
        entry = files[0]
        if not isinstance(entry, Mapping) or entry.get("path") != ARTIFACT_WEIGHTS_NAME:
            raise ValueError(f"artifact weights file must be named {ARTIFACT_WEIGHTS_NAME!r}")
        weights_path = (root / entry["path"]).resolve()
        if weights_path.parent != root.resolve():
            raise ValueError("artifact weights file must sit inside the artifact directory")
        adapter = manifest.get("adapter")
        declared = None
        if isinstance(adapter, Mapping):
            declared = (adapter.get("rank"), adapter.get("alpha"), list(adapter.get("targets", [])))
        if declared != (LORA_RANK, LORA_ALPHA, list(LORA_TARGETS)):
            raise ValueError(
                f"artifact adapter must declare rank {LORA_RANK}, alpha {LORA_ALPHA} and targets {list(LORA_TARGETS)}"
            )
        if not isinstance(manifest.get("tensors"), list):
            raise ValueError("artifact manifest must list its tensors")
        return weights_path

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest, scope and digest, then overwrite exactly the LoRA tensors."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        weights_path = self.check_artifact_manifest(root, manifest)
        if not self.use_lora:
            raise ValueError("this adapter carries LoRA tensors; build the pipeline with use_lora=True")
        model = self._require_transformer()
        expected = lora_parameter_names(model)
        if sorted(manifest["tensors"]) != expected:
            raise ValueError(f"artifact tensor list does not match the {len(expected)} LoRA tensors of this model")
        entry = manifest["files"][0]
        if _sha256_file(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != expected:
            raise ValueError("artifact tensor names differ from the validated manifest")
        params = dict(model.named_parameters())
        for key, value in tensors.items():
            if tuple(value.shape) != tuple(params[key].shape):
                raise ValueError(f"artifact tensor {key} has shape {tuple(value.shape)}, model has {tuple(params[key].shape)}")
        self._write_lora_state(model, tensors)
        model.eval()
        self.adapter = {**manifest["adapter"], "trainable_names": manifest["tensors"], "history": manifest.get("history", [])}
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        compute_dtype: str = COMPUTE_DTYPE,
        prompt_cache: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> FluxSchnellPipeline:
        """Fresh pipeline with the adapter applied. `prompt_cache` (from `export_prompt_cache`) lets the reload skip
        the text encoders; without it, encode prompts before the first generate/evaluate call."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        cls.check_artifact_manifest(root, manifest)
        pipeline = cls.from_pretrained(
            device=device, weights_dir=weights_dir, allow_download=allow_download, use_lora=True, compute_dtype=compute_dtype
        )
        if prompt_cache:
            pipeline.import_prompt_cache(prompt_cache)
        pipeline.load_artifact(artifact_dir)
        return pipeline
