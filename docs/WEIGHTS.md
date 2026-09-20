# Weight provenance, the three pinned snapshots and DIMER hosting

This repository pins **three** Hugging Face snapshots, each with its own `dimer-base-manifest.json` (format `dimer_hf_snapshot` v1: byte size and SHA-256 per file, the immutable revision, the licence) and each staged and verified separately by `src/pixart_sigma_generation_pipeline/pipeline.py`. Every weight file is safetensors; nothing is unpickled, and the model classes come from `diffusers`, `transformers` and `peft` on PyPI — no Hub-hosted code is executed.

## 1. The transformer — PixArt-Σ XL-2 512-MS

- Upstream: `PixArt-alpha/PixArt-Sigma-XL-2-512-MS`
- Immutable revision: `76fb7eb5a9314bc1e4e479d2f13447517fca9be4` (2024-05-16, "Delete asset/asset_model.png"); the weights were first published in commit `1429642d` (2024-04-11, "Upload PixArt-Sigma-512px diffusers") and the safetensors file's LFS digest is identical at every later revision (checked 2026-09-19 with `HfApi.model_info(files_metadata=True)` at `1429642d`, `786c445c` and the pinned revision).
- Upstream weight license: **CreativeML Open RAIL++-M** (`license: openrail++` in the pinned README front matter). The licence permits use, redistribution and commercial use subject to its use-based restrictions (Attachment A), which must travel with any redistribution — including a DIMER profile — and with any adapter trained on the weights.
- Local layout: `weights/pixart-sigma-xl-2-512-ms/` holds the 3 manifest entries — upstream `README.md` (5,543 bytes), `transformer/config.json` (746 bytes) and `transformer/diffusion_pytorch_model.safetensors` (2,443,492,488 bytes, SHA-256 `20f3328e…`); 2,443,498,777 bytes in total. `verify_snapshot()` checks every entry by byte size and SHA-256 and refuses on the first mismatch; `stage_missing_files(allow_download=True)` fetches only absent entries, only at the pinned revision.
- Architecture (`transformer/config.json`): `Transformer2DModel` in the `ada_norm_single` (PixArt) configuration — 28 layers, 16 heads × 72, hidden size 1,152, patch size 2 over a 64 × 64 × 4 latent, cross-attention to 4,096-dimensional T5 caption embeddings, 8 output channels (4 noise + 4 learned variance). Loaded as `diffusers.PixArtTransformer2DModel`: 603 tensors, 610,856,096 parameters, float32 as shipped, run in float16 on CUDA. Files deliberately not staged: the `asset/` images.

## 2. The components — T5-XXL encoder, tokenizer, SDXL VAE, scheduler

- Upstream: `PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers` — the components the PixArt authors publish once and every Σ checkpoint composes with (`model_index.json` names the Σ pipeline with a `text_encoder`, `tokenizer`, `vae`, `scheduler` and a `transformer` slot).
- Immutable revision: `2c17b4e85261cd549b4068d086b7c2ba9d468e9f` (2024-04-07, "Upload config.json").
- Upstream license: `license: openrail` in the pinned README front matter (the repository carries the T5 encoder and the SDXL VAE, whose own upstream licences are Apache-2.0 and MIT respectively; the pinned README's declaration is the one this repository records).
- Local layout: `weights/pixart-sigma-t5-vae/` holds the 12 manifest entries (19,384,731,509 bytes): `README.md`, `model_index.json`, `scheduler/scheduler_config.json` (DPM-Solver++ 2nd order, linear betas 1e-4..0.02, 1,000 training timesteps, ε-prediction), `text_encoder/{config.json, model.safetensors.index.json, model-00001-of-00002.safetensors (9,989,150,328 bytes), model-00002-of-00002.safetensors (9,060,119,392 bytes)}` (T5-XXL encoder: 24 layers, d_model 4,096, float32 as shipped — 4.76 B parameters, loaded in float16 on CUDA), `tokenizer/{special_tokens_map.json, spiece.model, tokenizer_config.json}`, `vae/{config.json, diffusion_pytorch_model.safetensors (334,643,238 bytes)}` (the SDXL VAE, `force_upcast: true`, kept in float32). `verify_base_snapshot()` and `stage_missing_base_files()` mirror the transformer's functions.
- The 19 GB float32 encoder shards dominate every download; the pipeline never rewrites them. The encoder is loaded lazily by `encode_prompts` and dropped by `release_text_encoder`, because its 9.5 GB (float16) does not fit beside a training graph on a 16 GB GPU; encoded prompts (300 × 4,096 float16 per prompt) are cached and can be exported to a second pipeline.

## 3. The scorer — CLIP ViT-B/32 (evaluation only)

- Upstream: `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`, revision `1a25a446712ba5ee05982a381eed697ef9b435cf` (2025-01-22), MIT licence.
- Local layout: `weights/clip-vit-b-32-laion2b/`, 9 manifest entries (608,782,299 bytes): `model.safetensors` (605,157,884 bytes) plus the config, tokenizer and preprocessor files. Loaded as `transformers.CLIPModel` in float32, frozen; used by `metrics.ClipScorer` only — never part of generation, never trained, and its scores are not a human judgement of image quality.

## Fidelity

No upstream regression fixture is published for the transformer. The evidence is the strict load of all 603 tensors into `PixArtTransformer2DModel` from the pinned `config.json`, the parameter count asserted by `from_pretrained`, and generation that CLIP recognises: on the build run the frozen model's 512² renderings of the six bird captions are classified among the six captions by CLIP with the accuracy recorded in `MODEL_CARD.md` (*Runtime*). The `use_resolution_binning=False` generation path at exactly 512 × 512 is the only one exercised; other aspect ratios are not.

## Runtime facts

- Precision: transformer float16 on CUDA (float32 on CPU, unsupported for the tutorial), LoRA parameters float32 with float16 autocast and loss scaling, VAE float32, text encoder float16 on CUDA.
- Generation: 20 DPM-Solver++ steps at guidance 4.5 by default (`MAX_STEPS = 100`, `MAX_GUIDANCE = 20`), one seeded CPU generator per image (image *i* uses `seed + i`), prompt embeddings taken from the cache (`text_encoder=None` in the upstream pipeline).
- Evaluation: the held-out denoising MSE re-encodes each photograph with the VAE (seeded posterior sample), noises it at timesteps 100 / 300 / 500 / 700 / 900 with seeded noise, and scores the transformer's noise prediction; the same seed gives identical inputs for the frozen and the adapted model.
- Adaptation: LoRA rank 8 / alpha 8 on `to_q`, `to_k`, `to_v`, `to_out.0` of both attentions of all 28 blocks — 448 tensors, 4,128,768 parameters (0.68 % of the transformer) — injected with `peft.inject_adapter_in_model`; AdamW, grad-norm clip 1.0, uniform timesteps, batch size 1, epoch selected by validation MSE with the frozen model as epoch 0 (LoRA B = 0 at initialisation, so epoch 0 *is* the frozen model); transactional on failure.
- `diffusers`, `transformers`, `peft`, `accelerate`, `sentencepiece` (for the T5 tokenizer) and `protobuf` are the runtime pins beyond torch; the package imports all of them lazily so manifest verification and input validation run first (fleet RTM-001).

## The tutorial data: pinned iNaturalist CC0 photographs

`samples.py` fetches 60 research-grade iNaturalist photographs of six North American birds (Song Sparrow, Chipping Sparrow, White-throated Sparrow, Dark-eyed Junco, House Finch, American Goldfinch — 10 per species, one per observer per species) from the public open-data bucket `https://inaturalist-open-data.s3.amazonaws.com/photos/<photo id>/medium.<ext>`, 5,789,324 bytes in all, each pinned by byte size and SHA-256 in `SAMPLE_RECORDS` and refused on a mismatch before decoding; every photo is CC0 1.0 and its observation page and observer are recorded. Captions come from one template per species. `build_sample_dataset` draws a seeded 6 / 2 / 2 stratified split per species (36 / 12 / 12); the cache is `weights/inat-birds/` (git-ignored).

## DIMER hosting

- **Licence gate:** the transformer is Open RAIL++-M, not a permissive licence. Hosting it in the DIMER model store, and any adapter trained on it, requires that the licence text and its use-based restrictions accompany the profile; whether DIMER's catalogue can carry a RAIL-licensed profile is a policy decision recorded as open in `STATUS.md`, not settled here.
- **Size:** the served set is the 2.44 GB transformer plus the 19.7 GB components (the float32 T5 shards and the VAE) plus, for evaluation, the 0.6 GB scorer — about 22.8 GB, far above the fleet's 9 GB publication convenience gate. A profile could carry a float16 re-export of the encoder (9.5 GB) instead of the float32 shards, but that would be a maintainer-converted file with a new digest, which this repository does not produce.
- **Upload set (if hosted):** `transformer/diffusion_pytorch_model.safetensors`, the `text_encoder/`, `tokenizer/`, `vae/` and `scheduler/` folders of the components snapshot; the scorer is evaluation-only and not part of a serving profile. No pickle exists anywhere in the set.
- Loader trust boundary: no `trust_remote_code`, no Hub-hosted code, safetensors only; the classes are `diffusers.PixArtTransformer2DModel`, `diffusers.AutoencoderKL`, `transformers.T5EncoderModel`, `transformers.AutoTokenizer` and `diffusers.PixArtSigmaPipeline` at the pinned versions.
- Serving shape: generation needs the transformer (1.2 GB float16) and the VAE (0.33 GB) resident, plus either the text encoder (9.5 GB float16) or pre-encoded prompt embeddings; a 512² image at 20 steps takes a few seconds on a T4 and about a second on an RTX 5070 Ti (`MODEL_CARD.md`, *Runtime*). An adapted profile adds a 448-tensor LoRA adapter of about 17 MB (float32).
- Line endings: `.gitattributes` carries `weights/** -text`, so a Windows checkout cannot rewrite a snapshot file's newlines and break its recorded digest.
