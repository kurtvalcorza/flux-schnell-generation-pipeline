# PixArt-Σ Generation Pipeline

DIMER-oriented pipeline for **PixArt-Σ XL-2 512-MS** (`PixArt-alpha/PixArt-Sigma-XL-2-512-MS`, the 0.6 B-parameter Diffusion Transformer for 512 × 512 text-to-image generation), pinned to an immutable Hugging Face revision together with the T5-XXL / SDXL-VAE / scheduler components the authors publish beside it and a CLIP scorer for evaluation. The repository exposes seeded text-to-image generation, a held-out denoising-loss and CLIP-scored evaluation against a real-photo ceiling, a captioned-image contract with explicit ceilings, a bounded LoRA fine-tuning contract with a portable safetensors adapter, a `MODEL_CARD.md` at DIMER Model Card Specification 1.1, and a standalone `E2E` tutorial at DIMER Notebook Specification 2.0.

## Upstream alignment

- Model: `PixArt-alpha/PixArt-Sigma-XL-2-512-MS`
- Revision: `76fb7eb5a9314bc1e4e479d2f13447517fca9be4`
- Components: `PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers` at `2c17b4e85261cd549b4068d086b7c2ba9d468e9f` (T5-XXL encoder, tokenizer, SDXL VAE, DPM-Solver++ scheduler config); scorer `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` at `1a25a446712ba5ee05982a381eed697ef9b435cf` (evaluation only)
- Upstream weight license: **CreativeML Open RAIL++-M** (`openrail++`) for the transformer; `openrail` for the components repository; MIT for the scorer
- Upstream task: text-to-image generation at 512 × 512 — a 28-block DiT denoising a 4-channel latent under classifier-free guidance from 300 T5 token embeddings, decoded by the SDXL VAE
- Runtime: `diffusers==0.40.0` + `transformers==5.17.0` + `peft==0.21.0` + `torch==2.14.0` — every weight file is safetensors, **nothing is unpickled and no Hub-hosted code is executed**
- Repository adaptation: **E2E** (bounded LoRA fine-tuning of the attention projections of all 28 blocks on captioned images, with a portable safetensors adapter)

## Two things to know before you start

**The text encoder is loaded once, used once and released.** The float32 T5-XXL shards are 19 GB on disk and 9.5 GB in float16 on the GPU — too much to keep beside a training graph on a 16 GB card. `encode_prompts()` loads the encoder, encodes every distinct prompt (and the empty negative prompt) into a cache of 300 × 4,096 embeddings, and `release_text_encoder()` drops it; `generate()`, `evaluate()` and `adapt()` read the cache, and `export_prompt_cache()` / `import_prompt_cache()` hand it to a second pipeline (the reload check) without loading the encoder again.

**Generation has no ground truth, and the numbers say what they are.** `evaluate()` reports the held-out *denoising loss* — the training objective on photographs the model never trained on, at five fixed timesteps with seeded noise, so the frozen and the adapted model see identical inputs. `metrics.score_generations()` reports CLIP prompt similarity, the fraction of generated images CLIP assigns to their own caption among the dataset's captions, and similarity to the held-out real photographs; `real_photo_baseline()` reports the same three numbers on the real photographs — the ceiling. None of these is a human judgement of image quality, and the sample results in `MODEL_CARD.md` are sample-sanity observations, not a quality claim.

## Quick start

```python
from pixart_sigma_generation_pipeline import PixArtSigmaPipeline, fetch_sample_dataset, sample_prompts
from pixart_sigma_generation_pipeline.metrics import ClipScorer, score_generations, real_photo_baseline

pipe = PixArtSigmaPipeline.from_pretrained(use_lora=True)   # verifies the three snapshots, loads transformer + LoRA, VAE, tokenizer
splits = fetch_sample_dataset()                              # 36 / 12 / 12 pinned CC0 iNaturalist bird photographs, six captions
pipe.encode_prompts(sample_prompts(splits["train"] + splits["validation"] + splits["test"]))
pipe.release_text_encoder()                                  # the 9.5 GB encoder is gone; the embeddings stay
print(pipe.evaluate(splits["test"])["denoising_mse"])       # frozen model, held-out denoising loss
scorer = ClipScorer()
images = pipe.generate(sample_prompts(splits["test"]), seed=1000)["images"]
print(score_generations(scorer, images, references=splits["test"]), real_photo_baseline(scorer, splits["test"]))
pipe.adapt(splits["train"], splits["validation"])            # bounded LoRA fine-tuning, epoch selected by validation loss
print(pipe.evaluate(splits["test"])["denoising_mse"])       # adapted model, identical inputs
pipe.save_artifact("outputs/adapter")
```

`evaluate()` and `adapt()` take records — `{id, image, caption}` with an RGB `PIL.Image` (or a file path) whose shorter side is 256..4096 px and a caption of 1..1000 characters; images are resized so the shorter side is 512 px and centre-cropped to 512 × 512, captions are truncated to 300 T5 tokens (both reported). `generate()` takes prompts, a seed, `steps` (default 20, at most 100) and `guidance_scale` (default 4.5, at most 20). Validation is structural: nothing checks that a caption describes its image.

## Weights layout

```
weights/pixart-sigma-xl-2-512-ms/   README.md  transformer/config.json  dimer-base-manifest.json
                                    transformer/diffusion_pytorch_model.safetensors  (git-ignored, 2.44 GB)
weights/pixart-sigma-t5-vae/        README.md  model_index.json  scheduler/  tokenizer/  vae/config.json  text_encoder/config.json
                                    text_encoder/model.safetensors.index.json  dimer-base-manifest.json
                                    text_encoder/model-0000{1,2}-of-00002.safetensors  vae/diffusion_pytorch_model.safetensors  (git-ignored, 19.4 GB)
weights/clip-vit-b-32-laion2b/      config, tokenizer and preprocessor files  dimer-base-manifest.json  model.safetensors  (git-ignored, 605 MB)
weights/inat-birds/                 the 60 pinned photographs, cached on first fetch (git-ignored)
```

`from_pretrained()` calls `stage_missing_files()` and `stage_missing_base_files()` (fetch only absent manifest entries, only at the pinned revisions, only with `allow_download=True`), then `verify_snapshot()` and `verify_base_snapshot()` (byte size + SHA-256 of every manifest entry), and refuses on the first mismatch; `ClipScorer()` does the same for the scorer snapshot. `docs/WEIGHTS.md` records the provenance of all three snapshots, the licence gate and the DIMER hosting notes.

## Sample data

`fetch_sample_dataset()` fetches 60 research-grade iNaturalist photographs of six North American birds (10 per species, one per observer per species, every one CC0) from the public open-data bucket, each pinned by byte size and SHA-256 in `SAMPLE_RECORDS` and refused on a mismatch before decoding, with the observer and observation page recorded. Captions come from one template per species. `build_sample_dataset` draws a seeded 6 / 2 / 2 stratified split per species (36 / 12 / 12) and `check_split_disjoint` asserts no image appears twice. `write_dataset_csv` / `load_byod_dataset` round-trip the `captions.csv` + image files layout, the BYOD format.

## Adapter artifacts

`save_artifact(dir)` writes `adapter.safetensors` (the 448 LoRA tensors — rank 8 on `to_q` / `to_k` / `to_v` / `to_out.0` of both attentions of all 28 blocks, 4,128,768 parameters) and a `manifest.json` recording the artifact format, the exact base model id and revision, the components revision, the LoRA configuration, the tensor names, the file size and SHA-256, the training configuration and the epoch history. `PixArtSigmaPipeline.from_artifact(dir)` re-verifies both snapshots, checks the manifest, scope and digest before deserialising, rebuilds the pipeline with an untrained LoRA and overlays the tensors.

## Tests

```
pip install -e . --no-deps
pytest -q -o addopts= tests
```

Tests are offline: temporary manifests, synthetic images, an injected fetcher and a stub transformer, never the weights or the model libraries; the model-backed smoke is recorded in `MODEL_CARD.md` (*Runtime*).

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/pixart-sigma-generation-pipeline/blob/main/tutorials/pixart_sigma_generation_colab.ipynb)

`tutorials/pixart_sigma_generation_colab.ipynb` is declared `E2E` and is **standalone** (DIMER Notebook Specification 2.0 §4): it is generated by `tools/build_notebook.py` from `tools/notebook_template.py` and embeds the 3 package modules (`pipeline.py`, `samples.py`, `metrics.py`) verbatim in dependency order, the three pinned identities and manifests and the exact runtime pins, so the exported `.ipynb` keeps working without this repository being reachable. It needs a GPU runtime with at least 15 GB of memory and about 25 GB of disk. It stages and verifies the three snapshots (22 GB of downloads), fetches the pinned photographs, and runs the sample path: validation, prompt encoding and encoder release, the frozen model's held-out loss and CLIP-scored generations against the real-photo ceiling, bounded LoRA fine-tuning, the paired held-out comparison, a new prompt, adapter export and reload parity. Do not edit the notebook by hand; regenerate it (`python tools/build_notebook.py`; `--check` is enforced by the validator and CI).

## Release status

**Release-grade** — the `E2E` notebook blob `99829b51` (committed at `4862a6e`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-19 (11/11 ok (1 restart after install cell), 1582.8 s); the record is in `docs/release-verification.md` and `STATUS.md`. Static and unit checks — including the standalone generator parity checks — are necessary but were never the evidence; the hosted run is. A later change to the carried modules or the notebook returns the status to Candidate until re-verified.

## Licensing

- Upstream weights: CreativeML Open RAIL++-M (`PixArt-alpha/PixArt-Sigma-XL-2-512-MS`), staged from the pinned revision and not modified; the licence's use-based restrictions apply to the weights, to any adapter trained on them and to any DIMER profile that carries them. The components repository declares `openrail`; the scorer is MIT.
- Tutorial data: iNaturalist research-grade photographs, each CC0 1.0 (observers credited in `samples.py`); fetched at run time, never committed.
- This repository's code and documentation: Apache-2.0 (`LICENSE`).
- The upstream licences govern your use of the weights, including commercial use and redistribution; this repository grants no rights beyond them.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
