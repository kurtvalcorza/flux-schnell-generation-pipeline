# FLUX.1 [schnell] Generation Pipeline

DIMER-oriented pipeline for **FLUX.1 [schnell]** (`black-forest-labs/FLUX.1-schnell`, the Apache-2.0, 12 B-parameter rectified-flow transformer distilled for one-to-four-step text-to-image generation), pinned to an immutable Hugging Face revision, staged from an ungated mirror against a committed digest manifest, loaded 4-bit (bitsandbytes NF4) because nothing else fits a 16 GB GPU, with a CLIP scorer for evaluation. The repository exposes seeded few-step generation, a held-out flow-matching-loss and CLIP-scored evaluation against a real-photo ceiling, a captioned-image contract with explicit ceilings, a bounded QLoRA fine-tuning contract with a portable safetensors adapter, a `MODEL_CARD.md` at DIMER Model Card Specification 1.1, and a standalone `E2E` tutorial at DIMER Notebook Specification 2.0. **The DIMER upload of the weights is on HOLD by decision (2026-09-20); this repository is the E2E carrier.**

## Upstream alignment

- Model (identity of record): `black-forest-labs/FLUX.1-schnell`
- Revision: `741f7c3ce8b383c54771c7003378a50191e9efe9`
- Staging source: `unsloth/FLUX.1-schnell` at `9df3faa7ae3b6ddf0b2b69bb78616372897cc65c` — an ungated, unmodified mirror; the upstream repository is click-through gated and hides its file digests, so every byte is verified against the manifest committed here (`docs/WEIGHTS.md` records the byte-identity evidence). Scorer `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` at `1a25a446712ba5ee05982a381eed697ef9b435cf` (evaluation only)
- Upstream weight license: **Apache-2.0**; MIT for the scorer
- Upstream task: few-step text-to-image generation — a 19 double-stream + 38 single-stream rectified-flow transformer (hidden size 3,072, 11.9 B parameters) predicting the velocity of a 16-channel latent from CLIP-L pooled and T5-XXL token embeddings, no classifier-free guidance, decoded by the FLUX VAE
- Runtime: `diffusers==0.40.0` + `transformers==5.17.0` + `peft==0.21.0` + `bitsandbytes==0.50.2` + `torch==2.14.0` — every weight file is safetensors, **nothing is unpickled and no Hub-hosted code is executed**
- Repository adaptation: **E2E** (bounded LoRA fine-tuning of the image-stream attention projections of all 57 blocks over the 4-bit base — QLoRA — on captioned images, with a portable safetensors adapter)

## Three things to know before you start

**The transformer is 4-bit or nothing.** The three bfloat16 shards are 23.8 GB; `build_transformer()` loads them through `bitsandbytes` NF4 with double quantisation (about 6.6 GB on the GPU, float16 compute) and asserts the parameter count the checkpoint declares (11,891,178,560). The adapter is trained over that quantised base and its manifest records `nf4`; every number in this repository is the 4-bit model's, and nothing here measures what quantisation costs against the bfloat16 checkpoint.

**The text encoders and the transformer never share the GPU.** CLIP-L and T5-XXL are 9.7 GB in float16. `encode_prompts()` loads them, encodes every distinct prompt into a cache of (256 × 4,096) embeddings and (768,) pooled vectors, and `release_text_encoder()` drops them; `load_transformer()` (called by the first `generate()`, `evaluate()` or `adapt()`) releases them itself if they are still resident, and `encode_prompts()` refuses to run while the transformer is loaded — encode first, or `release_transformer()`. `export_prompt_cache()` / `import_prompt_cache()` hand the embeddings to a second pipeline (the reload check) without loading the encoders again.

**Generation has no ground truth, and the numbers say what they are.** `evaluate()` reports the held-out *flow-matching loss* — the training objective on photographs the model never trained on, at five fixed noise levels with seeded noise, so the frozen and the adapted model see identical inputs. `metrics.score_generations()` reports CLIP prompt similarity, the fraction of generated images CLIP assigns to their own caption among the dataset's captions, and similarity to the held-out real photographs; `real_photo_baseline()` reports the same three numbers on the real photographs — the ceiling. None of these is a human judgement of image quality, and the sample results in `MODEL_CARD.md` are sample-sanity observations, not a quality claim.

## Quick start

```python
from flux_schnell_generation_pipeline import FluxSchnellPipeline, fetch_sample_dataset, sample_prompts
from flux_schnell_generation_pipeline.metrics import ClipScorer, score_generations, real_photo_baseline

pipe = FluxSchnellPipeline.from_pretrained(use_lora=True)   # verifies the snapshot; loads VAE + tokenizers + scheduler config (CUDA required)
splits = fetch_sample_dataset()                              # 36 / 12 / 12 pinned CC0 iNaturalist bird photographs, six captions
pipe.encode_prompts(sample_prompts(splits["train"] + splits["validation"] + splits["test"]))
pipe.release_text_encoder()                                  # the 9.7 GB of encoders are gone; the embeddings stay
pipe.load_transformer()                                      # 4-bit NF4, about 6.6 GB, untrained LoRA attached
print(pipe.evaluate(splits["test"])["flow_matching_mse"])   # frozen model, held-out flow-matching loss
scorer = ClipScorer()
images = pipe.generate(sample_prompts(splits["test"]), seed=1000, steps=4)["images"]
print(score_generations(scorer, images, references=splits["test"]), real_photo_baseline(scorer, splits["test"]))
pipe.adapt(splits["train"], splits["validation"], epochs=3)  # bounded QLoRA fine-tuning, epoch selected by validation loss
print(pipe.evaluate(splits["test"])["flow_matching_mse"])   # adapted model, identical inputs
pipe.save_artifact("outputs/adapter")
```

`evaluate()` and `adapt()` take records — `{id, image, caption}` with an RGB `PIL.Image` (or a file path) whose shorter side is 256..4096 px and a caption of 1..1000 characters; images are resized so the shorter side is 512 px and centre-cropped to 512 × 512, captions are truncated to 256 T5 tokens (both reported). `generate()` takes prompts, a seed and `steps` (default 4, at most 8); schnell is guidance-distilled, so there is no guidance scale. Validation is structural: nothing checks that a caption describes its image.

## Weights layout

```
weights/flux1-schnell/            model_index.json  scheduler/  tokenizer/  tokenizer_2/  transformer/config.json  vae/config.json
                                  text_encoder/config.json  text_encoder_2/{config.json, model.safetensors.index.json}
                                  transformer/diffusion_pytorch_model.safetensors.index.json  dimer-base-manifest.json
                                  transformer/diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors  (git-ignored, 23.8 GB)
                                  text_encoder_2/model-0000{1,2}-of-00002.safetensors  (9.5 GB)  text_encoder/model.safetensors  (0.25 GB)
                                  vae/diffusion_pytorch_model.safetensors  (0.17 GB)  tokenizer_2/spiece.model  (fetched with the shards)
weights/clip-vit-b-32-laion2b/    config, tokenizer and preprocessor files  dimer-base-manifest.json  model.safetensors  (git-ignored, 605 MB)
weights/inat-birds/               the 60 pinned photographs, cached on first fetch (git-ignored)
```

`from_pretrained()` calls `stage_missing_files()` (fetch only absent manifest entries, only from the pinned mirror revision, only with `allow_download=True`), then `verify_snapshot()` (byte size + SHA-256 of every manifest entry, and the manifest must name the pinned staging source), and refuses on the first mismatch; `ClipScorer()` does the same for the scorer snapshot. `docs/WEIGHTS.md` records the provenance of both snapshots, the mirror caveats, the state of the byte-identity proof and the DIMER hosting notes.

## Sample data

`fetch_sample_dataset()` fetches 60 research-grade iNaturalist photographs of six North American birds (10 per species, one per observer per species, every one CC0) from the public open-data bucket, each pinned by byte size and SHA-256 in `SAMPLE_RECORDS` and refused on a mismatch before decoding, with the observer and observation page recorded. Captions come from one template per species. `build_sample_dataset` draws a seeded 6 / 2 / 2 stratified split per species (36 / 12 / 12) and `check_split_disjoint` asserts no image appears twice. `write_dataset_csv` / `load_byod_dataset` round-trip the `captions.csv` + image files layout, the BYOD format.

## Adapter artifacts

`save_artifact(dir)` writes `adapter.safetensors` (the 380 LoRA tensors — rank 8 on `to_q` / `to_k` / `to_v` of all 57 blocks and `to_out.0` of the 19 double-stream blocks, 9,338,880 parameters in float32, about 37 MB) and a `manifest.json` recording the artifact format, the exact base model id, revision and licence, the staging source, the quantisation (`nf4`), the LoRA configuration, the tensor names, the file size and SHA-256, the training configuration and the epoch history. `FluxSchnellPipeline.from_artifact(dir, prompt_cache=...)` re-verifies the snapshot, checks the manifest, scope and digest before deserialising, rebuilds the pipeline with an untrained LoRA over a fresh 4-bit base and overlays the tensors in place.

## Tests

```
pip install -e . --no-deps
pytest -q -o addopts= tests
```

Tests are offline: temporary manifests, synthetic images, an injected fetcher and a stub transformer (the flow-matching target, the training loop, epoch selection, the transactional guarantee, the residency rule and the artifact round trip all run on it), never the weights or the model libraries; the recorded GPU runs are in `MODEL_CARD.md` (*Runtime*) and `docs/release-verification.md`.

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/flux-schnell-generation-pipeline/blob/main/tutorials/flux_schnell_generation_colab.ipynb)

`tutorials/flux_schnell_generation_colab.ipynb` is declared `E2E` and is **standalone** (DIMER Notebook Specification 2.0 §4): it is generated by `tools/build_notebook.py` from `tools/notebook_template.py` and embeds the 3 package modules (`pipeline.py`, `samples.py`, `metrics.py`) verbatim in dependency order, the two pinned identities and manifests and the exact runtime pins, so the exported `.ipynb` keeps working without this repository being reachable. It needs a CUDA GPU runtime with at least 15 GB of memory, `bitsandbytes` support and about 36 GB of disk. It stages the snapshot from the mirror and verifies it (34 GB of downloads), fetches the pinned photographs, and runs the sample path: validation, prompt encoding and encoder release, the 4-bit transformer load, the frozen model's held-out loss and CLIP-scored four-step generations against the real-photo ceiling, bounded QLoRA fine-tuning, the paired held-out comparison, a new prompt, adapter export, release of the adapted transformer and reload parity. Do not edit the notebook by hand; regenerate it (`python tools/build_notebook.py`; `--check` is enforced by the validator and CI).

## Release status

**Candidate** — the E2E carrier, its tests, the generator parity and the release-asset validation are in place; promotion to Release-grade follows the exact committed notebook blob executing top-to-bottom in a clean Kaggle Tesla T4 runtime with no repository checkout (`docs/release-verification.md`). The DIMER upload of the weights stays on HOLD by decision (2026-09-20). Static and unit checks are necessary but are never the evidence.

## Licensing

- Upstream weights: Apache-2.0 (`black-forest-labs/FLUX.1-schnell`), staged unmodified from the pinned mirror revision and verified against the committed manifest; the mirror's `NOTICE` file carries the wrong (FLUX.1 [dev] non-commercial) text and is not part of the snapshot — the upstream front matter, the upstream `LICENSE.md` and the mirror's own front matter all say Apache-2.0. The scorer is MIT.
- Tutorial data: iNaturalist research-grade photographs, each CC0 1.0 (observers credited in `samples.py`); fetched at run time, never committed.
- This repository's code and documentation: Apache-2.0 (`LICENSE`).
- The upstream licence governs your use of the weights, including commercial use and redistribution; this repository grants no rights beyond it.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
