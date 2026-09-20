# Release verification

`tutorials/flux_schnell_generation_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate record.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `samples.py`, `metrics.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` and `SCORER_MANIFEST` equal to the two committed snapshot
  manifests and the inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its restart-on-stale-import
  guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before staging), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the staging mirror's
  revision `9df3faa7…` and the scorer revision `1a25a446…` are the only other 40-hex commits the documents may name);
- the profile-specific public-API calls (`stage_missing_files` / `stage_missing_scorer_files` with
  `allow_download=True`, `verify_snapshot` and `verify_scorer_snapshot`,
  `FluxSchnellPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, use_lora=True)`, `fetch_sample_dataset` from the
  pinned cache path, `load_byod_dataset`, `dataset_manifest`, `write_dataset_csv`, `validate_dataset` with the
  refusal probes, `pipe.encode_prompts`, `pipe.export_prompt_cache` and `pipe.release_text_encoder`,
  `pipe.load_transformer` with the parameter count printed, `pipe.evaluate` and `pipe.generate` + `score_generations`
  + `real_photo_baseline` on the frozen model, `pipe.adapt` with its explicit hyperparameters, `pipe.evaluate` after
  adaptation with the guaranteed assertions (kept-epoch validation loss ≤ frozen; re-scored validation loss matches the
  history), the new-prompt generation, `pipe.save_artifact`, `pipe.release_transformer`,
  `FluxSchnellPipeline.from_artifact(..., prompt_cache=cache)` and the reload-parity assertion, and the provenance
  fields `staging`, `safetensors_only: True`, `remote_code_executed: False` and the data base URL), the six expected
  `outputs/` paths, the learner-facing statements (ungated mirror, the transformer does not fit in 16-bit, the encoders
  do not fit beside it, generation has no ground truth, flow-matching loss, real-photo ceiling, not a human judgement,
  Apache-2.0, sample-sanity, CC0, every number is the 4-bit model's) and the gated-off BYOD default; forbidden patterns
  (credential-in-URL, any `git clone` / `github.com` / repository import on the primary path, a mutable
  `revision='main'`, direct `huggingface_hub` / `safetensors` / `urllib` / `diffusers` / `transformers` / `peft` /
  `bitsandbytes` / `BitsAndBytesConfig` / `FluxTransformer2DModel` / `T5EncoderModel` / `CLIPTextModel` / `CLIPModel`
  use, `torch.load(` / `pickle.load` / `Unpickler`, `torch.no_grad(` / `torch.inference_mode(` / `.backward(` /
  `pipe.transformer(` **outside the carried module cells**, `trust_remote_code=True`, `add_adapter(` /
  `inject_adapter_in_model(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_samples.py`, `tests/test_adaptation.py` (stub transformer, skipped without
torch), `tests/test_role_helpers.py`, `tests/test_import_boundary.py`, `tests/test_notebook_parity.py`; temporary
manifests, synthetic images, an injected fetcher, no weights and none of `diffusers`, `transformers`, `peft` or
`bitsandbytes`). These are source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab GPU runtime (T4 or better, ≥ 15 GB, CUDA with `bitsandbytes` support) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |

No local pre-flight path exists for this row: the workstation runs no GPU jobs by decision (2026-09-20), and the
transformer cannot be exercised without a CUDA GPU. Every executed run is a hosted clean-runtime run.

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new GPU runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshots `weights/flux1-schnell/`, `weights/clip-vit-b-32-laion2b/` or the data cache `weights/inat-birds/` (the
   standalone path writes the two manifests itself, stages all 32 listed files — 23 from the mirror
   `unsloth/FLUX.1-schnell` at its pinned revision, 9 from the scorer repository — about 34.3 GB — and fetches the 60
   pinned photographs, so none of the directories may be seeded); the runtime needs about 36 GB of free disk and a
   CUDA GPU of at least 15 GB;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `STEPS = 4`, `IMAGES_PER_PROMPT = 1`, `EPOCHS = 3`, `LEARNING_RATE = 1e-4`, `BATCH_SIZE = 1`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `diffusers==0.40.0`,
   `transformers==5.17.0`, `peft==0.21.0`, `bitsandbytes==0.50.2`, `torchao==0.18.0`, `accelerate==1.15.0`,
   `tokenizers==0.23.2`, `sentencepiece==0.2.2`, `protobuf==7.36.2`, `safetensors==0.8.0`, `huggingface-hub==1.32.0`,
   `numpy==2.5.3`, `pillow==11.3.0` (an interpreter restart after the install is expected where the runtime's
   preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `FluxSchnellPipeline`, `build_transformer`,
     `count_parameters`, `lora_parameter_names`, `pack_latents` / `unpack_latents` / `latent_image_ids`, the two
     `verify_*_snapshot` and `stage_missing_*` functions, `validate_inputs`, `validate_dataset`, `validate_prompts`,
     `preprocess_image`, `fetch_corpus`, `fetch_sample_dataset`, `build_sample_dataset`, `split_dataset`,
     `load_byod_dataset`, `write_dataset_csv`, `dataset_manifest`, `sample_prompts`, `ClipScorer`,
     `score_generations`, `real_photo_baseline`) with no import of the repository package;
   - the inline manifests asserted against the module's constants, then the two staging calls reporting 23 + 9
     entries fetched (the 23 from the mirror at `9df3faa7…`) and the two verifications reporting 23 / 9 verified files;
   - the model cell loading the VAE in float32, both tokenizers and the scheduler config with `source`
     "local-snapshot (manifest verified; safetensors only; staged from unsloth/FLUX.1-schnell)" and `device` `cuda`
     — the transformer is **not** loaded here;
   - the dataset manifest with 36 / 12 / 12 records, six distinct captions, `outputs/flux_schnell_generation_sample_captions.csv`
     written, and three refusals (missing caption, 200 px image, duplicate id);
   - `pipe.encode_prompts` reporting the encoders loaded in float16 on `cuda`, 7 prompts encoded (six captions and
     the new prompt), 0 truncations, finite embeddings of shape (256, 4096) and pooled vectors of shape (768,), and
     `release_text_encoder` returning `True` with the GPU memory back near zero;
   - `pipe.load_transformer` reporting the 4-bit load (`quantization nf4`, 11,891,178,560 parameters as the checkpoint
     counts them, 380 LoRA tensors) and the GPU memory after the load;
   - the frozen model's held-out flow-matching MSE on the validation and test records, six 512² four-step generations
     scored by CLIP with `outputs/flux_schnell_generation_frozen_grid.jpg` written, and the real-photo ceiling on the
     test photographs;
   - `pipe.adapt` printing epoch 0 as the frozen model, 9,338,880 trainable of 11,900,517,440 parameters, 108 steps,
     and a three-epoch history with the validation flow-matching MSE at the kept epoch no higher than the frozen
     model's;
   - `pipe.evaluate` on the validation and test records with the paired comparison, the assertions that the kept
     epoch's validation loss is no higher than the frozen model's and that the re-scored validation loss matches the
     history within 10⁻⁴ (the test change is printed as an observation), the adapted generations scored with
     `outputs/flux_schnell_generation_adapted_grid.jpg` written, and `outputs/flux_schnell_generation_evaluation_report.json`;
   - the new prompt rendered twice and CLIP-scored, and one parity image of the first training prompt rendered by the
     adapted pipeline;
   - `pipe.save_artifact` writing `outputs/flux_schnell_generation_adapter/{adapter.safetensors,manifest.json}`
     (380 tensors, `quantization nf4`), `pipe.release_transformer` returning `True`, and
     `FluxSchnellPipeline.from_artifact` reloading the artifact into a fresh 4-bit pipeline that adopts the exported
     prompt cache, with the held-out MSE and a seeded generation matching the adapted pipeline (the cell asserts
     `flow_matching_mse_diff < 1e-5` and `mean_abs_pixel_diff < 1.0`);
   - `outputs/flux_schnell_generation_result.json` written with `NOTEBOOK_SOURCE`, the identities and licences, the
     provenance block (`staging`, `safetensors_only: true`, `remote_code_executed: false`, the encoders released before
     training, the transformer loaded after encoding, the data base URL), the runtime versions, the comparison and the
     reload parity;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, device), the model identifier and
   immutable revision, whether the model cache, the weights directories and the data cache were clean, outcome,
   produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable `SHOULD`
   deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `flux_schnell_generation_colab.ipynb` (`E2E`) | pending | — | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-flux-schnell-generation`) | **Not yet executed** — the first clean run is queued behind the feasibility probe recorded below |

## Feasibility probe (not a notebook execution)

Before the notebook was run, a throwaway Kaggle kernel (`kurtvalcorza/dimer-probe-flux-schnell-qlora`, Tesla T4) staged
the 23 pinned files from the mirror with SHA-256 verification, encoded one prompt with the float16 text encoders,
released them, loaded the transformer 4-bit, generated one 512² image in 4 steps, attached a rank-8 LoRA and ran three
flow-matching training steps with gradient checkpointing — once with float16 compute and once with bfloat16 (emulated on
a T4) — printing staging time, load time, generation time, per-step time, peak memory and finiteness. Its outcome is
recorded here when it completes; it is engineering evidence for the recipe, not release evidence for the notebook.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/flux_schnell_generation_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/flux_schnell_generation_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| — | pending | Kaggle Tesla T4 | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout | — | **Not yet executed** |

## Current status

**Candidate.** The `E2E` carrier, its offline tests, the generator parity checks and the release-asset validation are in
place at the current revision; no clean-runtime execution of the committed notebook blob has been recorded yet. The
registry stays at **Candidate** until a clean Kaggle Tesla T4 (or Colab) run of the exact committed blob is recorded
above. The DIMER upload of the weights is on HOLD by the maintainer's decision (2026-09-20) independently of this gate.
