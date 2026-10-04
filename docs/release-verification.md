# Release verification

`tutorials/flux_schnell_generation_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate record.

> **2026-10-04 notebook review fixes (FS-M1..M4, FS-m1..m5; review PR #9).** The tutorial's blob changed from
> `52226812`. It no longer installs into the kernel: Section 1 builds a uv isolated environment (managed CPython
> 3.12.12) from the hash lock `tutorials/requirements-colab.lock.txt` (`--require-hashes --only-binary :all:`, the
> same pins as `pyproject.toml`) and routes every later cell to it, so no restart is part of the procedure; it runs on
> **Linux x86_64 only**. The real photographs are now a leave-one-out reference line (`real_photo_reference`), not a
> ceiling; Section 5 and Section 9 release every GPU resident so a BYOD re-run fits a 15 GB T4 (projected, not run);
> model loads stream to the GPU (`device_map`) for a 12.7 GiB Colab VM; the guided layer, the BYOD fixes and an
> optional steps activity were added. The status is **Candidate**: the 2026-09-20 run below needed a restart after the
> install cell (2 passes, not a one-pass `Run all`); the new blob `d26eb848` (committed at `7b4782c`) then completed
> one pass under the Colab CLI 0.7.4 on a fresh Colab Tesla T4 on 2026-10-04 (14/14 code cells, 0 errors, no restart, 3129.5 s; default path only, not a browser `Run all`).

> **2026-10-03 uv isolated environment (capstone only).** The Bosch capstone notebook's blob changed from `c64f21cc` to
> `84061cc7`. It now gets uv from a pinned wheel instead of a kernel install, and it uses hash locks
> (`--require-hashes --only-binary :all:`) with the same pins. It runs on Linux x86_64 only. A hosted re-run is pending,
> and the status is **Candidate**. Details are in [`capstone-verification.md`](capstone-verification.md). This file's
> notebook, `flux_schnell_generation_colab.ipynb`, did not change.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, a revision that contains the carried modules — never a
  `+working-tree` label — the module SHA-256 overall and per file, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `samples.py`, `metrics.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` and `SCORER_MANIFEST` equal to the two committed snapshot
  manifests and the inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; exactly two kernel cells — the uv isolated install from
  the carried hash lock (`--managed-python`, `--require-hashes`, `--only-binary :all:`, the uv wheel's size and SHA-256,
  Linux x86_64 only) and the router whose `google.colab` stubs carry a module spec — and every other cell routed to the
  isolated environment; the Infrastructure titles on the seven setup cells; `NOTEBOOK_SOURCE` with the per-module
  SHA-256 recorded in exports;
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
  + `real_photo_reference` (leave-one-out) on the frozen model, `pipe.adapt` with its explicit hyperparameters, `pipe.evaluate` after
  adaptation with the guaranteed assertions (kept-epoch validation loss ≤ frozen; re-scored validation loss matches the
  history), the new-prompt generation, `pipe.save_artifact`, `pipe.release_transformer`,
  `FluxSchnellPipeline.from_artifact(..., prompt_cache=cache)` and the reload-parity assertion, and the provenance
  fields `staging`, `safetensors_only: True`, `remote_code_executed: False` and the data base URL), the six expected
  `outputs/` paths, the learner-facing statements (ungated mirror, the transformer does not fit in 16-bit, the encoders
  do not fit beside it, generation has no ground truth, flow-matching loss, reference line, not a ceiling, leave-one-out,
  pretraining overlap, run-to-run variability, stratified within each caption, not a human judgement, Apache-2.0,
  sample-sanity, CC0, every number is the 4-bit model's), the guided-layer markers, the absence of the stale text the
  review removed (no markdown calls the real photographs a ceiling) and the gated-off BYOD default; forbidden patterns
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
   `USE_BYOD = False`, `BYOD_PATH = ''`, `STEPS = 4`, `IMAGES_PER_PROMPT = 1`, `MAX_GENERATION_PROMPTS = 12`,
   `EPOCHS = 3`, `LEARNING_RATE = 1e-4`, `BATCH_SIZE = 1`, `RUN_ACTIVITY = False`); record the number of passes —
   a run that needed a restart or a second pass is not a one-pass `Run all` and does not promote the notebook;
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `diffusers==0.40.0`,
   `transformers==5.17.0`, `peft==0.21.0`, `bitsandbytes==0.50.2`, `torchao==0.18.0`, `accelerate==1.15.0`,
   `tokenizers==0.23.2`, `sentencepiece==0.2.2`, `protobuf==7.36.2`, `safetensors==0.8.0`, `huggingface-hub==1.32.0`,
   `numpy==2.5.3`, `pillow==11.3.0`, installed by Section 1 into the isolated environment from the hash lock (the
   kernel's own packages are not touched, so no restart is needed);
5. verify every default-path stage completes:
   - the isolated environment built from the carried hash lock with no GitHub access, and every later cell routed to it;
   - the three carried module cells execute (defining `FluxSchnellPipeline`, `build_transformer`,
     `count_parameters`, `lora_parameter_names`, `pack_latents` / `unpack_latents` / `latent_image_ids`, the two
     `verify_*_snapshot` and `stage_missing_*` functions, `validate_inputs`, `validate_dataset`, `validate_prompts`,
     `preprocess_image`, `fetch_corpus`, `fetch_sample_dataset`, `build_sample_dataset`, `split_dataset`,
     `load_byod_dataset`, `write_dataset_csv`, `dataset_manifest`, `sample_prompts`, `ClipScorer`,
     `score_generations`, `real_photo_reference`) with no import of the repository package;
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
   - `pipe.load_transformer` reporting the 4-bit load (`quantization nf4`, 11,900,517,440 parameters — the checkpoint's
     11,891,178,560 counted at their original shapes plus the 9,338,880 of the attached LoRA — 380 LoRA tensors) and the
     GPU memory after the load (about 6.5 GB);
   - the frozen model's held-out flow-matching MSE on the validation and test records, six 512² four-step generations
     scored by CLIP with `outputs/flux_schnell_generation_frozen_grid.jpg` written and displayed, and the leave-one-out
     real-photo reference on the test photographs;
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
     `flow_matching_mse_diff < 1e-5` and `mean_abs_pixel_diff < 1.0`), then the reloaded pipeline released;
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
| `flux_schnell_generation_colab.ipynb` (`E2E`) | `d4fdf72` / `52226812` | 2026-09-20 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-flux-schnell-generation` v2; image `torch 2.10.0+cu128` before the pinned install, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0`, `bitsandbytes 0.50.2` after; Python 3.12.13) | **PASSED in 2 passes — not a one-pass `Run all`**: pass 1 stopped at the install cell's stale-import guard (`RuntimeError: Core dependencies changed while older modules were loaded: cuda-bindings …, numpy …, protobuf …; Restart the runtime`), pass 2 after a restart: 11/11 code cells ok (1 restart after the install cell), 2911.9 s, 130 files / 34,341 MB fetched and digest-verified inside the notebook (the 23-file FLUX snapshot from the mirror, the 9-file scorer, the 60 pinned photographs); encoders loaded in 49.8 s and 7 prompts encoded in 3.3 s at 12.15 GB, released to 0.35 GB; 4-bit transformer loaded in 116.2 s at 6.52 GB (11,900,517,440 parameters with the 380 LoRA tensors attached); **frozen held-out flow-matching MSE validation 0.834244 / test 0.795300** (by σ 0.1..0.9: 1.073 / 0.955 / 0.750 / 0.593 / 0.606); six four-step generations in 43.6 s scored CLIP prompt similarity / label accuracy / reference similarity **31.08 / 0.667 / 71.95** while the real photographs scored 30.25 / 0.917 / 88.66 (that version included each photo in its own reference; leave-one-out, the reference similarity is 57.7, derived exactly from the run's per-image numbers); `adapt` 3 epochs, 108 steps, 1024.5 s, peak 12.23 GB: validation MSE 0.834244 → 0.594587 / 0.533845 / 0.523882 (best epoch 3; train loss 0.624 / 0.669 / 0.569); **adapted validation 0.523882, test 0.506942 (−0.288358 against the frozen model, an observation), generations 31.77 / 0.833 / 75.01**; the new prompt (a House Finch on a snow-covered branch) rendered twice in 14.6 s at CLIP prompt similarity 32.48; adapter 380 tensors / 37,404,064 bytes float32 (SHA-256 `34c3699a…`, `quantization nf4`); the adapted transformer released to 0.96 GB, the fresh 4-bit reload (`best_epoch 3`) reproduced the held-out MSE and the seeded image exactly (flow_matching_mse_diff 0.0, mean_abs_pixel_diff 0.0); run summary and executed notebook archived under `.agent/backups/tier-c-build-2026-09-20/kaggle-flux/dimer-nb2-flux-schnell-generation/v2/evidence/` in the workspace |
| `flux_schnell_generation_colab.ipynb` (`E2E`) | `77567b7` / `5c82c199` | 2026-09-20 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-flux-schnell-generation` v1) | **FAILED at the model cell** — install, restart and the carried modules ok; the 23-file snapshot staged from the mirror and sha256-verified (358 s), then `verify_scorer_snapshot` refused `README.md: unexpected file type in a code-free snapshot` because `SNAPSHOT_FILE_SUFFIXES` lacked `.md`; fixed in `d4fdf72` with an offline test that asserts every committed manifest entry passes the rule; no model loaded |

## Feasibility probe (not a notebook execution)

Before the notebook was run, a throwaway Kaggle kernel (`kurtvalcorza/dimer-probe-flux-schnell-qlora`, Tesla T4, 2026-09-20)
staged the 23 pinned files from the mirror with SHA-256 verification (608.6 s), encoded one prompt with the float16 text
encoders (46.5 s, peak 11,329 MiB, finite), released them, loaded the transformer 4-bit (99.6 s, 6,563 MiB), generated one
512² image in 4 steps (5.2 s, finite, a recognisable bird), attached the rank-8 LoRA (380 tensors, 9,338,880 parameters)
and ran three flow-matching training steps with gradient checkpointing (4.2 / 3.8 / 3.8 s, loss 0.59..0.60, peak 7,423 MiB)
with float16 compute; the bfloat16 branch ran out of memory at load because the float16 branch's tensors were still held
(a probe artefact; the T4 has no native bfloat16 in any case). Engineering evidence for the recipe, not release evidence.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/flux_schnell_generation_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/flux_schnell_generation_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-04 | `7b4782c` / `d26eb848` | Colab CLI 0.7.4 sequential execution, fresh Colab VM, Tesla T4 (not a browser `Run all`; the CLI records no execution counts, so the order is evidenced by `exec.log` "Executing cell k/14"); isolated environment Python 3.12.12, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0`, `bitsandbytes 0.50.2`, `cuda` | Default sample path only: every code cell in order in one kernel, the notebook fetched from GitHub at the full commit SHA with its blob SHA-1 verified before the VM was allocated; the snapshot, the scorer and the pinned photographs fetched and digest-verified by the notebook. BYOD, the invalid BYOD input and the Section 10 activity (`RUN_ACTIVITY = False`) were not exercised | 3129.5 s (suite wall time including VM allocation; the CLI reports no per-cell times) | **PASSED in one pass, no restart, 0 errors**: 14/14 code cells (cells 4–6 are carried module definitions with no output; cell 14 printed "Activity not run"); 23 snapshot files and 9 scorer files verified; encoders loaded in 40.3 s and 7 prompts encoded in 3.6 s at 12.14 GB, released to 0.34 GB; 4-bit transformer loaded in 102.6 s at 6.52 GB (11,900,517,440 parameters with the 380 LoRA tensors attached); **frozen held-out flow-matching MSE validation 0.834244 / test 0.795300** (by σ 0.1..0.9: 1.073 / 0.955 / 0.750 / 0.593 / 0.606); six four-step generations in 39.0 s scored CLIP prompt similarity / label accuracy / reference similarity **31.08 / 0.667 / 71.95**, the leave-one-out real-photo reference 30.25 / 0.917 / 57.72 (0 photos without a reference), 7.13 GB resident; `adapt` 3 epochs, 108 steps, 938.3 s, peak 12.22 GB: validation MSE 0.834244 → 0.591159 / 0.532501 / 0.523245 (best epoch 3; train loss 0.6218 / 0.665 / 0.5674); **adapted validation 0.523245, test 0.506467 (−0.288833 against the frozen model, an observation; lower at every σ), generations 30.35 / 0.667 / 71.89**; per-species reference similarity rose for three species (Song Sparrow 80.18 → 81.60, White-throated Sparrow 71.60 → 72.96, House Finch 60.61 → 63.24) and fell for three (Chipping Sparrow 70.24 → 67.31, American Goldfinch 79.89 → 79.08, Dark-eyed Junco 69.16 → 67.14), correct labels unchanged at 4/6; the new prompt rendered at CLIP prompt similarity 33.11 in 13.2 s; adapter 380 tensors / 37,404,064 bytes (SHA-256 `738a9797…`, `quantization nf4`); the adapted transformer released to 0.96 GB; the fresh 4-bit reload (`best_epoch 3`) reproduced the held-out MSE and the seeded image exactly (flow_matching_mse_diff 0.0, mean_abs_pixel_diff 0.0) at 7.47 GB and was released to 0.96 GB; no out-of-memory or host-RAM failure. Against the worked answers (recorded from the 2026-09-20 Kaggle run of the previous version): Sections 6 and 9 match; the validation curve differs in the third decimal (0.591 / 0.533 / 0.523 against 0.595 / 0.534 / 0.524) and peak memory is 12.22 GB against 12.23 GB; the Section 8 answer's label accuracy 0.667 → 0.833 and mean reference similarity 71.95 → 75.01 were **not** reproduced (0.667 → 0.667 and 71.95 → 71.89 here), and three species, not two, moved down. Evidence, byte for byte: `docs/execution-evidence/2026-10-04/flux_schnell_generation_colab_7b4782c_colab-cli-t4.ipynb` (SHA-256 `229a1350…2fab`), `…_7b4782c_run_summary.json` (`fc82c4e9…e259`), `…_7b4782c_exec.log` (`df56609e…559c`) |
| 2026-09-20 | `d4fdf72` / `52226812` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-flux-schnell-generation` v2; image `torch 2.10.0+cu128` before the pinned install, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0`, `bitsandbytes 0.50.2` after; Python 3.12.13, `cuda`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution); the snapshot staged from the mirror and digest-verified by the notebook, the scorer and the pinned photographs fetched by the notebook | 2911.9 s | **PASSED in 2 passes — not a one-pass `Run all`**: pass 1 stopped at the install cell's stale-import guard (`RuntimeError: Core dependencies changed while older modules were loaded: cuda-bindings …, numpy …, protobuf …; Restart the runtime`), pass 2 after a restart: 11/11 code cells ok (1 restart after the install cell), 2911.9 s, 130 files / 34,341 MB fetched and digest-verified inside the notebook (the 23-file FLUX snapshot from the mirror, the 9-file scorer, the 60 pinned photographs); per-cell: install 27 s (then restart), stage + verify 536 s, sample 30 s, encode + release 54 s, 4-bit load + frozen evaluation + generations 364 s, `adapt` 1025 s, adapted evaluation + generations 251 s, new prompt + export + release + reload 432 s; encoders loaded in 49.8 s and 7 prompts encoded in 3.3 s at 12.15 GB, released to 0.35 GB; 4-bit transformer loaded in 116.2 s at 6.52 GB (11,900,517,440 parameters with the 380 LoRA tensors attached); **frozen held-out flow-matching MSE validation 0.834244 / test 0.795300** (by σ 0.1..0.9: 1.073 / 0.955 / 0.750 / 0.593 / 0.606); six four-step generations in 43.6 s scored CLIP prompt similarity / label accuracy / reference similarity **31.08 / 0.667 / 71.95** while the real photographs scored 30.25 / 0.917 / 88.66 (that version included each photo in its own reference; leave-one-out, the reference similarity is 57.7, derived exactly from the run's per-image numbers); `adapt` 3 epochs, 108 steps, 1024.5 s, peak 12.23 GB: validation MSE 0.834244 → 0.594587 / 0.533845 / 0.523882 (best epoch 3; train loss 0.624 / 0.669 / 0.569); **adapted validation 0.523882, test 0.506942 (−0.288358 against the frozen model, an observation), generations 31.77 / 0.833 / 75.01**; the new prompt (a House Finch on a snow-covered branch) rendered twice in 14.6 s at CLIP prompt similarity 32.48; adapter 380 tensors / 37,404,064 bytes float32 (SHA-256 `34c3699a…`, `quantization nf4`); the adapted transformer released to 0.96 GB, the fresh 4-bit reload (`best_epoch 3`) reproduced the held-out MSE and the seeded image exactly (flow_matching_mse_diff 0.0, mean_abs_pixel_diff 0.0) |
| 2026-09-20 | `77567b7` / `5c82c199` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-flux-schnell-generation` v1) | Default sample path, same executor | — | **FAILED at the model cell** — the 23-file snapshot staged and digest-verified in 358 s, then the scorer manifest refused on `README.md` (suffix rule; fixed in `d4fdf72`) |

## Current status

**Candidate.** The 2026-10-04 review fixes produced the notebook blob `d26eb848` (committed at `7b4782c`), which
completed one pass under the Colab CLI 0.7.4 on a fresh Colab Tesla T4 on 2026-10-04 (14/14 code cells, 0 errors, no restart, 3129.5 s; default path only, not a browser `Run all`); see Recorded executions above. The
earlier `E2E` blob `52226812` (committed at `d4fdf72`) completed on a clean Kaggle Tesla T4 runtime on 2026-09-20 only
after a restart following the install cell (2 passes), and the v1 run and the feasibility probe above are history. What
remains before release: a BYOD run after the default path that reaches reload parity, one invalid BYOD input, and the
Section 10 activity, recorded here with the pass count; whether this CLI pass also stands for the one-pass `Run all` the
procedure names (it is not a browser `Run all`) and promotion are the maintainer's decisions. The DIMER upload of
the weights is on HOLD by the maintainer's decision (2026-09-20) independently of this gate.
