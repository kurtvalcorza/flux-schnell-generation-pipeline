# Release verification

`tutorials/pixart_sigma_generation_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
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
  generator's documented rewrites; the inline `MANIFEST`, `BASE_MANIFEST` and `SCORER_MANIFEST` equal to the three
  committed snapshot manifests and the inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook
  byte-identical (on LF) to `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with
  its restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before staging), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the components
  revision `2c17b4e8…` and the scorer revision `1a25a446…` are the only other 40-hex commits the documents may name);
- the profile-specific public-API calls (`stage_missing_files` / `stage_missing_base_files` /
  `stage_missing_scorer_files` with `allow_download=True`, the three `verify_*_snapshot` calls,
  `PixArtSigmaPipeline.from_pretrained(weights_dir=..., base_dir=..., use_lora=True)`, `fetch_sample_dataset` from
  the pinned cache path, `load_byod_dataset`, `dataset_manifest`, `write_dataset_csv`, `validate_dataset` with the
  refusal probes, `pipe.encode_prompts` and `pipe.release_text_encoder`, `pipe.evaluate` and `pipe.generate` +
  `score_generations` + `real_photo_baseline` on the frozen model, `pipe.adapt` with its explicit hyperparameters,
  `pipe.evaluate` after adaptation with the guaranteed assertions (kept-epoch validation loss ≤ frozen; re-scored validation loss matches the history), the new-prompt generation, `pipe.save_artifact`,
  `PixArtSigmaPipeline.from_artifact` + `import_prompt_cache` and the reload-parity assertion, and the provenance
  fields `safetensors_only: True`, `remote_code_executed: False` and the data base URL), the six expected `outputs/`
  paths, the learner-facing statements (three pinned snapshots, the encoder does not fit beside a training graph,
  generation has no ground truth, denoising loss, real-photo ceiling, not a human judgement, Open RAIL++-M,
  sample-sanity, CC0) and the gated-off BYOD default; forbidden patterns (credential-in-URL, any `git clone` /
  `github.com` / repository import on the primary path, a mutable `revision='main'`, direct `huggingface_hub` /
  `safetensors` / `urllib` / `diffusers` / `transformers` / `peft` / `T5EncoderModel` / `CLIPModel` use,
  `torch.load(` / `pickle.load` / `Unpickler`, `torch.no_grad(` / `torch.inference_mode(` / `.backward(` /
  `pipe.transformer(` **outside the carried module cells**, `trust_remote_code=True`, `add_adapter(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_samples.py`, `tests/test_adaptation.py` (stub transformer, skipped without
torch), `tests/test_role_helpers.py`, `tests/test_import_boundary.py`, `tests/test_notebook_parity.py`; temporary
manifests, synthetic images, an injected fetcher, no weights and none of `diffusers`, `transformers` or `peft`).
These are source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab GPU runtime (T4 or better, ≥ 15 GB) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | WSL workstation GPU (12 GB), sequential cell executor with a `google.colab` shim, pre-staged pins and snapshots | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new GPU runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshots `weights/pixart-sigma-xl-2-512-ms/`, `weights/pixart-sigma-t5-vae/`, `weights/clip-vit-b-32-laion2b/`
   or the data cache `weights/inat-birds/` (the standalone path writes the three manifests itself, stages all 24
   listed files from the Hub — about 22.4 GB — and fetches the 60 pinned photographs, so none of the directories may
   be seeded); the runtime needs about 25 GB of free disk and a GPU of at least 15 GB;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `STEPS = 20`, `GUIDANCE_SCALE = 4.5`, `IMAGES_PER_PROMPT = 2`, `EPOCHS = 4`,
   `LEARNING_RATE = 1e-4`, `BATCH_SIZE = 1`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `diffusers==0.40.0`, `transformers==5.17.0`,
   `peft==0.21.0`, `torchao==0.18.0`, `accelerate==1.15.0`, `tokenizers==0.23.2`, `sentencepiece==0.2.2`, `protobuf==7.36.2`,
   `safetensors==0.8.0`, `huggingface-hub==1.32.0`, `numpy==2.5.3`, `pillow==11.3.0` (an interpreter restart after
   the install is expected where the runtime's preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `PixArtSigmaPipeline`, `build_transformer`,
     `lora_parameter_names`, the three `verify_*_snapshot` and `stage_missing_*` functions, `validate_inputs`,
     `validate_dataset`, `validate_prompts`, `preprocess_image`, `fetch_corpus`, `fetch_sample_dataset`,
     `build_sample_dataset`, `split_dataset`, `load_byod_dataset`, `write_dataset_csv`, `dataset_manifest`,
     `sample_prompts`, `ClipScorer`, `score_generations`, `real_photo_baseline`) with no import of the repository
     package;
   - the inline manifests asserted against the module's constants, then the three staging calls reporting 3 + 12 + 9
     entries fetched at the immutable revisions and the three verifications reporting 3 / 12 / 9 verified files;
   - the model cell loading the transformer in float16 with the untrained LoRA attached (603 base tensors,
     610,856,096 parameters; 448 LoRA tensors, 4,128,768 parameters), the VAE in float32, the tokenizer and the
     scheduler config, with `source` "local-snapshot (three manifests verified; safetensors only)";
   - the dataset manifest with 36 / 12 / 12 records, six distinct captions, `outputs/pixart_sigma_generation_sample_captions.csv`
     written, and three refusals (missing caption, 200 px image, duplicate id);
   - `pipe.encode_prompts` reporting the encoder loaded in float16 on `cuda`, 8 prompts encoded (six captions, the
     new prompt, the empty negative prompt), 0 truncations, and `release_text_encoder` returning `True`;
   - the frozen model's held-out denoising MSE on the validation and test records, twelve 512² generations scored by
     CLIP with `outputs/pixart_sigma_generation_frozen_grid.jpg` written, and the real-photo ceiling on the test
     photographs;
   - `pipe.adapt` printing epoch 0 as the frozen model, 4,128,768 trainable of 614,984,864 parameters, 144 steps,
     and a four-epoch history with the validation denoising MSE at the kept epoch below the frozen model's;
   - `pipe.evaluate` on the validation and test records with the paired comparison, the assertions that the kept
     epoch's validation loss is no higher than the frozen model's and that the re-scored validation loss matches the
     history within 10⁻⁴ (the test change is printed as an observation), the adapted generations scored with
     `outputs/pixart_sigma_generation_adapted_grid.jpg` written, and `outputs/pixart_sigma_generation_evaluation_report.json`;
   - the new prompt rendered twice and CLIP-scored;
   - `pipe.save_artifact` writing `outputs/pixart_sigma_generation_adapter/{adapter.safetensors,manifest.json}`
     (448 tensors), and `PixArtSigmaPipeline.from_artifact` reloading it into a fresh pipeline that adopts the
     exported prompt cache, with the held-out MSE and a seeded generation matching the adapted pipeline (the cell
     asserts `denoising_mse_diff < 1e-6` and `mean_abs_pixel_diff < 1.0`);
   - `outputs/pixart_sigma_generation_result.json` written with `NOTEBOOK_SOURCE`, the three identities and
     licences, the provenance block (`safetensors_only: true`, `remote_code_executed: false`, the data base URL),
     the runtime versions, the comparison and the reload parity;
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
| `pixart_sigma_generation_colab.ipynb` (`E2E`) | `4862a6e` / `99829b51` | 2026-09-19 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-pixart-sigma-generation` v3; image `torch 2.10.0+cu128` before the pinned install, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0` after, Python 3.12.13, `cuda`) | **PASSED** — 11/11 code cells ok (1 restart after install cell); 117 files, 22443 MB fetched into a clean runtime (the three Hub snapshots + the pinned photographs); comparison held-out denoising MSE (test) frozen 0.105983 → adapted 0.105838 (best epoch 3, validation 0.109093 → 0.108941; the guaranteed validation inequality asserted, the test change printed as an observation), CLIP prompt similarity / label accuracy / reference similarity of 12 generations frozen 26.79 / 0.50 / 65.57 → adapted 28.01 / 0.50 / 68.95 against the real-photo ceiling 30.25 / 0.917 / 88.66 (sample-sanity, not a quality claim); reload parity denoising_mse_diff: 0.0, mean_abs_pixel_diff: 0.0; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-pixart-sigma-generation/v3/evidence/` in the workspace |
| `pixart_sigma_generation_colab.ipynb` | generated, pre-commit | 2026-09-19 | Local pre-flight harness (WSL, CPython 3.12.3, CUDA RTX 5070 Ti laptop, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/pixart_sigma_generation_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/pixart_sigma_generation_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-19 | `4862a6e` / `99829b51` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-pixart-sigma-generation` v3; image `torch 2.10.0+cu128` before the pinned install, `torch 2.14.0+cu130`, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0` after, Python 3.12.13, `cuda`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution); the three snapshots staged and digest-verified by the notebook, the pinned photographs fetched by the notebook | 1582.8 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 117 files, 22443 MB fetched into a clean runtime (the three Hub snapshots + the pinned photographs); comparison held-out denoising MSE (test) frozen 0.105983 → adapted 0.105838 (best epoch 3, validation 0.109093 → 0.108941; the guaranteed validation inequality asserted, the test change printed as an observation), CLIP prompt similarity / label accuracy / reference similarity of 12 generations frozen 26.79 / 0.50 / 65.57 → adapted 28.01 / 0.50 / 68.95 against the real-photo ceiling 30.25 / 0.917 / 88.66 (sample-sanity, not a quality claim); reload parity denoising_mse_diff: 0.0, mean_abs_pixel_diff: 0.0; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-pixart-sigma-generation/v3/evidence/` in the workspace |
| 2026-09-19 | generated, pre-commit | Local pre-flight harness (WSL, CPython 3.12.3, `torch 2.14.0+cu130`, RTX 5070 Ti laptop 12 GB, `diffusers 0.40.0`, `transformers 5.17.0`, `peft 0.21.0`) | Default sample path (stage → verify the three snapshots → load transformer + LoRA / VAE → pinned-photo fetch from the cache → validate → refusal probes → encode prompts + release encoder → frozen evaluation, generations and real-photo ceiling → LoRA adapt → paired evaluation → new prompt → export → reload); the three snapshots and the 60 photographs were pre-staged, so every staging call fetched 0 entries and the run verified 24 files by digest | 1994.8 s | **PASSED** — 11/11 code cells; probes refused; frozen test denoising MSE 0.105939 → adapted 0.105800 (best epoch 3; printed, not asserted); encoder peak on the 12 GB GPU; adapter 448 tensors; reload parity identical. Pre-flight; hosted clean-runtime run still required |

## Current status

**Release-grade.** The `E2E` notebook blob `99829b51` (committed at `4862a6e`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-19 (11/11 ok (1 restart after install cell), 1582.8 s, 117 files, 22443 MB fetched and digest-verified inside the notebook, three snapshots) with no repository checkout — the REL1/REL10 supported-runtime evidence this file gates on. The local pre-flight rows above are what preceded it and remain history. Any later change to the carried modules or to the notebook produces a new blob, and the registry returns to **Candidate** until a clean run of that blob is recorded here.
