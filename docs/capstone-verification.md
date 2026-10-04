# Capstone verification record

`tutorials/DIMER_Bosch_Synthetic_Defect_Augmentation_Capstone.ipynb` (`E2E`, `GUIDED`, DIMER Notebook Specification 2.2) is a **candidate**. A fresh Colab T4 `Run all` of `9eae6c6` passed on 2026-09-29, and a Colab CLI sequential execution of `439c34f` (notebook blob `84061cc7`, uv isolated environment) passed on a fresh Colab T4 on 2026-10-04 (both recorded below); promotion beyond candidate is a maintainer decision and is not made by this record. Static checks, unit tests and CPU runs of individual stages are not execution evidence (REL8).

## Automatic coverage (every pull request)

- `python tools/build_capstone_notebook.py --check`: the committed notebook equals the generator output from `tools/capstone/`.
- `tests/test_capstone.py`: notebook JSON, metadata (profile, mode, spec, standalone), embedded scripts equal to their sources, both locks embedded (as raw strings, byte-identical), fully pinned and SHA-256 hashed, nothing installed into the kernel (uv from a hash-pinned wheel, `--require-hashes --only-binary :all:`, every stage in a locked venv, kernel `PYTHON*` settings dropped), no cell line over 2,000 characters, code cells compile, no placeholders, no persisted outputs, no clone / repository install / credential prompt / pickle load / `extractall` on the default path, form fields, collapsed infrastructure cells, guided-layer markers, citations resolving to references, pinned model revisions and the FLUX manifest digests; and the core's behaviour: archive traversal, absolute paths, symlinks, unsupported files and size ceilings refused; digest mismatch, LFS pointers and interrupted downloads handled; corrupt images recorded; group leakage detected; label-conflict duplicates excluded; insufficient classes and budgets refused; BYOD manifest rejections; bounded prompts and recorded fallbacks; candidate eligibility including non-finite pixels; generation shortfall; the matched slot schedule; undefined metrics; the stratified group bootstrap; frozen-record tampering; artifact digest, file-set, shape and format failures.

## What CI cannot cover

Phi-4 and FLUX need a CUDA GPU with `bitsandbytes`, about 45 GB of weights and the two locked environments, so the model stages run only in a hosted runtime. The combined run on one T4 (including host-memory fit and time) has not yet been measured.

## Pre-release evidence gathered while writing the notebook (not a notebook run)

Recorded 2026-09-28 in a CPU container, Python 3.12, `torch 2.14.0+cpu`, NumPy 2.5.3, Pillow 11.3.0:

- The archive `SDI_DATASET_v1.zip` from the LFS media host at commit `c6e0afe6…` was 1,647,097,112 bytes with SHA-256 `d33ea340…fa7c`, matching the pin; 18,841 entries (18,807 JPEGs, 34 directories), no symlinks, 1,644,493,674 expanded bytes, largest member 232,910 bytes.
- Measured product counts: A 6,184 / 340 / 108, B 6,199 / 167 / 670, C 4,760 / 121 / 258 (normal / scratches / spots); the published notes list 6,250 normal images per product. No undecodable file. Formats: A grayscale 423 × 423; B and C include 301 × 301, 680 × 680 grayscale and 256 × 256 RGB.
- The source assigns only defect images to train/val/test; normal images have no official split. No specimen or session identifiers are supplied.
- Product A has 5 exact decoded-pixel duplicate pairs, each labelled `scratches` in one copy and `spots` in the other (several across official partitions); these 10 images are excluded as label conflicts. 18 further pairs have thumbnail correlation 0.98–0.99 (consecutive normal frames); 73 pairs fall within 0.005 below the 0.98 cut, which is a declared threshold, not a natural gap.
- The split manifest SHA-256 `8b5ef942…65aa` was reproduced on repeated runs: train 3,710 / 201 / 61, validation and test 1,237 / 67 / 21 each; budget 128 / 64 / 32 selected.
- The data, feature, fit, test, export and reload stages, and every controller cell of the notebook in an IPython kernel, ran end to end on the real archive with stand-in scripts in place of the two GPU stages; reload parity was exact. The stand-ins make no scientific result.
- A BYOD zip built from 800 Bosch images with renamed classes ran through every stage; a manifest without a `class` column was refused with the missing column named.

## Review of `eb7f0d5` (findings BSA-01 to BSA-06) and fixes

A notebook review of `eb7f0d5` (verdict: needs revision) reported three major and three smaller findings, each with an acceptance check, and supplied a probe package that ran the notebook's extracted stage scripts with stand-ins (authored captions instead of Phi-4, NumPy pixels instead of FLUX, an adaptive-average feature module instead of ResNet-18). The fixes are in `tools/capstone/` and the generator; the notebook was regenerated, not hand-edited.

| Finding | Fix | Offline check (post-fix) |
|---|---|---|
| BSA-01 the freeze did not bind held-out pixels, overlap signatures or stage code | `load_verified_gray` re-checks every manifest-listed image's file SHA-256 and decoded-pixel digest before the feature, test, export and reload stages use it; the freeze also pins `audit/signatures.npz` and all ten stage sources (`STAGE_SOURCES`). The evaluate, export and reload scripts are therefore saved in Section 8, before the fit | reviewer's mutations re-run: changed test image, signature file, preprocessing script and split-manifest row each refused before scoring (exit 2); unchanged inputs pass |
| BSA-02 reload checked numbers, not meaning | `check_artifact_semantics` binds classes, order, `class_index`, decision rule, preprocessing, backbone, experiment record, selection and head digest to the frozen record; `replay_check` requires identical logits **and** identical named decisions; the reload reference is digest-bound in the manifest | reversed class list, inconsistent indices, duplicate names and a changed preprocessing field each refused (exit 2); unchanged artifact passes with max logit difference 0.0 and identical decisions |
| BSA-03 an exact-budget BYOD set crashed the new-image preview | fixed-header `write_new_predictions`; `new_image_inference.status` (`ran`, `not_run_no_inputs`, `not_run_all_inputs_rejected`) in `reload_parity.json`; the data stage reports spare images per class and warns early | the reviewer's 376-image set (214 / 108 / 54) ran all nine stages; reload passed and recorded `not_run_no_inputs` with a header-only CSV |
| BSA-04 optional activities raised `KeyError` without canonical C | both activity cells skip with an explanation when C was not run; the default-on activity is explained | exact cell with a no-C record: no exception, no fit started |
| BSA-05 BYOD records repeated Bosch facts | limitations, attribution, dataset-change text and the metrics evidence line are built from the active data audit; Bosch licence text only on the Bosch path | notebook-kernel harness on a BYOD zip: no Bosch terms or counts in `limitations.md` or `ATTRIBUTION.md` |
| BSA-06 96 tokens was described as truncation | `stop_reason` (`end_of_sequence`, `max_new_tokens`, `other`) replaces `hit_token_limit`; the table and prose use it | source check only; the real tokenizer's stop behaviour needs the hosted run |

User-visible changes: `new_image_rejections.json` is replaced by `new_image_inference` inside `reload_parity.json`; `caption_records.jsonl` carries `stop_reason`; the artifact manifest carries `reload_reference_sha256`; reload refuses artifacts whose metadata disagrees with the frozen experiment; the evaluate, export and reload scripts are saved in Section 8.

Evidence boundary: the reviewer's probe scripts were re-run with expectations inverted, and the notebook's own cells were executed in an IPython kernel on the real Bosch archive and on a BYOD zip, both with stand-ins for the two GPU stages and a CPU backbone. None of this is a hosted run, and no Phi-4 or FLUX output was produced.

## 2026-10-03 uv isolated environment

Notebook blob `c64f21cc` (passed on Colab T4 at `9eae6c6`, recorded below) → `84061cc7`. Status: **Candidate**. The new blob passed a Colab CLI sequential execution on a fresh T4 at `439c34f` on 2026-10-04 (recorded below), with every printed result equal to the 2026-09-29 run.

- uv now comes from a pinned wheel (`uv 0.12.19`, size and SHA-256 checked) unpacked into `work/sdi_capstone/bin/`, instead of `pip install uv` into the kernel. Nothing is installed into the notebook kernel.
- Both locks (`tools/capstone/locks/lab.lock`, `phi4.lock`) carry SHA-256 hashes from `uv pip compile --generate-hashes --only-binary :all:`, with the same 74 / 51 pins as before. They install with `--require-hashes --only-binary :all: --no-deps` into a uv-managed CPython 3.12.12, which replaces whatever Python 3.12 the host offered (3.12.3 on the recorded run).
- `PYTHONPATH`, `PYTHONHOME` and `PYTHONSTARTUP` are dropped from the environment of uv and of every stage process.
- The stage scripts and the shared core are unchanged byte for byte, so the default results are expected to match the 2026-09-29 run (for example, mean test macro-F1 for C − B +0.0443). Differences we expect: the environment build times, and the isolated interpreter's patch version.
- User-visible: the notebook runs on Linux x86_64 only (Colab, Kaggle, Linux Jupyter), and it refuses other hosts with a message.

## Verification procedure

1. Resolve the exact commit and notebook blob under review; confirm CI is green.
2. Open that revision in a fresh Colab runtime with a T4 GPU, no repository checkout and an empty working directory.
3. Run all with the default form fields (`DATA_SOURCE = 'bosch'`, `SEEDS = '17,29,43'`, `ALLOW_PHI4_REMOTE_CODE = True`, `DELETE_MODEL_WEIGHTS_AFTER_USE = True`).
4. Confirm: no prompt, upload, restart or credential; the split reports `matches the reference audit: True` (or record why not); both environments see the GPU; Phi-4 and FLUX digests verify; generation status; the three arms and three seeds; the frozen record verifies before the test stage; `reload parity ... PASS`; the terminal summary.
5. Keep the executed notebook and `outputs/bosch_sdi_capstone/` (`run_summary.json` holds times, peak child memory, peak GPU memory, lowest available host memory and disk per stage).
6. Record the row below. Release blockers to close: acquisition reliability from Colab, host-memory fit of the 4-bit loads on standard Colab RAM, total time against the 90-minute model-time target, disk use, and the environments' GPU visibility on the Colab driver.

## Recorded executions

| Date | Commit / notebook blob | Runtime | Path | Wall time | Outcome |
|---|---|---|---|---|---|
| 2026-10-04 | `439c34f` (notebook blob `84061cc7`; executed file SHA-256 `d2683ad3…`) | Colab CLI 0.7.4, fresh Colab Tesla T4 15,360 MiB, driver 580.82.07, 12.7 GiB host RAM, kernel Python 3.13.15 (environments uv-managed CPython 3.12.12), 207.5 GB free disk | Colab CLI sequential execution (`colab exec`, one kernel), not a browser `Run all`; default path only, form fields at their defaults | 2,106 s (35 min) session wall time | **PASSED**: 41 of 41 code cells in order, no errors; every printed result equal to the 2026-09-29 run (C − B +0.0443). Details in the section below. Evidence: `docs/execution-evidence/2026-10-03/DIMER_Bosch_Synthetic_Defect_Augmentation_Capstone_439c34f_colab-cli-t4.ipynb` |
| 2026-09-29 | `9eae6c6` (notebook SHA-256 `660e685a…`) | Google Colab, Tesla T4 15,360 MiB, driver 580.82.07, 12.7 GiB host RAM, kernel Python 3.13.15 (environments Python 3.12.3), 202 GB free disk | Default `Run all` in a fresh runtime, no checkout, form fields at their defaults (the synthetic-fraction activity is on by default) | 2,769 s (46 min) from the configuration cell | **PASSED**: 41 of 41 code cells, sequential, no errors; source identical to the PR head. Details in the section below. Evidence: `docs/execution-evidence/2026-09-29/DIMER_Bosch_Synthetic_Defect_Augmentation_Capstone_9eae6c6.ipynb` |
| 2026-09-28 | `eb7f0d5` (notebook SHA-256 `415e1083…`) | Google Colab, Tesla T4 15,360 MiB, driver 580.82.07, 12.7 GiB host RAM, kernel Python 3.13.15 (both environments resolved Python 3.12.3), 202 GB free disk | Default `Run all`, fresh runtime, no checkout | about 31 min to the crash | **FAILED in Section 6 (FLUX), host memory.** Completed: both environments (lab 163.3 s: torch 2.14.0+cu126, transformers 5.17.0, diffusers 0.40.0, bitsandbytes 0.50.2; phi4 138.9 s: torch 2.6.0 cu124, transformers 4.48.2, bitsandbytes 0.45.5; both saw the T4); data stage 160.2 s (archive downloaded in 67.1 s and verified, 18,807 files, split SHA-256 matched the reference, peak child RSS 314 MiB); real features 35.2 s on CUDA; preview 7.1 s, reproducing the CPU validation macro-F1 of arm A exactly (0.6393 / 0.6295 / 0.6349); Phi-4 414.0 s (4-bit load 55.4 s, peak GPU allocated 7.09 GiB, peak child RSS 5,263 MiB, 9 descriptions); prompts 0.5 s (4 of 6 Phi-4-guided); FLUX staged and verified all 23 files. The Jupyter log shows the kernel restarted (`AsyncIOLoopKernelRestarter`) while the text encoders loaded: `T5EncoderModel.from_pretrained(torch_dtype=float16)` without `device_map` converts the bfloat16 checkpoint to float16 in host RAM (about 9.5 GB) before `.to("cuda")`, against about 9.9 GB available. Reproduced on CPU with a smaller bfloat16 T5 (anonymous memory grew by the whole converted model). Fix: every model load in the FLUX stage passes `device_map="cuda"`, which converts and moves one tensor at a time; `tests/test_capstone.py::test_gpu_model_loads_stream_to_the_device` guards it. The earlier Kaggle runs of this loading code had about 30 GB of host RAM, which is why it had not surfaced. Fixed in `0cef303`. Evidence: `docs/execution-evidence/2026-09-28/DIMER_Bosch_Synthetic_Defect_Augmentation_Capstone_eb7f0d5_failed.ipynb` |

## Colab CLI execution of `439c34f` — 2026-10-04

- **File:** `docs/execution-evidence/2026-10-03/DIMER_Bosch_Synthetic_Defect_Augmentation_Capstone_439c34f_colab-cli-t4.ipynb`, SHA-256 `d2683ad34dadaaa4c562776fc8f41e476299f8c8024acead0ef20bd91e0749eb`, copied byte for byte from the CLI's output notebook.
- **Source:** fetched from `raw.githubusercontent.com` at `439c34f56396aa21a3fdd8a5a22ecd827a43bd94`; git blob `84061cc7dfdfe270cd75840c1d48d22d198d3fba` checked before the VM was allocated.
- **Executor:** Colab CLI 0.7.4, fresh Colab Tesla T4 (15,360 MiB), driver 580.82.07, 12.7 GiB host RAM; session started 2026-10-04 01:39 UTC and stopped after the run.
- **Execution:** 41 of 41 code cells, in order (the CLI log shows `Executing cell 1/41` to `41/41`), no error outputs. The CLI does not set execution counts.
- **Environments:** lab: CPython 3.12.12, torch 2.14.0+cu126, transformers 5.17.0, diffusers 0.40.0, bitsandbytes 0.50.2, built in 133.9 s; phi4: CPython 3.12.12, torch 2.6.0 (CUDA 12.4), transformers 4.48.2, bitsandbytes 0.45.5, built in 99.8 s. Both saw the T4.
- **Stages:** data 114.4 s (archive 29.3 s, digest verified, split `8b5ef942…` matches the reference); real features 29.0 s; preview 7.0 s; Phi-4 303.7 s (4-bit load 55.9 s, 7.09 GiB); prompts 0.3 s (4 of 6 Phi-4-guided); FLUX 1,330.3 s (23 files verified, encoders 43.9 s at 11.08 GiB, transformer 100.9 s, 64 candidates in 431.3 s, all eligible); synthetic features 8.8 s; fit + freeze 11.7 s; test 17.9 s; export 4.8 s; reload 5.1 s. Lowest host memory available 8,909 MiB (FLUX).
- **Results:** test macro-F1 mean A 0.6472, B 0.6427, C 0.6870, majority 0.3219; **C − B +0.0443** (per seed +0.0415, +0.0559, +0.0355), interval [+0.0193, +0.0671]; B − A −0.0045, interval [−0.0244, +0.0161]; reload parity max logit difference 0.0, replayed decisions identical; 6 new images scored; synthetic-fraction activity C at 0.25: 0.635 / 0.638 / 0.631.
- **Comparison with the 2026-09-29 run of `9eae6c6`:** `difflib` over the normalised printed lines of both notebooks (338 lines each; clock times, durations and memory/disk readings stripped). Every metric line is identical: the validation curves of all arms and seeds, the export choice, the test contrasts, the reload parity and the new-image scores. The remaining differences are all explained:
  - the environments' Python is 3.12.12 instead of 3.12.3, the intended change in this revision;
  - free disk at start 207.5 GB instead of 202.2 GB, and the stage `seconds` fields (host-dependent timing);
  - the environment cell's return value is 1282 instead of 1099: it is the byte count of `environments.json`, which now also stores the uv version and wheel digest;
  - the frozen record SHA-256 is `b3d7cd86…` instead of `1faf8309…`: the record embeds `features_stage.json`, whose `seconds` field changed (26.0 s against 49.9 s), so the hash differs while the 32 pinned inputs and all results agree;
  - the Hugging Face "unauthenticated requests" warning (stderr) is printed one line earlier relative to a `fetching` line (stdout/stderr interleaving).
- **Journeys:** default path and the synthetic-fraction activity **passed**; the human-review extension and BYOD were skipped by their default flags (not assessed).
- **Evidence boundary:** Colab CLI sequential execution in one kernel, not a browser `Run all`; forms were not rendered and no upload or download dialog was exercised; default path only.

## Maintainer-supplied Colab execution of `9eae6c6` — 2026-09-29

- **File:** `docs/execution-evidence/2026-09-29/DIMER_Bosch_Synthetic_Defect_Augmentation_Capstone_9eae6c6.ipynb`, SHA-256 `660e685a0086811f318c46d91ec24be9ae058e7d51efa7e3dbce6aa29989014c`, copied byte for byte from the upload.
- **Source match:** all 92 cells have the same ids, order and source as the notebook at `9eae6c6`; no form field was changed.
- **Execution:** 41 of 41 code cells, execution counts 1 to 41, no error outputs.
- **Runtime:** Colab Tesla T4 (15,360 MiB), driver 580.82.07, 12.7 GiB host RAM.
  - Lab environment: torch 2.14.0+cu126, transformers 5.17.0, diffusers 0.40.0, bitsandbytes 0.50.2; built in 145.7 s.
  - Phi-4 environment: torch 2.6.0 (CUDA 12.4), transformers 4.48.2, bitsandbytes 0.45.5; built in 146.7 s.
  - Both environments saw the T4.

| Stage | Time | Result | Peak child RSS | Peak GPU in use | Lowest host memory available |
|---|---|---|---|---|---|
| Data | 136.4 s | archive downloaded in 43.4 s and digest-verified; 18,807 files; split SHA-256 `8b5ef942…` matched the reference | 314 MiB | 0 | 9,700 MiB |
| Real features + preview | 55.5 s + 7.6 s | validation macro-F1 of arm A 0.6393 / 0.6295 / 0.6349, identical to the CPU runs | 1,386 MiB | 841 MiB | 9,004 MiB |
| Phi-4 | 358.3 s | 4-bit load 56.9 s, peak GPU allocated 7.09 GiB; 9 descriptions, 35–69 tokens, all `end_of_sequence` | 5,265 MiB | 7,625 MiB | 7,324 MiB |
| Prompts | 0.3 s | 4 of 6 Phi-4-guided, 2 recorded fallbacks | 107 MiB | 0 | 9,966 MiB |
| FLUX | 1,853.8 s | 23 files verified; encoders 44.6 s (peak 11.08 GiB GPU, host RSS 5.88 GiB); 4-bit transformer 101.5 s (6.06 GiB); 64 candidates in 357.0 s, all eligible. About 1,350 s of the stage was download and digest verification | 6,485 MiB | 11,529 MiB | 6,965 MiB |
| Synthetic features | 8.6 s | highest candidate-to-training correlation 0.966; none flagged | 1,071 MiB | 841 MiB | 9,233 MiB |
| Fit + freeze | 12.0 s | 32 inputs pinned, record SHA-256 `1faf8309…`; export choice C, seed 17, epoch 29 | 941 MiB | 105 MiB | 9,376 MiB |
| Test | 20.4 s | frozen record verified; comparison valid | 1,312 MiB | 841 MiB | 9,011 MiB |
| Export + reload | 5.2 s + 6.4 s | semantic checks passed; max logit difference 0.0; replayed decisions identical; 6 new images scored | 815 MiB | 0 | 9,467 MiB |

**Results (tutorial evidence from one grouped split of product A and one generated pool; not a benchmark or production result):**

| Arm | Test macro-F1 seed 17 / 29 / 43 | Mean | Balanced accuracy (mean) | Accuracy (mean) |
|---|---|---|---|---|
| Majority baseline | 0.322 / 0.322 / 0.322 | 0.322 | 0.333 | 0.934 |
| A real-only | 0.651 / 0.641 / 0.650 | 0.647 | 0.830 | 0.913 |
| B conventional augmentation | 0.645 / 0.639 / 0.644 | 0.643 | 0.837 | 0.911 |
| C synthetic augmentation | 0.687 / 0.695 / 0.679 | 0.687 | 0.833 | 0.931 |

- **C − B:** +0.0443 (per seed +0.0415, +0.0559, +0.0355), approximate 95% interval [+0.0193, +0.0671] from the paired group bootstrap, conditional on this split and this pool.
- **B − A:** −0.0045, interval [−0.0244, +0.0161].
- **Where C's gain comes from:** for the canonical seed, mainly fewer false alarms on `normal` (recall 0.935 against 0.917 for B) and higher spot precision (0.254 against 0.192). Spot recall is unchanged at 15 of 21. Arm C's seed 29 kept epoch 2.
- **Synthetic-fraction activity (validation only):** C at probability 0.25 scored 0.635 / 0.638 / 0.631, against 0.662 / 0.666 / 0.662 at 0.5.

**Observations for learners.** This is the domain mismatch and label noise the notebook asks learners to inspect:

- several candidates contain rendered text, although every prompt ends with "no text";
- many show whole tables or benches in perspective rather than close-up texture;
- some intended labels are doubtful (for example `syn_spots_00` shows a line).

The human-review extension was not run.

**Journeys:**

| Journey | Verdict |
|---|---|
| Default path (acquisition, audit, preview, Phi-4, prompts, FLUX, three arms × three seeds, freeze, test, export, reload, records) | **Passed** |
| Synthetic-fraction activity (on by default) | **Passed** |
| Human-review extension | not assessed in this run |
| BYOD full run and BYOD check | not assessed in this run |

**Evidence boundary:** the saved outputs of the executed notebook were inspected and checked against the PR head; execution was not independently repeated here.

**Spec release blockers observed once:**

- acquisition from Colab;
- host-memory fit (lowest available 6,965 MiB);
- combined execution without a restart;
- GPU visibility in both environments.

Model time, excluding the FLUX download, was well under the 90-minute target.

**Still open:**

- a hosted BYOD run, including the exact-budget case;
- a human-review run;
- learner observation;
- the maintainer's promotion decision.
