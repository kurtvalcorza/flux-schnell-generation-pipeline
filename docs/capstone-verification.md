# Capstone verification record

`tutorials/DIMER_Bosch_Synthetic_Defect_Augmentation_Capstone.ipynb` (`E2E`, `GUIDED`, DIMER Notebook Specification 2.2) is a **candidate** until one fresh Colab T4 `Run all` of an exact notebook revision is recorded below. Static checks, unit tests and CPU runs of individual stages are not execution evidence (REL8).

## Automatic coverage (every pull request)

- `python tools/build_capstone_notebook.py --check`: the committed notebook equals the generator output from `tools/capstone/`.
- `tests/test_capstone.py`: notebook JSON, metadata (profile, mode, spec, standalone), embedded scripts equal to their sources, both locks embedded and fully pinned, code cells compile, no placeholders, no persisted outputs, no clone / repository install / credential prompt / pickle load / `extractall` on the default path, form fields, collapsed infrastructure cells, guided-layer markers, citations resolving to references, pinned model revisions and the FLUX manifest digests; and the core's behaviour: archive traversal, absolute paths, symlinks, unsupported files and size ceilings refused; digest mismatch, LFS pointers and interrupted downloads handled; corrupt images recorded; group leakage detected; label-conflict duplicates excluded; insufficient classes and budgets refused; BYOD manifest rejections; bounded prompts and recorded fallbacks; candidate eligibility including non-finite pixels; generation shortfall; the matched slot schedule; undefined metrics; the stratified group bootstrap; frozen-record tampering; artifact digest, file-set, shape and format failures.

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
| 2026-09-28 | `eb7f0d5` | Google Colab, Tesla T4 15,360 MiB, driver 580.82.07, 12.7 GiB host RAM, kernel Python 3.13.15 (both environments resolved Python 3.12.3), 202 GB free disk | Default `Run all`, fresh runtime, no checkout | about 31 min to the crash | **FAILED in Section 6 (FLUX), host memory.** Completed: both environments (lab 163.3 s: torch 2.14.0+cu126, transformers 5.17.0, diffusers 0.40.0, bitsandbytes 0.50.2; phi4 138.9 s: torch 2.6.0 cu124, transformers 4.48.2, bitsandbytes 0.45.5; both saw the T4); data stage 160.2 s (archive downloaded in 67.1 s and verified, 18,807 files, split SHA-256 matched the reference, peak child RSS 314 MiB); real features 35.2 s on CUDA; preview 7.1 s, reproducing the CPU validation macro-F1 of arm A exactly (0.6393 / 0.6295 / 0.6349); Phi-4 414.0 s (4-bit load 55.4 s, peak GPU allocated 7.09 GiB, peak child RSS 5,263 MiB, 9 descriptions); prompts 0.5 s (4 of 6 Phi-4-guided); FLUX staged and verified all 23 files. The Jupyter log shows the kernel restarted (`AsyncIOLoopKernelRestarter`) while the text encoders loaded: `T5EncoderModel.from_pretrained(torch_dtype=float16)` without `device_map` converts the bfloat16 checkpoint to float16 in host RAM (about 9.5 GB) before `.to("cuda")`, against about 9.9 GB available. Reproduced on CPU with a smaller bfloat16 T5 (anonymous memory grew by the whole converted model). Fix: every model load in the FLUX stage passes `device_map="cuda"`, which converts and moves one tensor at a time; `tests/test_capstone.py::test_gpu_model_loads_stream_to_the_device` guards it. The earlier Kaggle runs of this loading code had about 30 GB of host RAM, which is why it had not surfaced |
