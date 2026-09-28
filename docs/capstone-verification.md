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
| — | — | — | — | — | no hosted execution recorded yet |
