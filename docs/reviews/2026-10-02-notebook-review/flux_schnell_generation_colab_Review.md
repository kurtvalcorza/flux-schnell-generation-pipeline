# FLUX.1 [schnell] Text-to-Image QLoRA E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 3 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/flux-schnell-generation-pipeline`  
**Notebook:** `tutorials/flux_schnell_generation_colab.ipynb`  
**Reviewed commit:** `dea80fa6326aab23d327f7a27b4c29efcabcbcf2` (`main`, confirmed with `gh api repos/kurtvalcorza/flux-schnell-generation-pipeline/commits/main`)  
**Notebook Git blob:** `522268125c9147c498e6ad873d22eeb33f42d8b5`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-20 (commit `d4fdf72`); the notebook and the carried modules have not changed since.  
**Finding prefix:** `FS`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main`. The notebook declares 2.0.

## Executive assessment

The engineering is careful. The notebook carries its three modules verbatim, stages a 33.7 GB snapshot from an ungated mirror and verifies every byte against a committed manifest, fetches 60 digest-pinned CC0 photographs, keeps the 9.7 GB of text encoders and the 4-bit transformer off the GPU at the same time, measures a paired held-out flow-matching loss with identical noise before and after a bounded QLoRA run, and reloads a safetensors adapter into a fresh 4-bit pipeline with exact parity. The prose is honest that neither a training loss nor a CLIP score is image quality. A CPU run of the data stage in this review reproduced the recorded split exactly (36 / 12 / 12, dataset digest `9ca31704…`, the same three refusals).

| Measure | This review (CPU, data stage only) | Kaggle T4 record (blob `52226812`) |
|---|---|---|
| Code cells completed | carried cells 5, 7, 9 and Section 4 (cell 13); install and model cells not run | 11/11 on pass 2; pass 1 stopped at the install guard |
| Sample fetch and split | 60 photos, 36 / 12 / 12, digest `9ca31704…`, 85 s | identical split and digest |
| Real-photo reference similarity ("ceiling") | **57.7** with each photo excluded from its own reference (derived from the record) | printed **88.66** (each photo included in its own reference) |
| Frozen → adapted generations, reference similarity | not run | 71.95 → 75.01 |
| BYOD smallest single-caption dataset accepted | **6 images** (stated minimum 4 refused) | not run |
| GPU memory if BYOD is re-run as instructed | **≈19.3 GB projected** on a 15 GB T4 | after the reload: 7.48 GB resident |

Four problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (FS-M1).** The recorded run stopped at the install cell's stale-module guard and passed only after a restart; `docs/release-verification.md` calls the restart "expected", and the repository marks the blob `Release-grade` on that run.
2. **The "real-photo ceiling" is not a ceiling (FS-M2).** Its reference similarity compares each real photo with a mean that contains that same photo; excluding it, the real photos score about 57.7, below both the frozen (71.95) and the adapted (75.01) generations. Its CLIP prompt similarity (30.25) is also below both generated sets. The notebook tells the learner these are "the ceiling these numbers could reach".
3. **BYOD, followed as written, runs out of GPU memory (FS-M3).** Section 9 leaves a second 4-bit transformer resident in `reloaded`; the BYOD instruction re-runs from Section 4, and Section 5 then loads the encoders beside it.
4. **Guided layer largely absent (FS-M4).** Declared `GUIDED`, but there is no audience, how-to-use, roadmap, glossary, prediction prompt, checkpoint, troubleshooting or conclusion template, and 2,171 lines of carried modules sit in three unlabelled, uncollapsed cells.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. Prerequisites (cell 1, "Knowledge"): what a latent diffusion / flow-matching model does at inference, why a distilled few-step model has no classifier-free guidance, what 4-bit quantisation and a LoRA adapter change, why a training loss is not a quality score |
| Supported runtime | "a fresh supported **GPU** runtime (Google Colab T4 or better, or a Jupyter kernel with a CUDA GPU of at least 15 GB, bitsandbytes support and Python 3.12)"; about 36 GB of disk; "CPU-only runtimes are not supported" |
| Promised outcomes | Pinned install; carried package; mirror-staged snapshot digest-verified against a committed manifest, plus the CLIP scorer; 60 digest-pinned CC0 photos validated and split 36 / 12 / 12 with three refusals; prompts encoded and encoders released; 4-bit transformer load; frozen held-out flow-matching loss and CLIP-scored four-step generations "against the real-photo ceiling"; bounded QLoRA fine-tuning; paired comparison on identical inputs; a new prompt; safetensors adapter export, release and fresh 4-bit reload with parity; BYOD zip through the same contract |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; recorded generating revision `77567b7` |
| Release status | **`Release-grade`** (`STATUS.md`, `README.md`, `tutorials/README.md`, `docs/release-verification.md` Current status) |

### Evidence actually obtained

- **Source inspection.** All 25 cells (11 code; cells 5, 7, 9 are the carried `pipeline.py` 1,142 lines, `metrics.py` 137 lines, `samples.py` 892 lines). Also read: `src/…/pipeline.py` (`validate_dataset`, `load_transformer`, `release_transformer`, `encode_prompts`, `adapt`), `samples.py` (`fetch_corpus`, `build_sample_dataset`, `split_dataset`, `load_byod_dataset`, `dataset_manifest`), `metrics.py` (`score_generations`, `real_photo_baseline`); `tools/build_notebook.py` (`recorded_revision`, `load_context`, `main`) and the relevant parts of `tools/notebook_template.py`; `README.md`, `STATUS.md`, `tutorials/README.md`, `docs/release-verification.md`. The repository has no `AGENTS.md`; `docs/execution-evidence/` holds only the capstone notebook's runs.
- **Documented execution evidence.** `docs/release-verification.md` row 2026-09-20 and the workspace archive it cites (`.agent/backups/tier-c-build-2026-09-20/kaggle-flux/dimer-nb2-flux-schnell-generation/v2/evidence/`: `run_summary.json`, `executed.ipynb`, `outputs/…evaluation_report.json`): Kaggle Tesla T4, **the reviewed blob** (SHA-1 verified before execution), clean HF cache; pass 1 `RuntimeError: Core dependencies changed while older modules were loaded: cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3; protobuf: loaded=5.29.5, installed=7.36.2. Restart the runtime…`, `restarted_after_install_cell: true`, pass 2 11/11, 2,911.9 s total. No Colab run of this blob; no BYOD run; no optional experiment.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py`, Windows 11, CPU only, Python 3.12.14, torch 2.13.0+cpu, numpy 2.5.3, pillow 12.3.0 (shared read-only conda env; nothing installed). The install cell (3) was not run; the carried module cells were executed directly. No FLUX or CLIP weights were downloaded.
  - **Probes (87 s total):** P1 static structure; P2 provenance of the carried modules against the recorded revision; P3 cells 5, 7, 9 and 13 at defaults from an empty working directory (real fetch of the 60 photos); P4 cell 13's BYOD branch through a shimmed `google.colab.files.upload` with twelve archives; P5 the real-photo reference similarity with and without self-inclusion, from the recorded report, plus a synthetic identity check; P6 rerun-state checks in source; P7 GPU-memory arithmetic for the documented BYOD rerun, from the recorded run.
- **Not verified:** Sections 1, 3 and 5–9 (install, staging, encoding, 4-bit load, generation, fine-tuning, evaluation, export, reload) beyond the Kaggle record; any Colab run; the real upload dialog; BYOD beyond the validation/split stage; the optional experiments.
- **Learner observation:** none. No claim here is about measured learning effectiveness.

## 2. Separate judgments

- **Technical correctness:** strong supply-chain handling (mirror bytes checked against a committed manifest, refusal on the first mismatch, no pickle, no remote code) and a careful residency rule for the encoders and the transformer. Defects: the install pattern forces a restart (FS-M1); the residency rule does not cover the second pipeline Section 9 creates, so the documented BYOD rerun is projected to exceed the GPU (FS-M3); the exported provenance names a revision whose `pipeline.py` is not the one that runs (FS-m2); rerunning Section 6 or 7 for an optional experiment reuses the adapted weights and labels them frozen (FS-m3).
- **Scientific validity:** the paired held-out flow-matching loss (same latents, noise and noise levels before and after), validation-only epoch selection, an untouched test split, one photo per observer per species, and the explicit "not a human judgement" framing are all sound. The weak point is the reference the generations are read against: the "ceiling" is inflated by self-inclusion on one metric and below the generations on another (FS-M2). The notebook does not state pretraining overlap for iNaturalist photographs or run-to-run variability of the GPU fine-tune (FS-m4).
- **Promise fulfilment:** default-path promises are met on the documented run, except one-pass `Run all` (FS-M1) and the "real-photo ceiling" (FS-M2). BYOD's stated minimum is wrong and the documented route is projected to fail (FS-m1, FS-M3). The generated images the learner is asked to "compare by eye" are written to files but never displayed (FS-m5).
- **Learner experience:** precise, honest prose with "Look for" notes at Sections 4–8 and a careful limits section; but no guided layer (FS-M4), one expected-output note contradicts the run, and the interpretation pre-states the result (FS-m4).
- **Spec conformance:** unresolved applicable MUSTs — RUN1, RUN10, ENV6, REL2, REL11 (FS-M1); EVAL3 (FS-M2); DAT14, REL12 (FS-M3); DAT12, DAT19, VAL7 (FS-m1); OUT6 (FS-m2, see the finding); UX7 (FS-m3); DAT9, ENV8 (FS-m4). SHOULD deviations: EVAL10, GDL8 (FS-M2); GDL1–GDL7, GDL9–GDL14, UX8 (FS-M4); EXE2, UX10 (FS-m1); OUT7, OUT9 (FS-m2); GDL10, UX5 (FS-m3); GDL8, GDL14 (FS-m4); UX3, UX11 (FS-m5); EXE5 (FS-S2).

## 3. Promise and objective tracing

| Claim / objective | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| One-pass `Run all` | cell 3 in-kernel `pip install` + stale-module guard | Kaggle pass 1 `RuntimeError`, restart, pass 2 11/11 | Section 1 says the cell "stops with a restart instruction" | **Not met** (FS-M1) |
| Mirror-staged snapshot, digest-verified | cell 11 | Kaggle: 23 + 9 files fetched and verified, `cuda`, `source` names the mirror | clear | Met (documented) |
| 60 pinned photos, validation, 36 / 12 / 12 split, three refusals | cell 13 | P3: identical counts, digest and refusal messages to the Kaggle record; 0 species/observer pairs in two splits | "Look for" note matches | Met |
| Encode prompts, release encoders, then load 4-bit | cells 15, 17 | Kaggle: 12.15 GB → 0.35 GB → 6.52 GB, 380 LoRA tensors | clear | Met (documented) |
| Frozen held-out loss and CLIP-scored generations "against the real-photo ceiling" | cell 17, `real_photo_baseline` | Kaggle: reference similarity 71.95 vs "ceiling" 88.66; prompt similarity 31.08 vs 30.25 | told the real numbers are "the ceiling these numbers could reach"; "label accuracy below the real photographs' 1.0" (recorded 0.917) | **Misleading** (FS-M2, FS-m4) |
| Bounded QLoRA, explicit hyperparameters, validation selection | cell 19 → `adapt` | Kaggle: 9,338,880 trainable, 108 steps, best epoch 3 by validation loss | well explained | Met (documented) |
| Paired held-out comparison | cell 21 | Kaggle: test MSE 0.795 → 0.507; label accuracy 0.667 → 0.833 | careful "observation, not asserted" wording; grids not shown inline | Met; display gap (FS-m5) |
| New prompt, export, release, fresh reload with parity | cell 23 | Kaggle: 380 tensors, 37,404,064 bytes, parity 0.0 / 0.0 | clear | Met (documented) |
| Provenance export names the source revision | cell 3 `NOTEBOOK_SOURCE`, cell 23 result JSON | P2: recorded revision `77567b7`; its `pipeline.py` differs from the carried one | Interpretation: "proves that the recorded repository revision's pipeline modules…" | **Inaccurate** (FS-m2) |
| BYOD zip through the same contract, "at least four images, and at least one caption with three or more images" | cell 13 BYOD branch | P4: 4 and 5 images of one caption refused; 6 accepted; one-caption-per-image refused | contract states 4 | **Not met as stated** (FS-m1); rerun route projected OOM (FS-M3) |

| Learning objective (opening cell) | Learner activity | Evidence exercised |
|---|---|---|
| Install, inspect the carried modules, stage and verify a mirror snapshot | run cells | printed identity and verified-file counts |
| Fetch, validate and split captioned photographs | run cell 13 | counts, digest and refusals printed |
| Encode prompts and release the encoders; load 4-bit | run, read memory figures | memory printed; no question about why the order matters |
| Read held-out loss and CLIP scores against the ceiling | read output | the reference is misdescribed (FS-M2); no prediction or checkpoint |
| Run bounded QLoRA, compare on identical held-out inputs | run, read table | no question asks the learner to explain a per-species change (two of six reference similarities fell) |
| Render a new prompt; export and reload | run | parity printed and asserted; images saved, not shown |

The objectives are operations the code performs ("install", "inspect", "stage", "load", "run") rather than learner actions with a check (GDL5); there is no Predict → Change one thing → Run → Observe → Explain activity with rerun scope (GDL10). The "Optional experiments" paragraph is the only transfer prompt.

## 4. Journeys

| Journey | Basis | Result |
|---|---|---|
| First-time learner | Source inspection | Clear, honest prose with "Look for" notes and good limits; guided layer absent (FS-M4); the ceiling framing teaches a wrong reading (FS-M2); one stale expectation (FS-m4); images not shown (FS-m5). |
| Clean default | Documented execution (Kaggle T4, reviewed blob) + direct CPU execution of the data stage | Kaggle: pass 1 stopped by the install guard, restart, pass 2 11/11 (FS-M1). CPU: cells 5, 7, 9, 13 reproduced the split, digest and refusals exactly. Model stages not executed here. |
| Active learning | Source inspection only | The four optional experiments give no rerun scope; changing `STEPS` and rerunning Section 6 after Section 7 scores the adapted model as "frozen"; rerunning Section 7 continues from adapted weights with epoch 0 labelled "frozen model" (FS-m3). Not executed. |
| Reuse and recovery | Direct CPU execution of cell 13's BYOD branch (shimmed upload) + source/record arithmetic | Stated minimum refused; one-caption-per-image refused with an actionable message; subfolder path → bare `KeyError`; non-image → `UnidentifiedImageError` naming no file; not-a-zip → `BadZipFile`; cancel → `StopIteration`; 6 of 20 images silently dropped as duplicates (FS-m1). The documented "re-run from Section 4" route is projected to exceed the T4 (FS-M3). Downstream BYOD stages, the real upload dialog and a hosted BYOD run not verified. |

## 5. Findings

### Major

#### FS-M1 — `Run all` needs a manual restart after the install cell, and the blob is marked `Release-grade` on that run

- **Cell/section:** cell 3 (Section 1), generated by `tools/build_notebook.py` (install block, lines 58–69).
- **Observed issue:** the cell `pip install`s 16 pins (torch 2.14.0, numpy 2.5.3, protobuf 7.36.2, …) into the running kernel, then raises `RuntimeError(… Restart the runtime, then rerun from the top.)` when an already-imported distribution changed version. In the recorded Kaggle run pass 1 stopped there (cuda-bindings, numpy, protobuf); pass 2 after a restart completed. `docs/release-verification.md` step 4 calls the restart "expected", and `STATUS.md` and `README.md` record "11/11 code cells ok (1 restart after the install cell)" as the basis for `Release-grade`. Colab's preinstalled torch also differs from 2.14.0, so the guard is expected to trip there too (inferred; no Colab run).
- **Consequence:** a learner who presses Run all hits an error in the first code cell and must restart and run again; a non-interactive executor needs a two-pass harness. A known-failing one-pass path is not release-ready.
- **Evidence:** documented execution (Kaggle T4 run of the reviewed blob: `run_summary.json` pass 1 error text, `restarted_after_install_cell: true`); source inspection.
- **Recommended correction:** adopt the fleet's uv isolated-environment pattern instead of installing into the kernel: a carrier cell bootstraps uv, creates `uv venv --managed-python --python 3.12.12 <ROOT>/env`, installs a hash-locked `requirements.txt` with `uv pip install --require-hashes --only-binary :all:`, and runs the workload in that environment, so the kernel's preloaded torch/NumPy are never replaced (reference: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` on `main`; this repository's capstone notebook already creates its two locked environments with uv). Return the status to `Candidate` until a one-pass hosted run is recorded, and remove "a restart is expected" from the procedure.
- **Acceptance check:** in a fresh Colab T4 runtime, one Run all of the regenerated blob completes every code cell without a restart and without an error output; the record in `docs/release-verification.md` says so explicitly.
- **Spec:** RUN1, RUN10, ENV6, REL2, REL11.

#### FS-M2 — The "real-photo ceiling" is not a ceiling: one metric includes each photo in its own reference, another sits below the generations

- **Cell/section:** cells 16–17 (Section 6), 20–21 (Section 8), 24 (Interpretation); `metrics.py` `real_photo_baseline` (lines 131–137) and `score_generations`; template `tools/notebook_template.py` line 262.
- **Observed issue:** `real_photo_baseline` scores the 12 real test photos with `references=records`, so each photo's reference similarity is its cosine with the mean embedding of its caption's test photos **including itself**. With two test photos per caption this is √((1 + cos(a, b)) / 2), which is always ≥ 70.7 and reads 88.66 in the record. Excluding the photo itself (its one same-species counterpart), the six per-caption values are 82.1, 83.5, 37.7, 51.3, 46.6, 45.2 — mean **57.7**, below the frozen (71.95) and adapted (75.01) generations. The CLIP prompt similarity "ceiling" (30.25) is also below both generated sets (31.08, 31.77), as one would expect for a generator conditioned on the prompt. Only label accuracy (0.917) behaves like an upper reference. The notebook calls these numbers "the ceiling these numbers could reach" and "the real-photo ceiling"; the self-inclusion is disclosed only in a `note` field of the JSON export, and the function's docstring says "references = the other records", which the code does not do.
- **Consequence:** the learner is taught to read the adapted 75.01 as about two-thirds of the way to an 88.66 ceiling, when on comparable terms the generations already exceed the real photos on two of the three metrics. The comparison the notebook builds its conclusion around is misdescribed.
- **Evidence:** source inspection; documented execution (the recorded `evaluation_report.json`); P5 derives the leave-one-out values exactly from the recorded per-image numbers and checks the identity on synthetic unit vectors.
- **Recommended correction:** score real photos against references that exclude the photo itself (leave-one-out; with two per caption, the other photo), and say so in the printed output. Rename the row "real-photo reference" and state per metric whether it is an upper bound: label accuracy can be read as one; prompt similarity cannot (a generator optimises for it); reference similarity should be compared on matched terms (both against the same reference set). Fix the docstring and the `Look for` and Interpretation text accordingly.
- **Acceptance check:** the exported real-photo reference similarity for each test photo excludes that photo from its reference (a test with two identical-caption photos gives cos(a, b), not √((1 + cos(a, b))/2)); no notebook text calls a metric a ceiling unless the reference is an upper bound for it on matched terms.
- **Spec:** EVAL3 (MUST: metrics explained in terms of what they measure), EVAL10, GDL8.

#### FS-M3 — BYOD followed as written loads the text encoders beside the reloaded transformer and is projected to exceed the T4

- **Cell/section:** cell 0 BYOD paragraph ("set `USE_BYOD = True` in Section 4 and re-run from that cell"; "re-running from Section 4 releases the transformer before your prompts are encoded (Section 5 does this)"); cell 15 (Section 5) `if pipe.transformer is not None: pipe.release_transformer()`; cell 23 (Section 9) `reloaded = FluxSchnellPipeline.from_artifact(...)`. Template lines 67, 234, 414.
- **Observed issue:** after the default path, `pipe.transformer` is already released (Section 9), but the global `reloaded` pipeline still holds a fresh 4-bit transformer with the adapter (7.48 GB resident in the record). Section 5 only checks `pipe.transformer`, then `pipe.encode_prompts` loads CLIP-L and T5-XXL (11.8 GB in the record, 12.15 GB total with the VAE). Nothing releases `reloaded`.
- **Consequence:** the documented BYOD route — the only route the notebook gives for "Use your own data" — is projected to fail with a CUDA out-of-memory error at Section 5 on the stated 15 GB GPU, with no recovery guidance; a learner would have to restart and lose the run.
- **Evidence:** source inspection (P6: Section 5 releases only `pipe.transformer`; Section 9 binds `reloaded`); documented execution figures (P7: 7.48 GB after reload + 11.80 GB of encoders ≈ 19.3 GB vs 15,360 MiB). Inferred; not executed (no GPU).
- **Recommended correction:** release the reloaded pipeline at the end of Section 9 (or at the top of Section 5: `if 'reloaded' in globals(): reloaded.release_transformer(); del reloaded`), and make Section 5's residency check cover every pipeline the notebook created. Add a troubleshooting line for CUDA OOM.
- **Acceptance check:** on a T4, after a complete default run, setting `USE_BYOD = True` and re-running from Section 4 with a valid zip reaches the BYOD reload-parity assertion; the GPU memory printed in Section 5 with the encoders loaded is within 1 GB of the default run's value.
- **Spec:** DAT14 (MUST: BYOD reaches validate → … → export), REL12, RUN9.

#### FS-M4 — Declared `GUIDED`, but the guided layer is largely absent

- **Cell/section:** whole notebook; template `tools/notebook_template.py` opening and section markdown.
- **Observed issue:** there is no stated learner/audience (only a "Knowledge" prerequisite list), no "How to use this notebook" (Run all, form fields, learner vs infrastructure cells), no roadmap, no Input → Model → Output contract, no glossary although the notebook introduces rectified flow, velocity, σ, NF4, double quantisation, QLoRA, packed latents, flow-matching loss, CLIP cosine and zero-shot label accuracy; no prediction prompt before the frozen/adapted comparison, no interpretation checkpoint or sample answer, no troubleshooting section for a 34 GB download and a memory-bound run, and no conclusion template. The learning objectives are operations ("install", "inspect", "stage", "load"). Cells 5, 7, 9 carry 2,171 lines of package code with a one-line label and no Infrastructure marking or collapse (P1: `cellView` unset; "Infrastructure", "How to use", "Glossary", "Troubleshoot", "Sample answer" each 0 in the markdown).
- **Consequence:** a learner new to text-to-image fine-tuning can run the notebook but is not guided to predict, interpret or check understanding, and meets 2,171 lines of code before the first model step with no signal that they may skip it.
- **Evidence:** source inspection; P1 marker counts.
- **Recommended correction:** add the GDL layer in the template: audience, how-to-use, roadmap, task contract, glossary, observable objectives, a prediction before Section 6 and before Section 8, two checkpoints with collapsible sample answers (for example "why can a generated image score above a real photo on prompt similarity?"), a troubleshooting section (install, disk, OOM, mirror download, BYOD), a conclusion template, and label/collapse the carried-module cells as Infrastructure.
- **Acceptance check:** each of GDL1–GDL14 maps to a named cell; the three carried-module cells are labelled Infrastructure and collapsed by default; at least one Predict → Change → Run → Observe → Explain activity names its rerun scope.
- **Spec:** GDL1–GDL7, GDL9–GDL14, UX8.

### Minor

#### FS-m1 — BYOD: the stated minimum is refused, and several refusals are not actionable

- **Cell/section:** cell 0 BYOD paragraph and cell 1 data contract; cell 13 BYOD branch; `samples.py` `split_dataset`, `load_byod_dataset`.
- **Observed issue (P4):** the stated minimum — "at least four images, and at least one caption with three or more images" — is refused: 4 or 5 images of one caption leave 2 or 3 training records ("at least 4 are required"); 6 are needed. A dataset with one caption per image (common for captioned-image fine-tuning) is refused; that message is actionable. A `captions.csv` that lists `images/img0000.jpg` gives a bare `KeyError: 'images/img0000.jpg'` because members are keyed by basename. A non-image member gives `UnidentifiedImageError` naming no file; a non-zip upload gives `BadZipFile`; a cancelled upload gives `StopIteration`. Six of 20 images with duplicate pixels were dropped silently (20 → 14 records, nothing printed). There is no location field (BYOD always opens the upload dialog). Generation runs one image per distinct training caption with no cap or warning (26 per model for a 30-image set with 25 unique captions; up to 2,000 by `MAX_RECORDS`). The notebook says records are "split by caption" and "Hold out by caption, not by image", which reads as a caption-disjoint split; the code stratifies within each caption, so every test caption also appears in training.
- **Consequence:** a learner following the stated contract is refused, and several failures need the source to diagnose.
- **Evidence:** direct CPU execution of cell 13's BYOD branch (shimmed upload, twelve archives); source inspection.
- **Recommended correction:** state the real minimum (for example "at least six images of one caption, or …"), resolve CSV paths relative to the archive with containment checks or refuse them with a message, catch decode errors and name the file, report dropped duplicates and their count, handle a cancelled upload, add a `BYOD_PATH` location field, warn when the number of generation prompts is large, and describe the split as "stratified within each caption".
- **Acceptance check:** a zip at the stated minimum is accepted; each P4 invalid case ends in a message that names the file or rule and the fix; duplicates dropped are printed with a count.
- **Spec:** DAT12, DAT19, VAL7 (MUST); EXE2, UX10.

#### FS-m2 — The exported source revision is not the revision whose modules run

- **Cell/section:** cell 3 `NOTEBOOK_SOURCE['repository_revision']`, `metadata.dimer.generated_from.revision`, the Section 2 heading and the opening cell (all `77567b7`), and the result JSON in cell 23; `tools/build_notebook.py` `load_context` (`module_revision: revision or _head_revision(repo)`, line 347) and `main` (`recorded_revision`, lines 599–603).
- **Observed issue (P2):** the carried `pipeline.py` has SHA-256 `d6ac136e…`, equal to `src/…/pipeline.py` at `d4fdf72` and `HEAD`; at the recorded revision `77567b7` that file is `3e41d692…` — the version without the `.md` suffix rule whose absence failed the v1 Kaggle run. `metrics.py` and `samples.py` match both. The generator labels the notebook with `HEAD` at generation time, which was the parent of the commit that changed the module. The Interpretation says success "proves that the recorded repository revision's pipeline modules … can stage" the snapshot.
- **Consequence:** anyone reconstructing the run from the exported revision gets a `pipeline.py` that refuses the scorer snapshot; the exported provenance does not identify the code that ran.
- **Evidence:** source inspection and P2 hashes.
- **Recommended correction:** record the per-module SHA-256 values in `NOTEBOOK_SOURCE` and the result JSON (they are already in cell metadata), and either regenerate after the module commit so the label is a revision that contains those modules, or name the label "generated from working tree based on `<rev>`".
- **Acceptance check:** checking out the exported revision and hashing the three module files reproduces the carried cells' SHA-256 values.
- **Spec:** OUT7, OUT9 (SHOULD); framework dimension 2 ("exports that cannot be reconstructed as claimed"). OUT6 (model identity and revision) is met; the gap is the code revision.

#### FS-m3 — The optional experiments give no rerun scope, and two of them score the adapted model as "frozen"

- **Cell/section:** cell 24 "Optional experiments"; cells 17, 19; `pipeline.py` `load_transformer` (returns early when a transformer is resident) and `adapt`; template line 477.
- **Observed issue:** "set `STEPS = 1` or `2`" requires rerunning Section 6, but after Section 7 `pipe.load_transformer()` returns `{"loaded": False}` and keeps the adapted transformer, so the "frozen" loss and generations printed are the adapted model's. "Raise `EPOCHS` or `LEARNING_RATE`" reruns Section 7 on the already adapted LoRA: `adapt` snapshots the current weights as its restore point and labels epoch 0 "frozen model (LoRA at initialisation: B = 0)". No experiment says which cells to rerun, and none asks for a prediction.
- **Consequence:** the obvious way to do two of the four suggested experiments produces mislabelled comparisons.
- **Evidence:** source inspection (P6). Not executed.
- **Recommended correction:** give each experiment its rerun scope (for example "run `pipe.release_transformer()` then rerun from Section 6"), or have `adapt` refuse an already adapted pipeline with a message; frame one experiment as Predict → Change one thing → Run → Observe → Explain.
- **Acceptance check:** following each experiment's written instructions produces a run whose "frozen" rows come from the pinned base with an untrained LoRA (epoch 0 equal to the default run's frozen validation loss).
- **Spec:** UX7 (MUST: exercises must not leave the notebook inconsistent), GDL10, UX5.

#### FS-m4 — Expected outputs and conclusions that do not match the run; missing overlap and variability statements

- **Cell/section:** cell 16 ("label accuracy below the real photographs' 1.0"), cell 0 ("about an hour of model time"), cell 24 (Interpretation opening). Template lines 63, 263, 452.
- **Observed issue:** the recorded real-photo label accuracy is 0.917, not 1.0 (one White-throated Sparrow photo is nearest the Junco caption). The opening says about an hour of model time; the record is 2,912 s wall including 536 s of staging, and `tutorials/README.md` says about 40 minutes. The Interpretation states the outcome as fact ("lowers the held-out flow-matching loss … and moves its four-step generations towards the held-out real photographs") rather than as what to check — in the record two of the six per-caption reference similarities fell (Chipping Sparrow 70.24 → 68.58, Dark-eyed Junco 69.16 → 68.69). The notebook does not state that iNaturalist photographs may overlap the web-scale data behind the LAION-trained CLIP scorer and FLUX's undisclosed training set, and does not state that GPU fine-tuning is not bit-reproducible across runtimes.
- **Consequence:** a learner whose numbers differ cannot tell legitimate variation from a fault, and the conclusion is pre-written.
- **Evidence:** source inspection; documented execution (record values).
- **Recommended correction:** describe expected output by shape ("label accuracy at or below the real photos'"), make the timing consistent with the record, phrase the Interpretation conditionally with a prompt to check the per-caption rows, and add one sentence each on pretraining overlap and run-to-run variability.
- **Acceptance check:** no expected-output note contradicts the recorded run; the Interpretation contains no result stated before it is computed; DAT9 and ENV8 statements are present.
- **Spec:** DAT9, ENV8 (MUST); GDL8, GDL14.

#### FS-m5 — The generated images are never displayed

- **Cell/section:** cells 17, 21, 23 (`grid(...)` saves a JPEG and prints its path; the new-prompt images are saved as PNGs).
- **Observed issue:** Section 6 says "Look for … a first grid of six generated birds", and Section 8 asks the learner to compare the second grid "with the first by eye", but no cell displays an image; the learner must find the files in the runtime's file browser.
- **Consequence:** in a text-to-image tutorial the primary output is not shown where the learner is reading.
- **Evidence:** source inspection; the Kaggle `executed.ipynb` contains no image output in any cell (stream and progress-widget outputs only).
- **Recommended correction:** display both grids side by side with their per-caption reference similarities, and the two new-prompt images, inline (keeping the files).
- **Acceptance check:** after a run, cells 17, 21 and 23 each show the images they describe.
- **Spec:** UX3, UX11.

### Suggestions

- **FS-S1 — Declare the current spec.** The notebook, `tutorials/README.md` and the validator declare NOTEBOOK_SPEC 2.0; regenerate against 2.2 when the template is revised.
- **FS-S2 — Document `DIMER_NOTEBOOK_CI_PREINSTALLED`.** Cell 3 reads it to skip the install, but no markdown mentions it (EXE5).
- **FS-S3 — Record a Colab run and state host RAM.** The badge and the Prerequisites name Colab as the supported runtime, but the only executions are on Kaggle. Prerequisites state GPU memory and disk but not host RAM, which matters when 10 GB safetensors shards are quantised on load.
- **FS-S4 — Plot the per-σ held-out loss** (frozen vs adapted) from the comparison already computed, so the learner sees where the adaptation helps most.

## 6. Readiness

**Needs revision.** Open Majors FS-M1 to FS-M4; unresolved MUSTs RUN1, RUN10, ENV6, REL2, REL11 (FS-M1), EVAL3 (FS-M2), DAT14, REL12 (FS-M3), DAT12, DAT19, VAL7 (FS-m1), UX7 (FS-m3), DAT9, ENV8 (FS-m4). The repository's `Release-grade` status rests on a restart-dependent run and should return to `Candidate`. Remaining gates after the fixes: a one-pass hosted Run all of the regenerated blob (ideally on Colab, the stated runtime), re-recorded comparison numbers with a leave-one-out real-photo reference, and a hosted BYOD run after the default path that reaches reload parity, plus one invalid BYOD input.

## 7. Verified versus inferred

- **Verified by direct execution (CPU, data stage only):** the sample fetch, validation, refusals and split (identical to Kaggle), no species/observer pair in two splits (P3); the BYOD branch of cell 13 with twelve archives (P4); the carried-module hashes against the recorded revision (P2).
- **Verified from documented execution:** the install-cell restart and every model-stage number quoted here (Kaggle T4, reviewed blob), and the leave-one-out real-photo values derived exactly from the recorded per-image numbers (P5).
- **Inferred from source and recorded figures:** the BYOD rerun OOM (FS-M3), the optional-experiment state hazards (FS-m3), Colab behaviour and the real upload dialog.
- **Most likely to be wrong:** FS-M3's outcome. The projection adds the recorded encoder increment to the memory left by `reloaded`; if PyTorch's caching allocator or a garbage collection freed `reloaded`'s weights before Section 5, the rerun would fit. Nothing in the notebook deletes `reloaded`, so I expect the OOM, but it has not been run.

Probe ZIP: `flux_schnell_generation_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).
