#!/usr/bin/env python3
"""Generate tutorials/DIMER_Bosch_Synthetic_Defect_Augmentation_Capstone.ipynb from tools/capstone/.

The notebook is standalone (NOTEBOOK_SPEC 2.2 §4): every stage script, the shared core and both dependency locks are
embedded as `%%writefile` / form cells, so a fresh runtime needs no repository. Edit the sources in tools/capstone/
or the prose below, then regenerate:

    python tools/build_capstone_notebook.py           # write the notebook
    python tools/build_capstone_notebook.py --check   # exit 1 if the committed notebook differs
"""
# ruff: noqa: E501  -- learner-facing prose is kept on single lines so the rendered markdown stays readable
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tools" / "capstone"
NOTEBOOK = ROOT / "tutorials" / "DIMER_Bosch_Synthetic_Defect_Augmentation_Capstone.ipynb"
GENERATOR = "tools/build_capstone_notebook.py/1"
WORK_REL = "work/sdi_capstone"
REPO = "kurtvalcorza/flux-schnell-generation-pipeline"
COLAB = f"https://colab.research.google.com/github/{REPO}/blob/main/tutorials/{NOTEBOOK.name}"
SCRIPTS = ("stage_data.py", "stage_features.py", "stage_fit.py", "stage_phi4.py", "stage_prompts.py", "stage_flux.py", "stage_evaluate.py", "stage_export.py", "stage_reload.py")
SECTION_MARK = re.compile(r"^# %% \[section\] (.+)$", re.M)


def core_sections() -> list[tuple[str, str]]:
    """Split sdi_core.py at its section markers: (title, text). The first chunk is the module header."""
    text = (SRC / "sdi_core.py").read_text(encoding="utf-8")
    marks = list(SECTION_MARK.finditer(text))
    chunks = [("Module header, imports and helpers", text[: marks[0].start()])]
    for n, m in enumerate(marks):
        end = marks[n + 1].start() if n + 1 < len(marks) else len(text)
        chunks.append((m.group(1), text[m.start() : end]))
    return chunks


def writefile(rel: str, body: str, append: bool = False) -> str:
    # An appended section starts with a newline, so the file stays valid even if a runtime drops a trailing newline.
    return f"%%writefile {'-a ' if append else ''}{WORK_REL}/{rel}\n" + ("\n" if append else "") + body.strip("\n") + "\n"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n") + "\n"}


def code(text: str, *, form: bool = False, tag: str | None = None) -> dict:
    meta: dict = {}
    if form:
        meta["cellView"] = "form"
    if tag:
        meta["dimer"] = {"writes": tag}
    return {"cell_type": "code", "execution_count": None, "metadata": meta, "outputs": [], "source": text.strip("\n") + "\n"}


def details(summary: str, body: str) -> str:
    return f"<details><summary><b>{summary}</b></summary>\n\n{body.strip()}\n\n</details>"


def checkpoint(question: str, answer: str) -> str:
    return f"> **Check your reasoning.** {question}\n\n" + details("Sample answer (open after you have answered)", answer)


# ------------------------------------------------------------------------------------------------ prose

OPENING = f"""
# DIMER Guided Notebook: Can Synthetic Defects Improve Visual Inspection?

**Profile:** `E2E` · **Mode:** `GUIDED` · **Notebook spec:** DIMER Notebook Specification 2.2 · **Status:** candidate (the combined Colab T4 run has not yet been recorded)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})

This notebook runs a controlled experiment. You will take a small, fixed set of labelled photographs of manufactured surfaces from the Bosch Surface Defect Inspection (SDI) dataset, ask **Phi-4-multimodal-instruct** to describe what it sees, compile those descriptions into bounded text prompts, let **FLUX.1 [schnell]** generate candidate *scratch* and *spot* images from the prompts, and then test whether substituting those generated images into training improves a simple classifier on **real held-out photographs**, compared with conventional augmentation.

Success means a reproducible, interpretable comparison. A negative or null result is a valid outcome: the notebook does not require the synthetic images to help.

**Run all:** the default path downloads and verifies the dataset, audits it, freezes a grouped split, fits a majority baseline and a real-only classifier, describes training exemplars with Phi-4, generates 64 candidates with FLUX, fits three matched classifier arms with three seeds, freezes the experiment, scores the real test set once, exports the selected classifier, and rebuilds it in a fresh process to prove the export reproduces the same outputs. It needs no token, login, upload, restart, repository clone or DIMER service.

**Bring Your Own Data:** after the sample run, you can repeat the whole experiment on your own labelled images (Section 13). The same validation, split, generation, fitting, evaluation and export stages apply.

**By the end of this notebook you will be able to:**

- **explain** why accuracy misleads under class imbalance and **compare** classifiers with macro-F1;
- **identify** which parts of a model-written description are observed and which are guessed;
- **distinguish** a generated image's *intended* label from a verified annotation, and visual plausibility from measured usefulness;
- **diagnose** leakage risks: duplicate images across partitions, test-set peeking, and unequal training budgets;
- **interpret** a paired contrast between augmentation strategies with its bootstrap interval, per seed and on average;
- **apply** the same protocol to your own data and **write** a conclusion that stays within the evidence.

**This notebook does not demonstrate:** defect localisation or masks, dimensional measurement, material or severity inference, fine-tuning of Phi-4 or FLUX, physical defect simulation, certification, or a production inspection system. FLUX generates new images from text; it does not edit a supplied specimen and does not guarantee a physically valid defect.
"""

AUDIENCE = """
## Before you start

**Who this is for.** You can open a Colab notebook, run cells, and read basic Python. You have met image classification before. Class imbalance, augmentation, held-out evaluation and macro-F1 are introduced here. The dataset is an international manufacturing dataset; any local relevance is by analogy, not because it contains Philippine observations.

**Runtime.** A Colab **T4 GPU** runtime (Runtime → Change runtime type → T4 GPU) with at least 15 GB of GPU memory. The runtime check in Section 1 stops with an instruction if no suitable GPU is attached.

**Resource budget (estimates, not measurements).** About 53 GB of downloads (dataset 1.6 GB, Phi-4 11.2 GB, FLUX 33.7 GB, ResNet-18 0.05 GB, Python wheels about 6 GB) and about 55 GB of free disk with the default of deleting each model's weights after use. Model time is expected to be well under 90 minutes on a T4, excluding downloads. The notebook measures and records the actual download bytes, stage times, peak host memory and peak GPU memory; a later hosted run replaces these estimates.

### How to use this notebook

1. Select the T4 runtime, then choose **Runtime → Run all**. Nothing needs editing.
2. Cells titled **Infrastructure: …** set up environments, locks and helpers. Run them; you do not need to study them. Cells that begin with `%%writefile` save a stage script into `work/sdi_capstone/`; the *next* code cell runs it. The scripts are the experiment's actual logic, so read the ones a section discusses.
3. Form fields (the configuration cell in Section 1) are the only values you might change, and only for the optional activities. Every field has a safe default.
4. Each stage prints what it did. After each stage, read **What to notice**, answer the **Check your reasoning** question, then open the sample answer.
5. Outputs go to `outputs/bosch_sdi_capstone/` (or `outputs/byod_capstone/` for your own data). Nothing is uploaded anywhere.

### Why two Python environments?

Phi-4-multimodal needs `transformers 4.48.2` and `torch 2.6`; the FLUX pipeline needs `transformers 5.17`, `diffusers 0.40` and `torch 2.14`. They cannot share one interpreter. Section 1 creates two isolated environments from exact dependency locks, and each heavy stage runs as a separate process in the right environment. A process that exits returns all of its GPU and host memory, which is also how the notebook keeps Phi-4, the FLUX text encoders, the FLUX transformer and the feature extractor from ever occupying the GPU together.
"""

ROADMAP = """
## Roadmap

| § | Stage | Kind |
|---|---|---|
| 1 | Runtime check, locked environments, shared core | infrastructure |
| 2 | Dataset card, acquisition, digest verification, quality audit, frozen grouped split | data practice |
| 3 | Majority baseline and a real-only classifier (validation only) | core concept |
| 4 | Phi-4 descriptions of training exemplars: observed versus guessed | core concept |
| 5 | Bounded prompt compilation | engineering |
| 6 | FLUX generation, contact sheets, intended labels | core concept |
| 7 | Synthetic features and the synthetic-to-training audit | evaluation practice |
| 8 | Three matched arms, validation selection, frozen experiment | evaluation practice |
| 9 | One final real-test comparison | evaluation practice |
| 10 | Error analysis, uncertainty and an evidence-based conclusion | interpretation |
| 11 | Export, fresh-process reload parity, new-image inference | engineering |
| 12 | Records: environment, run summary, limitations, attribution | engineering |
| 13 | Optional: change one thing, human review, bring your own data, troubleshooting | optional |

**Fast path.** If you are short of time, run everything and read Sections 3, 6, 9 and 10.
"""

QUESTION = """
## The question, and your prediction

> Under a fixed real-data budget, classifier, class-sampling policy and training budget, does substituting synthetic defect examples into training improve **real-test macro-F1** compared with conventional augmentation?

The experiment trains a small classifier on top of frozen image features in three **arms**, each with the same number of optimiser updates, the same batch composition and the same random draws per seed:

| Arm | Training source | Purpose |
|---|---|---|
| Majority baseline | always predicts `normal` | shows why accuracy misleads under imbalance |
| **A** real-only | 128 normal, 64 scratch, 32 spot real images, unaugmented | reference classifier |
| **B** conventional augmentation | the same real images, each drawn as one of four fixed flip/brightness/contrast views | primary practical comparator |
| **C** synthetic augmentation | as B, but each defect slot draws a generated image with probability 0.5 | tests the generated pool |

The **primary contrast is C − B**; B − A is context.

**Make a prediction now, before any model runs.** Write down (1) whether C − B will be positive, zero or negative, (2) roughly how large, and (3) which class you expect synthetic images to help most. You will compare against the evidence in Section 10.

### Input → System → Output

```
labelled Bosch training images (product A: normal / scratches / spots)
   │  audit, group duplicates, freeze a 60/20/20 grouped split, pick 128/64/32 training images
   ├─► Phi-4-multimodal (frozen, 4-bit) ─► descriptions of 3 exemplars per class
   │        └─► deterministic, bounded prompt template ─► 6 prompts
   │                 └─► FLUX.1 [schnell] (frozen, 4-bit) ─► 32 candidates per defect class (intended labels)
   ├─► ResNet-18 (frozen, ImageNet) ─► 512-d features for real + synthetic images
   └─► linear head fitted per arm × seed ─► validation selects epochs ─► FREEZE
                                             └─► real test photographs, scored once
Output: per-image predictions, macro-F1 per arm/seed, paired C − B contrast with a bootstrap interval,
        an exported classifier that a fresh process reproduces, and an evidence record.
```

**Two ideas carry the whole notebook.** First, a generated image's label is *intended* (it is the label in the prompt), not verified by anyone. Second, only held-out *real* photographs can say whether generated images were useful; how realistic they look is a different question.
"""

GLOSSARY = details("Glossary (open when a term is unfamiliar)", """
- **Class imbalance** — one class (here `normal`, about 93% of images) vastly outnumbers others. A classifier that always says `normal` scores high accuracy while finding no defects.
- **Macro-F1** — compute F1 (the harmonic mean of precision and recall) separately for each class, then average the classes with equal weight. A missed rare class hurts it as much as a missed common one (Sokolova & Lapalme, 2009).
- **Balanced accuracy** — the average of per-class recall.
- **Augmentation** — label-preserving transformations of training images (here: horizontal flip, mild brightness and contrast) that add variation without new photographs.
- **Synthetic augmentation** — adding generated images to training (Trabucco et al., 2023; Wang et al., 2023).
- **Intended label** — the class named in the prompt that produced a generated image. Nobody has checked that the image shows that defect.
- **Held-out evaluation** — scoring on images that were never used for fitting or for any choice. **Validation** images are used for choices (epoch, exported arm); **test** images are used once, after everything is frozen.
- **Group / leakage** — near-identical images (duplicates, consecutive frames) must stay in the same partition; otherwise the test set contains near-copies of training images and scores are inflated.
- **Frozen features / linear head** — ResNet-18 (He et al., 2016), pretrained on ImageNet (Deng et al., 2009), turns each image into a 512-number vector and is never updated. Only a small linear layer (3 × 512 weights + 3 biases) is fitted.
- **NF4 / 4-bit** — a 4-bit weight format that lets Phi-4 and FLUX fit a 16 GB GPU (Dettmers et al., 2023). Every number from these models is the 4-bit model's.
- **Seed** — the number that fixes random draws. Three seeds show how much results move for reasons unrelated to the method.
- **Paired group bootstrap** — resample the test groups with replacement many times, re-score every arm on each resample, and read the spread of the C − B difference (Efron & Tibshirani, 1993).
""")

S1_INTRO = """
## 1 · Runtime, locked environments and the shared core  <sub>(infrastructure)</sub>

The configuration cell holds every value a reader can change. The defaults run the canonical experiment; the optional fields are explained in Section 13. The Phi-4 checkpoint ships its model code as Python files inside the model repository, which `transformers` executes. `ALLOW_PHI4_REMOTE_CODE` is set to `True` so that Run all proceeds; that code is pinned to one immutable revision and each file's SHA-256 is verified before it is imported (Section 4 explains the boundary). Set it to `False` to stop before Phi-4 runs.

**BYOD contract (read before choosing `byod`).** A zip containing `manifest.csv` with columns `image_id`, `path`, `class` and optional `group`, plus the images (`.jpg`, `.jpeg`, `.png`, sides 64–4096 px). Exactly three classes: one named `normal` and two defect classes (lowercase names such as `dent`, `stain`). Paths are relative to the manifest, without `..`. A `group` (for example a specimen or session id) must never span two classes. After grouping, each class needs at least 128 / 64 / 32 independent training groups (normal / more frequent defect / less frequent defect) and at least 5 images in each of validation and test. Data stays inside this runtime; the only transmission is the images you upload to the runtime yourself. **Do not upload confidential, restricted, personal or regulated images unless you are authorised to process them in Colab.**
"""

CONFIG = '''
# @title Configuration: form fields (the defaults run the canonical experiment)
DATA_SOURCE = 'bosch'  # @param ["bosch", "byod"]
BYOD_ZIP_PATH = ''  # @param {type:"string"}
NEW_IMAGE_DIR = ''  # @param {type:"string"}
ALLOW_PHI4_REMOTE_CODE = True  # @param {type:"boolean"}
SEEDS = '17,29,43'  # @param {type:"string"}
DELETE_MODEL_WEIGHTS_AFTER_USE = True  # @param {type:"boolean"}
RUN_SYNTHETIC_FRACTION_EXERCISE = True  # @param {type:"boolean"}
EXERCISE_SYNTHETIC_PROBABILITY = 0.25  # @param {type:"number"}
RUN_HUMAN_REVIEW_EXTENSION = False  # @param {type:"boolean"}
HUMAN_REVIEW_CSV = ''  # @param {type:"string"}
BYOD_CHECK_ZIP = ''  # @param {type:"string"}

import json
import time
from pathlib import Path

NOTEBOOK_STARTED = time.time()
BASE = Path.cwd()
WORK_ROOT = BASE / 'work' / 'sdi_capstone'  # scripts, locks, environments (shared by both data sources)
WORK = WORK_ROOT / DATA_SOURCE  # data, weights, features and heads for this data source
OUT = BASE / 'outputs' / ('bosch_sdi_capstone' if DATA_SOURCE == 'bosch' else 'byod_capstone')
for folder in (WORK_ROOT / 'locks', WORK / 'state', OUT / 'figures'):
    folder.mkdir(parents=True, exist_ok=True)
if DATA_SOURCE not in ('bosch', 'byod'):
    raise ValueError("DATA_SOURCE must be 'bosch' or 'byod'")
seeds = [int(s) for s in SEEDS.replace(' ', '').split(',') if s]
if not seeds or len(set(seeds)) != len(seeds):
    raise ValueError('SEEDS must be distinct integers separated by commas, for example 17,29,43')
if DATA_SOURCE == 'byod':
    print('BYOD: images stay in this runtime. Do not upload confidential, restricted, personal or regulated data '
          'unless you are authorised to process it here.')
    if not BYOD_ZIP_PATH:
        from google.colab import files  # optional interactive branch; the default path never reaches this line

        uploaded = files.upload()
        if len(uploaded) != 1:
            raise ValueError('upload exactly one zip file (manifest.csv + images)')
        BYOD_ZIP_PATH = str(WORK / 'byod_upload.zip')
        Path(BYOD_ZIP_PATH).write_bytes(next(iter(uploaded.values())))
RUN_CONFIG = WORK / 'run_config.json'
RUN_CONFIG.write_text(json.dumps({
    'base_dir': str(BASE), 'work_dir': str(WORK), 'out_dir': str(OUT), 'data_source': DATA_SOURCE,
    'byod_zip': BYOD_ZIP_PATH, 'new_image_dir': NEW_IMAGE_DIR, 'seeds': seeds,
    'allow_phi4_remote_code': bool(ALLOW_PHI4_REMOTE_CODE),
    'delete_model_weights_after_use': bool(DELETE_MODEL_WEIGHTS_AFTER_USE), 'delete_archive_after_extract': True,
}, indent=2))
print(f'data source: {DATA_SOURCE}; seeds: {seeds}' + ('' if len(seeds) >= 3 else ' (fewer than three seeds: a SMOKE run, not the canonical comparison)'))
print(f'working files: {WORK}\\noutputs: {OUT}')
'''

RUNTIME_CHECK = '''
# @title Infrastructure: runtime check (GPU, disk, host memory)
import platform
import shutil
import subprocess


def _smi(query):
    try:
        result = subprocess.run(['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return ''
    return result.stdout.strip().splitlines()[0] if result.returncode == 0 and result.stdout.strip() else ''


with open('/proc/meminfo') as handle:
    MEMINFO = {line.split(':')[0]: int(line.split()[1]) for line in handle}
REQUIRED_DISK_GB = 55 if DELETE_MODEL_WEIGHTS_AFTER_USE else 70
RUNTIME = {
    'python_kernel': platform.python_version(),
    'platform': platform.platform(),
    'gpu': _smi('name') or 'none',
    'gpu_memory_mib': int(_smi('memory.total') or 0),
    'driver': _smi('driver_version') or 'none',
    'host_memory_gib': round(MEMINFO['MemTotal'] / 2**20, 1),
    'disk_free_gb_at_start': round(shutil.disk_usage(BASE).free / 1e9, 1),
    'required_disk_gb': REQUIRED_DISK_GB,
}
print(json.dumps(RUNTIME, indent=2))
if RUNTIME['gpu_memory_mib'] < 15000:
    raise RuntimeError('This notebook needs a CUDA GPU with at least 15 GB. In Colab: Runtime > Change runtime type > T4 GPU, then Run all again.')
if RUNTIME['disk_free_gb_at_start'] < REQUIRED_DISK_GB:
    raise RuntimeError(f"Only {RUNTIME['disk_free_gb_at_start']} GB of disk is free; the run needs about {REQUIRED_DISK_GB} GB. Start a fresh runtime (Runtime > Disconnect and delete runtime) and Run all again.")
if RUNTIME['host_memory_gib'] < 12:
    print('WARNING: less than 12 GiB of host memory; the 4-bit model loads may run out of memory (see Troubleshooting).')
'''


def locks_cell() -> str:
    lab = (SRC / "locks" / "lab.lock").read_text(encoding="utf-8").strip()
    phi4 = (SRC / "locks" / "phi4.lock").read_text(encoding="utf-8").strip()
    return f'''
# @title Infrastructure: exact dependency locks (every transitive package pinned)
# Resolved with `uv pip compile` for CPython 3.12 on x86_64 manylinux from tools/capstone/locks/*.in.
# The lab lock takes torch/torchvision from the CUDA 12.6 wheel index (these run on any CUDA 12.x or newer driver).
LOCKS = {{
    'lab': """
{lab}
""",
    'phi4': """
{phi4}
""",
}}
for _name, _text in LOCKS.items():
    (WORK_ROOT / 'locks' / f'{{_name}}.lock').write_text(_text.strip() + '\\n')
print({{name: len(text.split()) for name, text in LOCKS.items()}}, 'pinned packages')
'''


INSTALL = '''
# @title Infrastructure: create the two locked environments (about 5-8 minutes on first run)
import hashlib
import sys
import sysconfig

UV_VERSION = '0.12.19'
PYTHON_REQUEST = '3.12'
INDEX_ARGS = {
    'lab': ['--index-url', 'https://download.pytorch.org/whl/cu126', '--extra-index-url', 'https://pypi.org/simple', '--index-strategy', 'unsafe-best-match'],
    'phi4': ['--index-url', 'https://pypi.org/simple'],
}
PROBE = """
import importlib.metadata as md, json, platform, sys
out = {'python': platform.python_version()}
for d in ('torch', 'torchvision', 'transformers', 'diffusers', 'bitsandbytes', 'accelerate', 'peft', 'huggingface-hub', 'safetensors', 'numpy', 'pillow', 'matplotlib'):
    try:
        out[d] = md.version(d)
    except md.PackageNotFoundError:
        pass
import torch
out['cuda_available'] = torch.cuda.is_available()
out['torch_cuda'] = torch.version.cuda
out['device'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
print(json.dumps(out))
"""
# uv is a standalone installer binary; installing it does not touch any module this kernel has loaded.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', f'uv=={UV_VERSION}'], check=True)
UV = shutil.which('uv') or str(Path(sysconfig.get_path('scripts')) / 'uv')
ENV_PYTHON, ENVIRONMENTS = {}, {}
for name in ('lab', 'phi4'):
    lock = WORK_ROOT / 'locks' / f'{name}.lock'
    root = WORK_ROOT / 'envs' / name
    python = root / 'bin' / 'python'
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    marker = root / '.lock_sha256'
    started = time.time()
    if not (python.exists() and marker.exists() and marker.read_text() == digest):
        shutil.rmtree(root, ignore_errors=True)
        subprocess.run([UV, 'venv', '--quiet', '--python', PYTHON_REQUEST, str(root)], check=True)
        subprocess.run([UV, 'pip', 'install', '--quiet', '--python', str(python), '--no-deps', '--no-cache', '--link-mode', 'copy', '-r', str(lock), *INDEX_ARGS[name]], check=True)
        marker.write_text(digest)
    probe = json.loads(subprocess.run([str(python), '-c', PROBE], capture_output=True, text=True, check=True).stdout)
    if not probe['cuda_available']:
        raise RuntimeError(f"the {name} environment cannot see the GPU (torch {probe['torch']}, CUDA {probe['torch_cuda']}, driver {RUNTIME['driver']}); see Troubleshooting")
    ENVIRONMENTS[name] = {**probe, 'lock_sha256': digest, 'uv': UV_VERSION, 'install_seconds': round(time.time() - started, 1)}
    ENV_PYTHON[name] = str(python)
    print(f"{name}: Python {probe['python']}, torch {probe['torch']} (CUDA {probe['torch_cuda']}), transformers {probe.get('transformers')}, "
          f"diffusers {probe.get('diffusers', '-')}, bitsandbytes {probe.get('bitsandbytes')} on {probe['device']} ({ENVIRONMENTS[name]['install_seconds']} s)")
(WORK / 'state' / 'environments.json').write_text(json.dumps(ENVIRONMENTS, indent=2))
'''

CORE_INTRO = """
### The shared core

The next cells save `sdi_core.py`, the small library every stage imports: acquisition and archive safety, image validation, duplicate grouping, the split, prompt compilation, candidate eligibility, preprocessing and augmentation views, the matched slot schedule, metrics, the bootstrap, the frozen record and the classifier artifact. It uses only the Python standard library, NumPy and Pillow, so the same code runs in both environments. Each section is one cell; later sections of this notebook point back to the parts they use. You may run these cells without reading them now.
"""

RUNNER = '''
# @title Infrastructure: stage runner and display helpers (run it; no need to study it)
import os
import threading

from IPython.display import Image as IPImage
from IPython.display import Markdown, display

STAGE_LOG = {}
CHILD_ENV = {k: v for k, v in os.environ.items() if k not in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'PYTHONPATH')}  # the default path uses no credential
CHILD_ENV.update(MPLBACKEND='Agg', PYTHONHASHSEED='0', HF_HOME=str(WORK_ROOT / 'hf_home'), HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_PROGRESS_BARS='1',
                 TOKENIZERS_PARALLELISM='false', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1',
                 HF_XET_CHUNK_CACHE_SIZE_BYTES='0')  # no hidden multi-GB download cache: weights live only where the stages put them


def _gpu_used_mib():
    value = _smi('memory.used')
    return int(value) if value else None


def _mem_available_mib():
    with open('/proc/meminfo') as handle:
        for line in handle:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) // 1024
    return None


def run_stage(name, env, script, *args):
    """Run one stage script in its locked environment, stream its output, and record wall time, the child's peak
    resident memory, peak GPU memory in use and the lowest available host memory while it ran."""
    cmd = [ENV_PYTHON[env], '-u', str(WORK_ROOT / script), str(RUN_CONFIG), *map(str, args)]
    peaks = {'peak_gpu_used_mib': 0, 'min_host_available_mib': None}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            gpu, mem = _gpu_used_mib(), _mem_available_mib()
            if gpu is not None:
                peaks['peak_gpu_used_mib'] = max(peaks['peak_gpu_used_mib'], gpu)
            if mem is not None:
                low = peaks['min_host_available_mib']
                peaks['min_host_available_mib'] = mem if low is None else min(low, mem)
            stop.wait(1.0)

    disk_before = shutil.disk_usage(BASE).free
    started = time.time()
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=CHILD_ENV, cwd=BASE)
    for line in proc.stdout:
        print(line, end='')
    _, status, usage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    stop.set()
    monitor.join()
    record = {'env': env, 'script': script, 'args': list(map(str, args)), 'exit_code': proc.returncode, 'seconds': round(time.time() - started, 1),
              'peak_child_rss_mib': usage.ru_maxrss // 1024, **peaks,
              'disk_free_gb_before': round(disk_before / 1e9, 1), 'disk_free_gb_after': round(shutil.disk_usage(BASE).free / 1e9, 1)}
    STAGE_LOG[name] = record
    (WORK / 'state' / 'stage_log.json').write_text(json.dumps(STAGE_LOG, indent=2))
    print(f"\\n[{name}] {record['seconds']} s | peak child RSS {record['peak_child_rss_mib']} MiB | peak GPU in use {record['peak_gpu_used_mib']} MiB | "
          f"lowest host memory available {record['min_host_available_mib']} MiB | disk free {record['disk_free_gb_after']} GB")
    if proc.returncode != 0:
        raise RuntimeError(f"stage '{name}' failed with exit code {proc.returncode}. Read the last lines above, then see Troubleshooting in Section 13.")
    return record


def load(name):
    path = OUT / name
    return json.loads(path.read_text()) if name.endswith('.json') else [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def table(headers, rows):
    lines = ['| ' + ' | '.join(map(str, headers)) + ' |', '|' + '---|' * len(headers)]
    lines += ['| ' + ' | '.join('—' if v is None else str(v) for v in row) + ' |' for row in rows]
    display(Markdown('\\n'.join(lines)))


def fmt(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def show(name, width=None):
    path = OUT / 'figures' / name
    if path.exists():
        display(IPImage(filename=str(path), width=width))
    else:
        print(f'(no figure {name})')
'''

S2_INTRO = """
## 2 · The dataset: acquisition, audit and a frozen grouped split  <sub>(data practice)</sub>

**Dataset card.** The Bosch Surface Defect Inspection (SDI) dataset (source repository pinned at commit `c6e0afe6…`) contains grayscale photographs of three products (A, B, C), each labelled `normal`, `scratches` or `spots`. It was released with the DT-GAN study (Wang et al., 2023); this notebook is a new experiment, not a reproduction of that paper's results. The repository declares **CC BY-SA 4.0**: attribution is required and adapted dataset material you redistribute must carry the same licence. The dataset's licence does not automatically apply to the notebook code, the models, or generated images; Section 12 records each separately.

**Acquisition.** The archive is a 1,647,097,112-byte Git LFS object. The data stage streams it with timeouts, resumes on interruption, verifies its byte count and SHA-256 before extraction, and refuses path traversal, symlinks, non-image files and anything beyond an expanded-size ceiling set from the inspected archive. It never substitutes another dataset.

**Audit and split.** Before any model sees an image, the stage inventories every file, compares the counts with the published notes (documented counts are expectations, measured counts are facts), finds exact decoded-pixel duplicates and near duplicates, and forms groups. Groups, not images, are assigned to train / validation / test (60/20/20, approximately class-stratified, deterministic). The source supplies no specimen or session identifiers, so groups approximate shared capture and this is an **exploratory image-group benchmark**, not a claim about independent physical specimens. Only product A is used. From the training partition, a fixed budget of 128 normal, 64 scratch and 32 spot images is selected (one per group); the rest of the training partition stays unused.

**Question for this stage:** do the measured counts match the documentation, and could the same picture end up in both training and test?
"""

S2_RUN = """
run_stage('data', 'lab', 'stage_data.py')
DATA = load('data_manifest.json')
if DATA_SOURCE == 'bosch':
    inv = DATA['inventory']
    table(['product', 'class', 'documented', 'measured', 'difference'],
          [[p, c, n, inv['measured_counts'].get(p, {}).get(c, 0), inv['differences'][p][c]] for p, cls in inv['documented_counts'].items() for c, n in cls.items()])
    print('image formats:', inv['modes'], '| undecodable:', len(inv['undecodable']), '|', inv['specimen_or_session_metadata'])
    print('official split of product A defects -> this notebook\\'s split:', DATA['split']['official_vs_custom'])
g = DATA['grouping']
print(f"groups: {g['groups']:,}; exact-duplicate links: {g['exact_duplicate_links']}; near-duplicate pairs (corr >= {g['threshold']}): {g['near_duplicate_pairs']}; "
      f"pairs just below the threshold: {g['pairs_just_below_threshold']['count']}")
print(f"label-conflict duplicates excluded: {g['label_conflict_groups']} groups ({len(g['label_conflict_images'])} images): {g['label_conflict_images']}")
print('similar pairs (corr >= 0.95) that cross partitions, by label pair:', g['similar_pairs_crossing_partitions']['by_label_pair'])
order = DATA['class_order']
table(['partition', *order], [[s, *[DATA['split']['counts'].get(s, {}).get(c, 0) for c in order]] for s in ('train', 'val', 'test', 'excluded')])
print(f"selected training budget: {DATA['selected']}; unused training images: {DATA['unused_training_images']:,}")
print(f"split manifest SHA-256 {DATA['split']['manifest_sha256'][:16]}...; matches the reference audit: {DATA['split']['matches_reference']}")
show('training_examples.png')
show('near_duplicate_pairs.png')
"""

S2_NOTICE = """
**What to notice.**

- The defect counts match the published notes, but the normal counts do not (product A has 66 fewer normal images than documented; product C about 1,490 fewer). The notebook reports this; it does not guess why.
- Five pairs of images are pixel-for-pixel identical yet labelled `scratches` in one copy and `spots` in the other, and several of those pairs sit in *different* official partitions. Those ten images are excluded as label conflicts: neither label can be trusted, and keeping them would put the same picture in two partitions.
- The near-duplicate threshold (correlation 0.98 of 32 × 32 thumbnails) is a declared cut, not a natural boundary; the printout shows how many pairs sit just below it and how many similar pairs cross partitions. Most crossing pairs are normal/normal, because these surfaces share one texture.
- The test set has only about 21 spot images. Every spot metric later rests on those few pictures.
- `matches the reference audit: True` means your split is byte-identical to the one this notebook was written against. `False` means grouping or assignment changed (for example a different image decoder); the run continues with its own frozen manifest and records the difference.
"""

S2_CHECK = checkpoint(
    "Why is a random image-level split risky for this dataset, and what does grouping by duplicates not protect against?",
    "Consecutive capture frames and duplicated files are near-copies. With an image-level random split, a near-copy of a test image can sit in training, so the classifier is partly tested on what it memorised and scores look better than they would on new parts. Grouping keeps each detected cluster in one partition. It cannot detect similarity below the threshold, and it cannot tell whether two dissimilar images come from the same physical part, line or session, because the source supplies no such metadata. That is why the notebook calls this an image-group benchmark and not a specimen-level one.",
)

S3_INTRO = """
## 3 · Baselines first: majority class and a real-only classifier  <sub>(core concept)</sub>

Before any generated image exists, it is worth knowing what "good" looks like. The feature stage turns each selected training image into a 512-number vector with a frozen ImageNet ResNet-18 (He et al., 2016), plus four fixed augmented views of each image for the conventional arm; validation images get one deterministic vector each. Preprocessing is the same everywhere: grayscale, letterbox to 224 × 224 without cropping, replicate to three channels, ImageNet normalisation.

Then a **preview** fits the linear head for arm A only (real images, no augmentation) and scores it on **validation**. The majority baseline always predicts `normal`.

**Predict first:** the majority baseline's accuracy on validation will be around what number? Its macro-F1?
"""

S3_RUN = """
run_stage('features_real', 'lab', 'stage_features.py', '--part', 'real')
show('augmentation_views.png')
run_stage('preview', 'lab', 'stage_fit.py', '--preview')
PREVIEW = load('extensions/real_only_preview.json')
rows = []
for arm in ('majority', 'A'):
    for seed, r in PREVIEW['results'][arm].items():
        v = r['val']
        rows.append([arm, seed, fmt(v['accuracy']), fmt(v['balanced_accuracy']), fmt(v['macro_f1']), *[fmt(v['per_class'][c]['recall']) for c in order]])
table(['arm', 'seed', 'val accuracy', 'val balanced acc.', 'val macro-F1', *[f'recall {c}' for c in order]], rows)
"""

S3_NOTICE = """
**What to notice.** The majority baseline's accuracy is about 0.93 while its macro-F1 is about 0.32 (F1 of roughly 0.96 for `normal` and 0 for both defects, averaged). High accuracy with zero defect recall is exactly the failure an inspection system cannot afford. The real-only head trades some `normal` accuracy for defect recall. Look at the augmentation sheet: a flip or a ±10% brightness/contrast change leaves every scratch and spot visible, which is the evidence that these transforms preserve labels. The three seeds of arm A differ only in the head's initial weights and the order of training draws, so their spread is a first look at noise that has nothing to do with the method.

These are validation numbers from the preview. The canonical arms are fitted again in Section 8 with identical settings, and only Section 9 touches the test set.
"""

S3_CHECK = checkpoint(
    "A colleague reports 94% accuracy for a defect classifier on this dataset. What would you ask before being impressed?",
    "Ask for per-class recall and macro-F1, the class distribution of the evaluation set, and whether the evaluation images were held out by group. At about 93% `normal`, a classifier that finds no defects already scores 93% accuracy, so 94% could mean almost nothing.",
)

S4_INTRO = """
## 4 · Phi-4 describes training exemplars: observed versus guessed  <sub>(core concept)</sub>

Phi-4-multimodal-instruct (Abouelenin et al., 2025) is a 5.6 B-parameter model that reads images and text. Here it is **frozen** and loaded 4-bit (NF4, float16 compute; the vision encoder, projector, embeddings and the checkpoint's own LoRA stay float16). It sees three exemplars per class, all drawn from the selected training images, and it is **not told the label**. Decoding is greedy, at most 96 new tokens per image.

The fixed instruction asks for visible texture, illumination and the shape of any mark, offers `material: unknown`, and forbids guessing alloy, process or severity:

> Describe only what is visible in this grayscale close-up photograph of a surface, in at most three short sentences. Mention the texture, the illumination, and the shape and size of any visible mark. If you cannot tell what the material is, write 'material: unknown'. Do not guess the alloy, the manufacturing process or how severe any mark is.

**Trust boundary.** The checkpoint `microsoft/Phi-4-multimodal-instruct` at the immutable revision `93f923e1…` implements its model, configuration and processor in five Python files inside the model repository, and `transformers 4.48.2` has no built-in implementation, so `trust_remote_code=True` is unavoidable. The stage refuses to run unless `ALLOW_PHI4_REMOTE_CODE` is `True`, pins every file (weights, configuration, tokenizer and the five code files) by SHA-256, and verifies them before `transformers` imports anything. Pinning bounds *which* code runs; it does not make the code safe by itself. The model is MIT-licensed. Its weights (11.2 GB) are deleted after this stage by default.
"""

S4_RUN = """
run_stage('phi4', 'phi4', 'stage_phi4.py')
CAPTIONS = load('caption_records.jsonl')
table(['class', '#', 'image', 'tokens', 'stop reason', 'description'], [[r['label'], r['exemplar_rank'], r['image_id'], r['new_tokens'], r['stop_reason'], r['response'].replace('|', '/')] for r in CAPTIONS])
"""

S4_NOTICE = """
**What to notice.** Read each description and sort its claims into three bins: *observed* (a line, a bright blob, horizontal banding, even lighting), *inferred but plausible* (a "scratch" where a thin bright line is visible), and *guessed* (a material, a process, a cause, a severity). Descriptions of `normal` images sometimes report a mark, and defect descriptions sometimes miss one; the model was not told the label. The `stop reason` column says how each description ended: `end_of_sequence` means the model finished; `max_new_tokens` means it was cut off at the 96-token limit (a description can use all 96 tokens and still end on its own, so the token count alone does not show truncation). Greedy decoding makes the text repeatable for the same model and inputs, not correct.
"""

S4_CHECK = checkpoint(
    "Why is it important that Phi-4 only saw selected training images, and never validation or test images?",
    "Anything that shapes the generated images (descriptions, prompts, which candidates are kept) is part of training. If held-out images informed those choices, information about the test set would flow into training through the prompts, and the final comparison would no longer be an independent test.",
)

S5_INTRO = """
## 5 · From descriptions to bounded prompts  <sub>(engineering)</sub>

Model output never becomes a prompt verbatim. `compile_prompts` (core section *Descriptions to bounded prompts*) searches each description for a fixed whitelist of texture, illumination and shape terms and fills a fixed template:

`close-up grayscale industrial inspection photograph of a flat manufactured surface, <texture>, <illumination>, showing one <defect> defect as a <shape> mark, sharp focus, no text`

Background terms come from normal exemplar *k*, mark terms from defect exemplar *k*, giving three prompts per defect class. A description that is empty, too long or contains no whitelisted term falls back to the neutral class template, recorded as a fallback and never described as Phi-4-guided. The template never names a material. The prompts are written to `prompts.json` and are frozen from here on.
"""

S5_RUN = """
run_stage('prompts', 'lab', 'stage_prompts.py')
PROMPTS = load('prompts.json')['prompts']
table(['prompt', 'guided by Phi-4', 'fallback reason', 'text'], [[p['prompt_id'], p['phi4_guided'], p['fallback_reason'] or '', p['text']] for p in PROMPTS])
"""

S5_NOTICE = """
**What to notice.** The whitelist deliberately throws information away: a rich description and a terse one can compile to the same prompt. That is the price of a bounded, auditable interface between two models. If most prompts are fallbacks, the Phi-4 step contributed little, and the notebook says so rather than calling the result Phi-4-guided.
"""

S5_CHECK = checkpoint(
    "A description says 'a deep gouge in brushed aluminium caused by tool chatter'. What can reach the prompt, and why is that a deliberate loss?",
    "Only whitelisted terms can: here `brushed` (texture), and possibly a shape word if one is present. 'Aluminium', 'tool chatter' and 'deep' are guesses about material, cause and severity that the image cannot establish; pasting them into a prompt would let unverified claims shape the synthetic data. The bounded template trades expressiveness for an interface that can be audited and cannot carry instructions or free text from one model into another.",
)

S6_INTRO = """
## 6 · FLUX generates candidate defects  <sub>(core concept)</sub>

FLUX.1 [schnell] (Black Forest Labs, 2024) is a 12 B-parameter text-to-image model distilled to produce an image in about four steps without guidance. The weights are the ungated mirror `unsloth/FLUX.1-schnell` at revision `9df3faa7…` of the identity of record `black-forest-labs/FLUX.1-schnell` at `741f7c3c…` (Apache-2.0); every one of the 23 files is verified by SHA-256 before loading. The stage never keeps the text encoders and the transformer on the GPU together: it loads CLIP-L and T5-XXL (float16), encodes the six prompts, keeps the embeddings, releases the encoders, then loads the transformer 4-bit (NF4) with a float32 VAE.

Exactly **32 candidates per defect class** are generated at 512 × 512, 4 steps, guidance 0, from the fixed seeds 100000 + *i* (scratches) and 200000 + *i* (spots); candidate *i* uses prompt *i* mod 3. Every attempt is kept with its disposition. Default eligibility checks only that the image decodes, has the right size, has finite pixel values and is not an exact duplicate of an earlier candidate or of a selected training image. It never uses classifier confidence, CLIP scores, test performance or Phi-4 approval, because filtering on any of those would quietly select for whatever makes the result look good. If either class ends with fewer than 24 usable candidates, the notebook reports a generation failure and does not present a synthetic comparison. It does not keep generating until the pictures look right.

**Predict first:** will the generated images look like the Bosch photographs? What will differ?
"""

S6_RUN = """
run_stage('flux', 'lab', 'stage_flux.py')
GEN = load('generation_manifest.jsonl')
STATUS = load('generation_status.json')
print('usable candidates per class:', STATUS['usable'], '| complete:', STATUS['complete'], '|', '; '.join(STATUS['shortfall']) or 'no shortfall')
table(['status', 'reason', 'count'], [[s, r or '', sum(1 for g in GEN if g['status'] == s and g['reason'] == r)] for s, r in sorted({(g['status'], g['reason']) for g in GEN})])
for label in order[1:]:
    show(f'generated_{label}.png')
"""

S6_NOTICE = """
**What to notice, and questions to answer while looking (execution does not pause):**

1. Do the candidates share the Bosch images' horizontal banding, grain and lighting, or do they look like a generic "metal surface"? This gap between generated and real images is **domain mismatch**.
2. In how many `scratches` candidates can you actually see a scratch? A spot? Is any candidate plainly the wrong class, or defect-free?
3. The label under each thumbnail is the *intended* label. If you disagree with some, that is label noise the classifier will learn from in arm C.
4. Would you trust these images more if they looked more realistic? Why is realism not the same as usefulness?

An optional human-review extension (Section 13) lets you record accept / reject / uncertain for each candidate and run a separately identified experiment. It never changes the canonical results.
"""

S6_CHECK = checkpoint(
    "Why does the notebook refuse to filter candidates by how confidently a classifier recognises them?",
    "Filtering by classifier confidence keeps only the generated images that already look like what a classifier expects. That makes the synthetic pool easier and more similar to the real training data, and it uses a model's judgement where the experiment claims to be testing unfiltered generation. Tuned against validation or test scores, it would also leak held-out information into training. A fixed, minimal eligibility rule keeps the pool honest; human review is allowed only as a separately labelled extension.",
)

S7_INTRO = """
## 7 · Synthetic features and the synthetic-to-training audit  <sub>(evaluation practice)</sub>

Each eligible candidate is converted to grayscale, letterboxed from 512 px to 224 px and passed through the same frozen ResNet-18, with four fixed augmented views, exactly as the real images were. Before any fitting, every candidate's thumbnail is compared with every **training-partition** image: a correlation of 0.98 or more would mean FLUX reproduced a real training photograph. Matches are reported, not used to filter.
"""

S7_RUN = """
run_stage('features_synthetic', 'lab', 'stage_features.py', '--part', 'synthetic')
AUDIT = load('synthetic_training_overlap.json')
print(f"highest candidate-to-training correlation: {fmt(AUDIT['max_correlation'])}; flagged at >= {AUDIT['threshold']}: {len(AUDIT['flagged'])}")
"""

S7_NOTICE = """
**What to notice.** A text-to-image model that never saw these photographs is not expected to reproduce one, so the highest correlation should sit well below 0.98. It is still worth checking, because a generator trained on web images may have seen public datasets; the notebook cannot rule out pretraining overlap for Phi-4, FLUX or ResNet-18.
"""

S8_INTRO = """
## 8 · Three matched arms, validation selection and a frozen experiment  <sub>(evaluation practice)</sub>

This is the controlled comparison. For each seed (17, 29, 43), `slot_schedule` (core section *Matched training schedule*) draws every random number the three arms need **once**: 30 epochs × 32 updates × 32 slots = 30,720 training slots, each batch holding 11 / 11 / 10 slots of the three classes, rotating which class gets 10. Every slot has a class, a real-image index, a view index, a coin and a synthetic-image index. The arms then differ only in where a slot's features come from:

- **A** uses the real image's deterministic features;
- **B** uses one of the real image's four fixed augmented views;
- **C** does the same as B, except that a **defect** slot whose coin is below 0.5 uses a synthetic candidate's view instead. Normal slots are always real.

So every arm sees the same number of updates, the same class balance and the same head initialisation per seed. C does not get more training; it gets a different distribution of defect examples. The head is fitted with AdamW (learning rate 0.001, weight decay 0.0001), batch 32, on the CPU with deterministic algorithms. After each epoch it is scored on **validation**; the epoch with the highest validation macro-F1 is kept (the earlier epoch wins a tie). Among the arms fitted with the canonical seed 17, the one with the highest validation macro-F1 is chosen for export (ties go to A, then B, then C).

Then the experiment is **frozen**: `experiment_config.json` records every setting and the SHA-256 of every input (split, prompts, generation manifest, features, heads, the thumbnail signatures used by the overlap audit) and of the code that defines preprocessing, fitting, evaluation and export (the shared core and every stage script). The test, export and reload stages refuse to run if any of them changes, and every held-out image is re-checked against its recorded digests before it is scored.

**Predict first:** on validation, will C beat B for every seed, some seeds, or none?
"""

S8_RUN = """
run_stage('fit', 'lab', 'stage_fit.py')
RECORD = load('experiment_config.json')
table(['arm', 'seed', 'kept epoch', 'val macro-F1', 'val balanced acc.', 'synthetic share of defect slots'],
      [[a, s, r['selected_epoch'], fmt(r['val']['macro_f1']), fmt(r['val']['balanced_accuracy']), fmt(r['synthetic_share_of_defect_slots'], 2)]
       for a in RECORD['arms_run'] for s, r in RECORD['validation'][a].items()])
print('export choice:', RECORD['selection'])
print(f"frozen: {len(RECORD['frozen_files'])} inputs pinned, record SHA-256 {RECORD['record_sha256'][:16]}...; smoke run: {RECORD['smoke_run']}")
"""

S8_NOTICE = """
**What to notice.** About half of the defect slots in arm C are synthetic, as designed; A and B have none. Kept epochs vary by arm and seed. Validation scores were used to choose epochs and the export, so they are slightly optimistic and are *not* the evidence for the question; the test set is. If generation fell short, arm C is absent and the notebook says so instead of comparing.
"""

S8_CHECK = checkpoint(
    "Why not simply add the 64 synthetic images to the training set of arm B and train for the same number of epochs?",
    "Adding images changes several things at once: the number of defect examples, the class balance, and (with a fixed number of epochs) the number of updates. Any difference in the result could come from any of these. Substituting synthetic images into a fixed slot schedule keeps updates, batch composition, class prior and random draws identical, so a difference between C and B can be attributed to the change in the defect-image distribution.",
)

S9_INTRO = """
## 9 · One final comparison on real test photographs  <sub>(evaluation practice)</sub>

The test stage first verifies the frozen record, then extracts features for the test partition (never seen until now), scores every arm and seed once, and computes:

- **macro-F1** (primary), balanced accuracy, accuracy, and per-class precision, recall, F1 and support for each arm and seed;
- the **C − B** contrast (primary) and **B − A** (context): the mean over seeds of the macro-F1 difference, with an approximate 95% interval from 1,000 **paired, class-stratified group bootstrap** resamples of the test set (every arm is scored on the same resamples; groups, not images, are resampled);
- after freezing, an **overlap audit** of every synthetic candidate against validation and test images and a check that no exact duplicate crosses partitions. A consequential match marks the comparison invalid; it is never quietly removed.

The decision rule is argmax over the head's logits. Softmax scores in `predictions.csv` are uncalibrated and are not probabilities of a defect.

Scoring the test set again after changing something, and picking the better result, would turn it into a second validation set. The notebook therefore scores it once.
"""

S9_RUN = """
run_stage('evaluate', 'lab', 'stage_evaluate.py')
METRICS = load('metrics.json')
print('test counts:', METRICS['test_counts'], '| comparison valid:', METRICS['comparison_valid'], METRICS['validity_note'])
rows = []
for arm, summ in METRICS['summary'].items():
    rows.append([METRICS['arms'][arm], *[fmt(summ['macro_f1']['per_seed'][str(s)]) for s in RECORD['seeds']], fmt(summ['macro_f1']['mean']),
                 f"{fmt(summ['macro_f1']['min'])}–{fmt(summ['macro_f1']['max'])}", fmt(summ['balanced_accuracy']['mean']), fmt(summ['accuracy']['mean'])])
table(['arm', *[f'macro-F1 seed {s}' for s in RECORD['seeds']], 'mean', 'range', 'balanced acc. (mean)', 'accuracy (mean)'], rows)
for name, c in METRICS['contrasts'].items():
    lo, hi = c['interval_95']
    verdict = 'the interval includes zero: no clear difference in this run' if lo <= 0 <= hi else ('the interval is above zero' if lo > 0 else 'the interval is below zero')
    print(f"{name}: mean difference {c['mean_difference']:+.4f} (per seed {', '.join(f'{v:+.4f}' for v in c['per_seed'].values())}); "
          f"approximate 95% interval [{lo:+.4f}, {hi:+.4f}] -> {verdict}")
show('macro_f1_by_arm.png')
show('confusion_matrices.png')
"""

S9_NOTICE = """
**What to notice.** Read the per-seed columns before the mean: if the sign of C − B changes between seeds, the mean hides disagreement. Compare the interval's width with the difference itself. The interval is **conditional on this split and this generated pool**: it reflects which test groups happened to be drawn, not what would happen with a different split, a new set of 64 generated images or another factory. In the confusion matrices, look at where each arm's errors go: false alarms on `normal`, missed defects, or scratch/spot confusion.
"""

S9_CHECK = checkpoint(
    "Why does the bootstrap resample groups rather than individual images, and why are the arms scored on the same resamples?",
    "Images in one group are near-copies, so they are not independent evidence; resampling them one by one would pretend there is more independent data than there is and make the interval too narrow. Scoring every arm on the same resample pairs the comparison: the difference C − B is computed on identical test images each time, so variation that affects both arms equally (an unusually hard set of spot images) cancels out instead of widening the interval.",
)

S10_INTRO = """
## 10 · Error analysis, uncertainty and a bounded conclusion  <sub>(interpretation)</sub>

Now that the experiment is frozen, test images may be inspected. The gallery shows misclassified test images for the canonical seed in arms B and C.
"""

S10_RUN = """
show('error_gallery.png')
rows = []
for arm in METRICS['per_run']:
    r = METRICS['per_run'][arm][str(RECORD['selection']['seed'])]
    for c in order:
        pc = r['per_class'][c]
        rows.append([arm, c, fmt(pc['precision']), fmt(pc['recall']), fmt(pc['f1']), pc['support'], pc['predicted']])
table(['arm (canonical seed)', 'class', 'precision', 'recall', 'F1', 'support', 'predicted'], rows)
undefined = {a: r[str(RECORD['selection']['seed'])]['undefined'] for a, r in METRICS['per_run'].items() if r[str(RECORD['selection']['seed'])]['undefined']}
print('undefined metrics (no predictions of that class):', undefined or 'none')
"""

S10_NOTICE = f"""
**What to notice.** Precision for the majority baseline's defect classes is undefined (it never predicts them) and is reported as such, not as zero. A change of one or two spot predictions moves spot recall by about 0.05, because the test set has about 21 spots.

{checkpoint("Suppose C − B is +0.02 with an interval of [−0.01, +0.05]. Your manager asks: 'So synthetic defects work?' What do you answer?", "Not established. On this split and this generated pool, the average difference favours synthetic substitution slightly, but the interval includes zero and the result may reverse with other test groups; it says nothing about other generated pools, other products or production. The honest statement is that this run found no clear evidence either way, and the next step is repeating the experiment with independently generated pools and another product (the P1 items in Section 13).")}

### Write your conclusion

Complete these sentences in your own words, using only numbers printed above:

1. **Task:** *On product A of the Bosch SDI dataset (or: on my own dataset, described by its source and classes), with 128 / 64 / 32 real training images and a frozen ResNet-18 with a linear head, we compared …*
2. **Principal result:** *Mean test macro-F1 was … for A, … for B and … for C; C − B was … (approximate 95% interval …), with per-seed differences of …*
3. **Baseline / reference:** *The majority baseline scored accuracy … but macro-F1 …*
4. **Important failure mode or uncertainty:** *Most remaining errors were …; the spot class rests on … test images; …*
5. **Limitations:** *One grouped split of one product without specimen metadata; one generated pool with intended, unverified labels; frozen models in 4-bit; no calibration; …*
6. **What would change my mind:** *…*

Your prediction from the start: was it right? What in the evidence would have to differ for the opposite conclusion?
"""

S11_INTRO = """
## 11 · Export, fresh-process reload and new-image inference  <sub>(engineering)</sub>

The exported classifier is the validation-selected arm for the predeclared canonical seed 17, chosen from validation scores only. The artifact in `classifier/` contains `head.safetensors` (weight 3 × 512 and bias, float32) and `manifest.json` (format, class order and indices, decision rule, backbone identity with revision and SHA-256, preprocessing definition, selection record, file digests). No images or features are embedded.

The export stage also computes reference logits and the decisions they imply for nine fixed real test images on the CPU. The reload stage is a **new process** that imports nothing from training. It checks the manifest, the file set and every digest; checks the artifact's **meaning** against the frozen experiment (class names, order and indices, decision rule, preprocessing, backbone, the selected head and the experiment record); loads the head with `safetensors` (plain arrays; nothing is unpickled); rebuilds ResNet-18 from its verified weights; re-verifies each reference image against its recorded digests; recomputes the nine logits; and requires them to match within an absolute tolerance of 1e-5 **and** to name the same class for every example. It then scores new images: up to six unused training-partition images (two per class) that were never used for fitting, captioning or generation, plus any images in `NEW_IMAGE_DIR`. If your data has no spare training images and `NEW_IMAGE_DIR` is empty, this step is recorded as not run and the CSV has only its header.
"""

S11_RUN = """
run_stage('export', 'lab', 'stage_export.py')
run_stage('reload', 'lab', 'stage_reload.py')
PARITY = load('reload_parity.json')
print(f"reload parity: max |logit difference| {PARITY['max_abs_logit_difference']:.2e} <= {PARITY['tolerance']:.0e}; replayed decisions identical: {PARITY['same_predictions']} -> {PARITY['passed']}")
NEW = PARITY['new_image_inference']
print(f"new-image inference: {NEW['status']} ({NEW['scored']} scored of {NEW['candidates']} candidates, {len(NEW['rejected'])} rejected)")
"""

S11_NOTICE = """
**What to notice.** Parity compares numbers, not just file loading. The new-image table has one row per image with a known label where available, the predicted class and one uncalibrated score per class in the artifact's class order. Unused training-partition images are distinct from the fitted images but come from the same partition; they are an inference demonstration, not additional evidence.
"""

S11_CHECK = checkpoint(
    "The reload stage loaded the files without error. Why is that not enough, and what does the parity check add?",
    "Loading proves that files parse, not that they reproduce the classifier that was evaluated. Two different checks are needed. Numbers: recomputing logits for fixed real images in a new process and requiring agreement within 1e-5 shows the tensors, the pinned backbone file and the preprocessing code reproduce the exported outputs. Meaning: a reversed class list would leave every logit identical while swapping what the columns are called, so numeric parity cannot catch it. The reload stage therefore also checks the manifest against the frozen experiment (class order and indices, decision rule, preprocessing, backbone, selected head, experiment record) and requires every replayed decision to name the same class. What neither check covers: whether the preprocessing is right for new kinds of images, or whether the scores are calibrated.",
)

S12_INTRO = """
## 12 · Records: environment, run summary, limitations and attribution  <sub>(engineering)</sub>

The last default cell writes the remaining records and prints a terminal summary. `run_summary.json` holds stage times, peak child memory, peak GPU memory in use, the lowest available host memory and disk use for every stage, measured in this runtime; they are measurements of this run, not guarantees for another.
"""

S12_RUN = r'''
# @title Write environment, run summary, limitations and attribution records
FLUX_STATE = json.loads((WORK / 'state' / 'flux_stage.json').read_text())
PHI4_STATE = json.loads((WORK / 'state' / 'phi4_stage.json').read_text())
FEATURE_STATE = json.loads((WORK / 'state' / 'features_stage.json').read_text())
environment = {'runtime': RUNTIME, 'environments': ENVIRONMENTS, 'uv': UV_VERSION,
               'notebook': {'name': 'DIMER_Bosch_Synthetic_Defect_Augmentation_Capstone.ipynb', 'profile': 'E2E', 'mode': 'GUIDED', 'notebook_spec': '2.2'},
               'models': {'phi4': PHI4_STATE['model'], 'flux': FLUX_STATE['model'], 'backbone': FEATURE_STATE['backbone']},
               'precision': {'phi4': 'nf4 weights, float16 compute', 'flux': FLUX_STATE['precision'], 'backbone': 'float32', 'head': 'float32 on CPU, deterministic algorithms'}}
(OUT / 'environment.json').write_text(json.dumps(environment, indent=2))
download_bytes = {'dataset': DATA['acquisition'].get('download', {}).get('downloaded_bytes'), 'phi4': PHI4_STATE['staging']['downloaded_bytes'], 'flux': FLUX_STATE['staging']['downloaded_bytes']}
C_B = METRICS['contrasts'].get('C - B')
summary = {
    'question': RECORD['question'], 'data_source': DATA_SOURCE, 'seeds': RECORD['seeds'], 'smoke_run': RECORD['smoke_run'],
    'split_sha256': DATA['split']['manifest_sha256'], 'split_matches_reference': DATA['split']['matches_reference'],
    'generation': STATUS, 'phi4_guided_prompts': sum(p['phi4_guided'] for p in PROMPTS), 'prompts': len(PROMPTS),
    'test_macro_f1_mean': {a: s['macro_f1']['mean'] for a, s in METRICS['summary'].items()},
    'contrasts': {k: {'mean_difference': v['mean_difference'], 'interval_95': v['interval_95'], 'per_seed': v['per_seed']} for k, v in METRICS['contrasts'].items()},
    'comparison_valid': METRICS['comparison_valid'], 'exported': RECORD['selection'], 'reload_parity': PARITY,
    'frozen_record_sha256': RECORD['record_sha256'], 'stages': STAGE_LOG, 'download_bytes': download_bytes,
    'peak_gpu_allocated_gib': {'phi4': PHI4_STATE['peak_gpu_allocated_gib'], 'flux_encoders': FLUX_STATE['encoder_peak_gpu_gib'], 'flux_transformer': FLUX_STATE['transformer_peak_gpu_gib']},
    'model_time_seconds_excluding_installs': round(sum(r['seconds'] for r in STAGE_LOG.values()), 1),
    'wall_seconds_since_configuration': round(time.time() - NOTEBOOK_STARTED, 1),
    'evidence': 'tutorial evidence from one grouped split and one generated pool; not a benchmark or production result',
}
(OUT / 'run_summary.json').write_text(json.dumps(summary, indent=2))
G = DATA['grouping']
if DATA_SOURCE == 'bosch':
    scope_line = '- One custom, deterministic 60/20/20 grouped split of product A. Groups come from exact and perceptual duplicates only; the source supplies no specimen or session identifiers, so this is an exploratory image-group benchmark, not evidence about independent physical specimens.'
else:
    declared = DATA['inventory'].get('declared_groups', 0)
    scope_line = (f"- One deterministic 60/20/20 grouped split of the user-supplied dataset ({DATA['inventory']['entries']} images). "
                  + (f'{declared} images carried a declared group; groups were also merged by exact and perceptual duplicates.' if declared else 'No groups were declared, so groups come from exact and perceptual duplicates only and the evaluation is an exploratory image-group benchmark.'))
limitations = [
    '# Limitations of this run', '',
    scope_line,
    f"- The test partition holds {METRICS['test_counts']} images; per-class metrics for the rarest class rest on few images.",
    '- One generated pool of 32 candidates per defect class from fixed seeds. Generated labels are the prompt\'s intended labels and were not validated by anyone. Results may differ for another pool.',
    '- Phi-4 and FLUX run frozen in 4-bit (NF4); every number is the 4-bit models\'. ' + ('Pretraining overlap of Phi-4, FLUX or ResNet-18 with this public dataset cannot be excluded.' if DATA_SOURCE == 'bosch' else 'If your images are, or resemble, publicly available images, pretraining overlap of Phi-4, FLUX or ResNet-18 with them cannot be excluded.'),
    '- The bootstrap interval is approximate and conditional on this split and pool; it does not include training or generation variability.',
    '- Scores are softmax of uncalibrated logits under an argmax rule; no deployment threshold was chosen or validated.',
    *(['- Documented dataset counts differ from the measured archive for normal images.'] if DATA_SOURCE == 'bosch' else []),
    f"- {len(G['label_conflict_images'])} images in {G['label_conflict_groups']} label-conflict duplicate groups were excluded.",
    '- This is not a production inspection system and makes no claim about factory-level performance.',
]
if RECORD['smoke_run']:
    limitations.append('- Fewer than three seeds were run: this is a smoke run, not the canonical comparison.')
if not STATUS['complete']:
    limitations.append('- Generation fell short (' + '; '.join(STATUS['shortfall']) + '); no synthetic comparison is reported.')
if not METRICS['comparison_valid']:
    limitations.append('- ' + METRICS['validity_note'])
(OUT / 'limitations.md').write_text('\n'.join(limitations) + '\n')
if DATA_SOURCE == 'bosch':
    dataset_terms = ('`split_manifest.csv` and the figures that show dataset images are adapted dataset material: if you redistribute them, credit the source above and apply CC BY-SA 4.0.\n\n'
                     'Cite the dataset\'s accompanying paper: Wang, R., Hoppe, S., Monari, E., & Huber, M. F. (2023). Defect Transfer GAN: Diverse defect synthesis for data augmentation. https://doi.org/10.48550/arXiv.2302.08366')
    dataset_row = '| Dataset | Bosch Surface Defect Inspection dataset, https://github.com/boschresearch/The-Surface-Defect-Inspection-Dataset at commit c6e0afe66e9dbd9d99326c986ea884ddf2aad9f8, archive SHA-256 d33ea340151cdf909f3807a37e00d335c66eb8960b0c2eb568fd9fc75d96fa7c | CC BY-SA 4.0 (attribution; share-alike for adapted material) |'
else:
    dataset_terms = ('The dataset was supplied by the user. Its terms are set by its owner and are not determined by this notebook; `split_manifest.csv` and the figures '
                     'that show its images carry those terms. No Bosch dataset material was used in this run.')
    dataset_row = f"| Dataset | user-supplied BYOD zip, SHA-256 {DATA['acquisition'].get('byod_zip_sha256')} | set by the data owner; not determined by this notebook |"
attribution = f"""# Attribution and licence record

Each component keeps its own licence; they are not assumed to share one.

| Component | Identity | Licence |
|---|---|---|
{dataset_row}
| Description model | {PHI4_STATE['model']['id']} @ {PHI4_STATE['model']['revision']} | {PHI4_STATE['model']['license']} |
| Generator | {FLUX_STATE['model']['id']} @ {FLUX_STATE['model']['revision']}, bytes from {FLUX_STATE['model']['staging']['repo']} @ {FLUX_STATE['model']['staging']['revision']} | {FLUX_STATE['model']['license']} |
| Feature extractor | {FEATURE_STATE['backbone']['id']} @ {FEATURE_STATE['backbone']['revision']} | {FEATURE_STATE['backbone']['license']} |
| Notebook code | kurtvalcorza/flux-schnell-generation-pipeline | see the repository's LICENSE |

Changes made to dataset material in this run: {DATA['experiment_scope']}; {len(G['label_conflict_images'])} label-conflict duplicate images excluded; images converted to three-channel 224 x 224 letterboxed tensors for feature extraction; augmented views (horizontal flip, brightness and contrast within +/-10%).

{dataset_terms}

Generated images in `generated/` were produced by FLUX.1 [schnell] from prompts compiled from descriptions of the dataset images above. Their licensing depends on the model licence, the source images' terms and how you use them; this record does not decide that question for you.
"""
(OUT / 'ATTRIBUTION.md').write_text(attribution)
(OUT / 'completion_record_template.md').write_text('# Completion record (a learning aid, not a submission)\n\n- My prediction before running:\n- C - B (mean, interval, per seed):\n- What the majority baseline taught me:\n- One description claim that was guessed, not observed:\n- One generated candidate I would reject, and why:\n- My bounded conclusion:\n- What I would run next:\n')
print('=' * 100)
print(f"DATA        {DATA_SOURCE}: split {DATA['split']['manifest_sha256'][:12]}... (reference match: {DATA['split']['matches_reference']}), budget {DATA['selected']}")
print(f"PROMPTS     {summary['phi4_guided_prompts']} of {len(PROMPTS)} Phi-4-guided | GENERATION usable {STATUS['usable']} complete={STATUS['complete']}")
print('TEST MACRO-F1 (mean over seeds) ' + ', '.join(f'{a}={v:.4f}' for a, v in summary['test_macro_f1_mean'].items()))
for name, c in METRICS['contrasts'].items():
    print(f"CONTRAST    {name}: {c['mean_difference']:+.4f}, approx. 95% interval [{c['interval_95'][0]:+.4f}, {c['interval_95'][1]:+.4f}]")
print(f"VALIDITY    comparison valid: {METRICS['comparison_valid']} | smoke run: {RECORD['smoke_run']}")
print(f"EXPORT      arm {RECORD['selection']['arm']} seed {RECORD['selection']['seed']} | reload parity {PARITY['passed']} (max diff {PARITY['max_abs_logit_difference']:.1e}, same decisions {PARITY['same_predictions']}) | new-image inference {PARITY['new_image_inference']['status']}")
print(f"RESOURCES   model stages {summary['model_time_seconds_excluding_installs']} s; wall {summary['wall_seconds_since_configuration']} s; "
      f"peak GPU in use {max(r['peak_gpu_used_mib'] for r in STAGE_LOG.values())} MiB; peak child RSS {max(r['peak_child_rss_mib'] for r in STAGE_LOG.values())} MiB; "
      f"lowest host memory available {min(r['min_host_available_mib'] for r in STAGE_LOG.values() if r['min_host_available_mib'] is not None)} MiB")
print(f'OUTPUTS     {OUT}')
for path in sorted(p for p in OUT.rglob('*') if p.is_file() and 'generated' not in p.parts):
    print('   ', path.relative_to(OUT))
print('=' * 100)
'''

S13_INTRO = """
## 13 · Optional activities, troubleshooting and further exploration  <sub>(optional)</sub>

None of these is needed for the default path, and a failure here does not affect the results above. The synthetic-fraction activity (13.1) is switched on by default because it reuses the features already computed, touches only validation data and takes seconds; set `RUN_SYNTHETIC_FRACTION_EXERCISE = False` to skip it. Its results are written to `extensions/` and are not part of `run_summary.json`, which was completed in Section 12. Both 13.1 and 13.2 are skipped with an explanation when the canonical arm C was not run.

### 13.1 Change one thing: the synthetic fraction (Predict → Change → Run → Observe → Explain)

**Predict:** if each defect slot in arm C used a synthetic image with probability 0.25 instead of 0.5, would validation macro-F1 go up or down? **Change:** set `EXERCISE_SYNTHETIC_PROBABILITY` in the configuration cell (default 0.25; try 0.75 or 1.0). **Run** the next cell. **Observe** the per-seed validation macro-F1 against the canonical C. **Explain** the pattern in terms of domain mismatch and label noise.

This activity uses **validation only** and only after the canonical result exists. Re-scoring the test set for each variant would turn repeated inspection of one test set into model selection; any such result would be exploratory, not new confirmatory evidence.
"""

S13_EXERCISE = """
if RUN_SYNTHETIC_FRACTION_EXERCISE and 'C' not in RECORD['arms_run']:
    print('Skipped: the canonical arm C was not run (' + '; '.join(RECORD['generation_status']['shortfall']) + '), so there is no canonical C to compare with. '
          'Fitting an exploratory C here would be a redesigned study, not a completion of the omitted condition.')
elif RUN_SYNTHETIC_FRACTION_EXERCISE:
    p = float(EXERCISE_SYNTHETIC_PROBABILITY)
    if not 0.0 <= p <= 1.0:
        raise ValueError('EXERCISE_SYNTHETIC_PROBABILITY must be between 0 and 1')
    tag = f'exercise_p{p:g}'
    run_stage(tag, 'lab', 'stage_fit.py', '--synthetic-probability', p, '--tag', tag)
    EXERCISE = load(f'extensions/{tag}.json')
    table(['seed', 'C at p=0.5 (canonical, validation)', f'C at p={p:g} (validation)', 'B (validation)'],
          [[s, fmt(RECORD['validation']['C'][s]['val']['macro_f1']), fmt(r['val']['macro_f1']), fmt(RECORD['validation']['B'][s]['val']['macro_f1'])] for s, r in EXERCISE['results']['C'].items()])
else:
    print('RUN_SYNTHETIC_FRACTION_EXERCISE is False; skipped.')
"""

S13_REVIEW = """
### 13.2 Optional human review of generated candidates

`outputs/.../human_review_template.csv` lists every candidate. Fill `decision` with `accept`, `reject` or `uncertain` and give a short `reason`, upload the file, set `HUMAN_REVIEW_CSV` to its path and `RUN_HUMAN_REVIEW_EXTENSION = True`, then run the next cell. Only accepted candidates enter a separately identified arm C (tag `human_review`), scored on validation only. The canonical frozen results are untouched. Ask yourself whether your decisions were independent of anything you saw in Sections 9–10.
"""

S13_REVIEW_RUN = """
if RUN_HUMAN_REVIEW_EXTENSION and 'C' not in RECORD['arms_run']:
    print('Skipped: the canonical arm C was not run (generation shortfall), so a human-reviewed C has no canonical counterpart.')
elif RUN_HUMAN_REVIEW_EXTENSION:
    if not HUMAN_REVIEW_CSV or not Path(HUMAN_REVIEW_CSV).is_file():
        print(f"Set HUMAN_REVIEW_CSV to a completed copy of {OUT / 'human_review_template.csv'}")
    else:
        run_stage('human_review', 'lab', 'stage_fit.py', '--review', HUMAN_REVIEW_CSV, '--tag', 'human_review')
        REVIEW = load('extensions/human_review.json')
        table(['seed', 'canonical C (validation)', 'human-reviewed C (validation)'], [[s, fmt(RECORD['validation']['C'][s]['val']['macro_f1']), fmt(r['val']['macro_f1'])] for s, r in REVIEW['results']['C'].items()])
else:
    print('RUN_HUMAN_REVIEW_EXTENSION is False; skipped.')
"""

S13_BYOD = """
### 13.3 Bring your own data

The BYOD path runs the **whole** experiment on your images, not just inference: validation, grouping and split, the budget, Phi-4 descriptions, prompts using your defect class names, FLUX generation, the three arms, validation selection, freezing, one test evaluation, export and reload. The contract is in Section 1. To check a zip first without running any model, set `BYOD_CHECK_ZIP` to its path and run the next cell; it applies the same extraction, manifest, image, grouping, split and budget rules and reports the first broken rule. To run the experiment, set `DATA_SOURCE = 'byod'` and `BYOD_ZIP_PATH` (or leave the path empty to get an upload dialog), then Run all. Outputs go to `outputs/byod_capstone/`. If your manifest has no `group` column, grouping falls back to duplicate detection and the same limited interpretation applies as for Bosch.
"""

S13_BYOD_RUN = """
if BYOD_CHECK_ZIP:
    try:
        run_stage('byod_check', 'lab', 'stage_data.py', '--check-byod', BYOD_CHECK_ZIP)
    except RuntimeError as exc:
        print(f'BYOD check did not pass: {exc}')
else:
    print('BYOD_CHECK_ZIP is empty; skipped.')
"""

S13_TROUBLE = """
### 13.4 Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| Runtime check: no GPU or < 15 GB | CPU or small-GPU runtime | Runtime → Change runtime type → T4 GPU, then Run all |
| Runtime check: not enough disk | a previous run's files, or a small disk | Runtime → Disconnect and delete runtime, then Run all; keep `DELETE_MODEL_WEIGHTS_AFTER_USE = True` |
| Environment "cannot see the GPU" | driver older than the CUDA 12 wheels need | record the driver shown by the runtime check and report it; do not switch to unpinned packages |
| `uv` or package download errors | transient network or index outage | re-run the environment cell; completed environments are reused via their lock digest |
| `download ... did not complete` | the dataset host timed out | re-run the data cell; completed bytes are resumed. A timeout is an access problem, not evidence of corrupt data |
| `CONTRACT FAILURE: ... digest` | a corrupted or substituted file | delete the named file under `work/sdi_capstone/` and re-run; never skip the check |
| exit code -9 or "Killed" during a model load | host memory exhausted | note `lowest host memory available` from the stage log; use a high-RAM runtime; the notebook does not silently change precision or sizes |
| CUDA out of memory | another process holds GPU memory | Runtime → Restart session, then Run all |
| `generation shortfall` | fewer than 24 usable candidates in a class | a reportable result: the synthetic comparison is not presented. Do not increase the sample count to get past it |
| `frozen input ... changed` | a file changed after freezing | re-run from Section 8; do not edit outputs by hand |
| `reload parity ... FAIL` | the artifact or backbone file differs | re-run Section 11; if it persists, the export is not trustworthy |
| BYOD `CONTRACT FAILURE` | the manifest or images break the contract | the message names the line, column, class, group or limit; fix the zip and use the BYOD check cell |
| HF warning about `HF_TOKEN` | unauthenticated downloads | harmless: every model used here is ungated, and the notebook removes tokens from child processes |

### 13.5 Further exploration

- **P1:** repeat the experiment with independently generated pools (new seed bases) and with products B and C; does C − B keep its sign?
- **P1:** run the human-reviewed arm and compare it with the canonical arm on validation.
- **P2:** train on product A and test on product B; does synthetic augmentation change the transfer gap?
- Apply the protocol to another inspection dataset with specimen identifiers, so groups mean physical parts.
"""

DISCLOSURE = """
## AI assistance disclosure

This notebook's code and text were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer is responsible for reviewing the implementation, validating results and making release decisions. AI assistance is not independent verification, provider endorsement or release approval.

## References

Abouelenin, A., Ashfaq, A., Atkinson, A., Awadalla, H., Bach, N., et al. (2025). *Phi-4-Mini technical report: Compact yet powerful multimodal language models via mixture-of-LoRAs*. arXiv. https://doi.org/10.48550/arXiv.2503.01743

Black Forest Labs. (2024). *FLUX.1 [schnell]* [Model card]. Hugging Face. https://huggingface.co/black-forest-labs/FLUX.1-schnell

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. In *2009 IEEE Conference on Computer Vision and Pattern Recognition* (pp. 248–255). https://doi.org/10.1109/CVPR.2009.5206848

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient finetuning of quantized LLMs*. arXiv. https://doi.org/10.48550/arXiv.2305.14314

Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC. https://doi.org/10.1201/9780429246593

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *2016 IEEE Conference on Computer Vision and Pattern Recognition* (pp. 770–778). https://doi.org/10.1109/CVPR.2016.90

Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management, 45*(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002

Trabucco, B., Doherty, K., Gurinas, M., & Salakhutdinov, R. (2023). *Effective data augmentation with diffusion models*. arXiv. https://doi.org/10.48550/arXiv.2302.07944

Wang, R., Hoppe, S., Monari, E., & Huber, M. F. (2023). *Defect Transfer GAN: Diverse defect synthesis for data augmentation*. arXiv. https://doi.org/10.48550/arXiv.2302.08366
"""

SCRIPT_NOTES = {
    "stage_data.py": "The next cell saves the data stage: acquisition, extraction, inventory, grouping, the split and the budget (with a `--check-byod` dry-run mode used in Section 13). The cell after it runs the stage.",
    "stage_features.py": "The next cell saves the feature stage. `--part real` handles the training budget and validation; `--part synthetic` (Section 7) handles generated candidates.",
    "stage_fit.py": "The next cell saves the fitting stage. It is used three ways: `--preview` (below), the canonical fit and freeze (Section 8), and validation-only extensions (Section 13). It never reads test images.",
    "stage_phi4.py": "The next cell saves the Phi-4 description stage; the cell after it runs it in the `phi4` environment.",
    "stage_prompts.py": "The next cell saves the prompt compilation stage.",
    "stage_flux.py": "The next cell saves the FLUX generation stage.",
    "stage_evaluate.py": "The next three cells save the stages that run **after** the freeze: the test stage (Section 9), the export stage and the fresh-process reload stage (Section 11). They are saved here, before the fit, so that the freeze binds their code together with the core, the data, feature and fitting stages: changing any of them afterwards makes the test, export and reload stages refuse to run. The test stage starts by verifying the frozen record and re-checking every test image against its recorded digests.",
    "stage_export.py": "",
    "stage_reload.py": "",
}


def script_cells(name: str) -> list[dict]:
    cells = []
    if SCRIPT_NOTES[name]:
        cells.append(md(SCRIPT_NOTES[name]))
    cells.append(code(writefile(name, (SRC / name).read_text(encoding="utf-8")), tag=f"{WORK_REL}/{name}"))
    return cells


def build() -> dict:
    cells: list[dict] = [md(OPENING), md(AUDIENCE), md(ROADMAP), md(QUESTION), md(GLOSSARY), md(S1_INTRO)]
    cells.append(code(CONFIG, form=True))
    cells.append(md("The runtime check records the GPU, driver, host memory and free disk, and stops early with an instruction if the run cannot fit."))
    cells.append(code(RUNTIME_CHECK, form=True))
    cells.append(md("Both environments are created from exact locks: every transitive package is pinned, and `uv` installs with `--no-deps`, so nothing floats. Reruns reuse an environment whose lock digest matches."))
    cells.append(code(locks_cell(), form=True))
    cells.append(code(INSTALL, form=True))
    cells.append(md(CORE_INTRO))
    for n, (title, text) in enumerate(core_sections()):
        cells.append(code(writefile("sdi_core.py", text, append=n > 0), tag=f"{WORK_REL}/sdi_core.py#{n}"))
        cells[-1]["metadata"]["dimer"]["section"] = title
    cells.append(md("The stage runner starts each script in its environment, streams its output, and measures time and memory."))
    cells.append(code(RUNNER, form=True))
    cells.append(md(S2_INTRO))
    cells += script_cells("stage_data.py")
    cells.append(code(S2_RUN))
    cells += [md(S2_NOTICE), md(S2_CHECK), md(S3_INTRO)]
    cells += script_cells("stage_features.py")
    cells += script_cells("stage_fit.py")
    cells.append(code(S3_RUN))
    cells += [md(S3_NOTICE), md(S3_CHECK), md(S4_INTRO)]
    cells += script_cells("stage_phi4.py")
    cells.append(code(S4_RUN))
    cells += [md(S4_NOTICE), md(S4_CHECK), md(S5_INTRO)]
    cells += script_cells("stage_prompts.py")
    cells.append(code(S5_RUN))
    cells += [md(S5_NOTICE), md(S5_CHECK), md(S6_INTRO)]
    cells += script_cells("stage_flux.py")
    cells.append(code(S6_RUN))
    cells += [md(S6_NOTICE), md(S6_CHECK), md(S7_INTRO), code(S7_RUN), md(S7_NOTICE), md(S8_INTRO)]
    cells += script_cells("stage_evaluate.py") + script_cells("stage_export.py") + script_cells("stage_reload.py")
    cells += [code(S8_RUN), md(S8_NOTICE), md(S8_CHECK), md(S9_INTRO)]
    cells.append(code(S9_RUN))
    cells += [md(S9_NOTICE), md(S9_CHECK), md(S10_INTRO), code(S10_RUN), md(S10_NOTICE), md(S11_INTRO)]
    cells.append(code(S11_RUN))
    cells += [md(S11_NOTICE), md(S11_CHECK), md(S12_INTRO), code(S12_RUN, form=True), md(S13_INTRO), code(S13_EXERCISE), md(S13_REVIEW), code(S13_REVIEW_RUN), md(S13_BYOD), code(S13_BYOD_RUN), md(S13_TROUBLE), md(DISCLOSURE)]
    for n, cell in enumerate(cells):
        cell["id"] = f"sdi-{n:03d}"
    sources = {f"tools/capstone/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SRC.glob("*.py"))}
    sources.update({f"tools/capstone/locks/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((SRC / "locks").glob("*.lock"))})
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "dimer": {
                "notebook_spec": "2.2",
                "profile": "E2E",
                "pedagogical_mode": "GUIDED",
                "standalone": True,
                "status": "candidate",
                "capability": "controlled comparison of synthetic (Phi-4 described, FLUX generated) versus conventional augmentation for a frozen-feature defect classifier on the Bosch SDI dataset",
                "generated_from": {"generator": GENERATOR, "repository": REPO, "sources": sources},
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render() -> str:
    return json.dumps(build(), indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render()
    if args.check:
        if not NOTEBOOK.exists() or NOTEBOOK.read_text(encoding="utf-8") != text:
            print(f"{NOTEBOOK.relative_to(ROOT)} is stale; run python tools/build_capstone_notebook.py", file=sys.stderr)
            return 1
        print(f"{NOTEBOOK.relative_to(ROOT)} is up to date")
        return 0
    NOTEBOOK.write_text(text, encoding="utf-8")
    print(f"wrote {NOTEBOOK.relative_to(ROOT)} ({len(build()['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
