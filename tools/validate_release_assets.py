"""Static release-asset validation for the FLUX.1 [schnell] text-to-image QLoRA DIMER pipeline.

Checks the STANDALONE tutorial notebook (DIMER Notebook Specification 2.2 §4), the tutorial
registry, model card, README, STATUS.md and weight documentation for source conformance and
cross-document identity consistency, and runs the generator parity checks (PAR1-PAR3).

This is source validation only. A PASS here is NOT clean-runtime execution evidence;
the release gate is defined in docs/release-verification.md.
"""
# ruff: noqa: E501  -- rule messages name the file and requirement in full; they are kept on one line
from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "flux_schnell_generation_pipeline"
REPO_NAME = "flux-schnell-generation-pipeline"
NOTEBOOK_NAME = "flux_schnell_generation_colab.ipynb"
CAPSTONE_NOTEBOOKS = ("DIMER_Bosch_Synthetic_Defect_Augmentation_Capstone.ipynb",)
CAPSTONE_SECTION = "## Scientific capstone notebooks"
EXPECTED_PROFILE = "E2E"
EXPECTED_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
PIPELINE_CLASS = "FluxSchnellPipeline"
MODEL_LOAD_EXPR = f"{PIPELINE_CLASS}.from_pretrained(weights_dir=WEIGHTS_DIR, use_lora=True)"
# The staging mirror's revision and the CLIP scorer's revision are cited beside the upstream model pin.
KNOWN_SHAS: frozenset[str] = frozenset({"9df3faa7ae3b6ddf0b2b69bb78616372897cc65c", "1a25a446712ba5ee05982a381eed697ef9b435cf"})
BYOD_GATES = ("USE_BYOD",)
EXPECTED_OUTPUTS = (
    "outputs/flux_schnell_generation_sample_captions.csv",
    "outputs/flux_schnell_generation_frozen_grid.jpg",
    "outputs/flux_schnell_generation_adapted_grid.jpg",
    "outputs/flux_schnell_generation_evaluation_report.json",
    "outputs/flux_schnell_generation_adapter",
    "outputs/flux_schnell_generation_result.json",
)
CODE_MARKERS = (
    # Stage 3 (fixed cell): the snapshot (staged from the mirror) and the scorer staged and verified; VAE/tokenizers loaded
    "use_lora=True)",
    "fetched_clip_vit_b_32_laion2b = stage_missing_scorer_files(SCORER_WEIGHTS_DIR, allow_download=True)",
    # Stage 4: pinned photographs, seeded splits, dataset manifest, sample CSV, refusal probes
    "USE_BYOD = False",
    "splits = fetch_sample_dataset(cache_dir='weights/inat-birds')",
    "byod_records = load_byod_dataset(byod_path)",
    "splits = split_dataset(byod_records, seed=0)",
    "dataset_report = dataset_manifest(",
    "captions_csv = 'outputs/flux_schnell_generation_sample_captions.csv'",
    "write_dataset_csv(test_records, captions_csv)",
    "validate_dataset(records)",
    # FS-m1: a path field, a cancelled upload named, duplicates reported, BYOD written under its own name
    "BYOD_PATH = ''",
    "if len(uploaded) != 1:",
    "duplicates = duplicate_images(byod_records)",
    "captions_csv = 'outputs/flux_schnell_generation_byod_captions.csv'",
    # Stage 5: prompts encoded once with both encoders, embeddings checked finite, encoders released
    "encode_report = pipe.encode_prompts(all_prompts)",
    "cache = pipe.export_prompt_cache()",
    "released = pipe.release_text_encoder()",
    # FS-M3: every GPU resident of an earlier pass is released before the encoders load
    "def release_gpu_residents():",
    "released_before_encoding = release_gpu_residents()",
    # FS-m3: Sections 6, 7 and 10 start from the pretrained base
    "def reset_to_pretrained():",
    # Stage 6: the 4-bit transformer loaded after encoding; frozen held-out flow-matching loss, CLIP-scored generations, ceiling
    "load_report = pipe.load_transformer()",
    "'parameters': count_parameters(pipe.transformer)",
    "frozen_test = pipe.evaluate(test_records, seed=EVAL_SEED)",
    "frozen_generation = pipe.generate(generation_prompts, seed=1000, steps=STEPS)",
    "frozen_scores = score_generations(scorer, frozen_generation['images'], references=test_records)",
    # FS-M2: the real photographs are a leave-one-out reference line, not a ceiling
    "real_reference = real_photo_reference(scorer, test_records)",
    "'real_photo_reference': real_reference",
    # FS-m1: a bounded number of generation prompts, with a warning
    "generation_prompts = [p for p in prompts[:MAX_GENERATION_PROMPTS] for _ in range(IMAGES_PER_PROMPT)]",
    # FS-m5: the images are displayed inline
    "display(Image.open(frozen_grid))",
    "display(Image.open(pair_grid))",
    # Stage 7: bounded QLoRA fine-tuning
    "adapt_result = pipe.adapt(",
    "lr=LEARNING_RATE",
    "if pipe.adapter is None:",
    "batch_size=BATCH_SIZE",
    # Stage 8: paired held-out comparison and the assertion
    "adapted_test = pipe.evaluate(test_records, seed=EVAL_SEED)",
    "adapted_scores = score_generations(scorer, adapted_generation['images'], references=test_records)",
    "assert best['val_loss'] <= adapt_result['history'][0]['val_loss']",
    "assert abs(adapted_val['flow_matching_mse'] - best['val_loss']) < 1e-4",
    # Stage 9: new prompt, artifact export, the adapted transformer released, reload parity, provenance
    "new_generation = pipe.generate([NEW_PROMPT, NEW_PROMPT], seed=2000, steps=STEPS)",
    "pipe.save_artifact(artifact_dir, metadata=",
    "'adapted_transformer_released': pipe.release_transformer()",
    "reloaded = FluxSchnellPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device, prompt_cache=cache)",
    "assert parity['flow_matching_mse_diff'] < 1e-5 and parity['mean_abs_pixel_diff'] < 1.0",
    # FS-M3: the reloaded pipeline is released after the parity check
    "reloaded.release_transformer()",
    "del reloaded",
    # FS-m2: the carried modules' per-file SHA-256 travel with the source revision
    "'module_sha256_per_file': NOTEBOOK_SOURCE['module_sha256_per_file']",
    # FS-M4: the change-one-thing activity, off by default
    "RUN_ACTIVITY = False",
    "'staging': MANIFEST['staging']",
    "'safetensors_only': True",
    "'remote_code_executed': False",
    "'data_base_url': CORPUS_BASE_URL",
)
MARKDOWN_MARKERS = (
    "**Capability:** few-step text-to-image generation with a 12 B-parameter rectified-flow transformer loaded 4-bit, held-out flow-matching-loss and CLIP-scored evaluation, and bounded QLoRA fine-tuning to a set of captioned photographs",
    "ungated mirror",
    "does not fit a 16 GB GPU in any 16-bit format",
    "do not fit beside it",
    "Generation has no ground truth",
    "flow-matching loss",
    "reference line, not a ceiling",
    "leave-one-out",
    "**Pretraining overlap.**",
    "**Run-to-run variability.**",
    "stratified within each caption",
    "not a human judgement",
    "Apache-2.0",
    "sample-sanity",
    "CC0",
    "Every number is the **4-bit** model's",
)
# Learner-facing text the review fixes removed; it must not come back (FS-M1 restart-dependent install, FS-M2 the
# real-photo "ceiling", FS-m1 the wrong BYOD minimum and split wording, FS-m4 stale expectations and a pre-stated result).
STALE_MARKDOWN = (
    "Restart the runtime, then rerun",
    "installs the pinned dependencies",
    "re-run from that cell",
    "real-photo ceiling",
    "the ceiling these numbers could reach",
    "the real-photo ceiling",
    "at least four images",
    "split by caption",
    "Hold out by caption",
    "below the real photographs' 1.0",
    "about an hour of model time",
    "Successful execution proves that the recorded repository revision",
)
# The guided layer (NOTEBOOK_SPEC 2.2 §3.5, GDL1-GDL14; review FS-M4): each marker with its minimum count.
GUIDED_MARKERS = (
    ("**Who this is for.**", 1),
    ("**Input → Model → Output.**", 1),
    ("**How to use this notebook.**", 1),
    ("**Roadmap:**", 1),
    ("**Predict before running:**", 6),
    ("**What to notice (Section", 7),
    ("<summary>Check your reasoning</summary>", 7),
    ("## 10. Your turn — change one thing", 1),
    ("**Predict → Change one thing → Run → Observe → Explain.**", 1),
    ("## Troubleshooting", 1),
    ("## Glossary", 1),
    ("## Conclusion (your notes)", 1),
    ("> **Infrastructure.**", 3),
    ("**Next experiments**", 1),
)
# Direct-library use that must stay inside the carried module cells (G2).
FORBIDDEN_OUTSIDE_MODULE = (
    "from huggingface_hub import",
    "import huggingface_hub",
    "hf_hub_download(",
    "urllib.request",
    "from safetensors",
    "load_file(",
    "save_file(",
    "torch.load(",
    "pickle.load",
    "Unpickler",
    "torch.no_grad(",
    "torch.inference_mode(",
    ".backward(",
    "pipe.transformer(",
    "from diffusers ",
    "import diffusers.",
    "from transformers ",
    "from peft ",
    "import bitsandbytes.",
    "BitsAndBytesConfig",
    "add_adapter(",
    "inject_adapter_in_model(",
    "T5EncoderModel",
    "CLIPTextModel",
    "CLIPModel",
    "FluxTransformer2DModel",
)
# ---------------------------------------------------------------------------
# Shared checks. Everything below is source/structure validation only. Passing
# these checks is NOT clean-runtime execution evidence under DIMER Notebook
# Specification 2.0; see docs/release-verification.md for the release gate.
# ---------------------------------------------------------------------------

NOTEBOOK_SPEC = "2.2"
ALLOWED_PROFILES = {"E2E", "ARTIFACT-INFERENCE", "TASK-INFERENCE", "MULTI-CAPABILITY", "SMOKE"}
STATUS_TOKENS = ("Candidate", "Release-grade")
PLACEHOLDER = re.compile(r"\b(TODO|TBD|FIXME)\b|Insert text here|Tooltip:", re.I)
SHA40 = re.compile(r"^[0-9a-f]{40}$")
IDENTITY_NAMES = ("MODEL_ID", "MODEL_REVISION", "MODEL_LICENSE", "MODEL_KEY")
UNSUPPORTED_CLAIMS = re.compile(
    r"\b(production[- ]ready|battle[- ]tested|state[- ]of[- ]the[- ]art results (were|are) reproduced"
    r"|benchmark superiority (is|was) (shown|established)|is release-grade|now release-grade)\b",
    re.I,
)
REQUIRED_CARD_HEADINGS = [
    (4, "Description"),
    (4, "Intended Use and Limitations"),
    (6, "Primary Intended Uses"),
    (6, "Primary Intended Users"),
    (6, "Out-of-scope use cases"),
    (4, "Factors"),
    (6, "Groups"),
    (6, "Instrumentation"),
    (6, "Environment"),
    (4, "Metrics"),
    (6, "Performance Measures"),
    (6, "Decision thresholds"),
    (6, "Approaches to uncertainty and variability"),
    (4, "Ethical considerations and biases"),
    (6, "Data"),
    (6, "Human Life"),
    (6, "Mitigations"),
    (6, "Risks and harms"),
    (6, "Use cases"),
]
# Markers every standalone DIMER tutorial in this fleet must carry, independent of profile.
# Matched on comment-stripped code, so a commented-out call does not count.
COMMON_CODE_MARKERS = (
    "PINS = [",
    "NOTEBOOK_SOURCE = {",
    "SKIP_INSTALL = os.environ.get('DIMER_NOTEBOOK_CI_PREINSTALLED') == '1'",
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *PINS], check=True)",
    "importlib.metadata.packages_distributions()",
    "importlib.invalidate_caches()",
    "platform.python_version()",
    "torch.__version__",
    "MANIFEST = {",
    "if (MANIFEST['modelId'], MANIFEST['revision']) != (MODEL_ID, MODEL_REVISION):",
    "WEIGHTS_DIR = DEFAULT_WEIGHTS_DIR",
    "json.dump(MANIFEST, handle, indent=2)",
    "fetched = stage_missing_files(WEIGHTS_DIR, allow_download=True)",
    "snapshot = verify_snapshot(WEIGHTS_DIR)",
    "'repository_revision': NOTEBOOK_SOURCE['repository_revision']",
    "'notebook_source': NOTEBOOK_SOURCE",
    "os.makedirs('outputs', exist_ok=True)",
    "from google.colab import files",
    "files.upload()",
)
COMMON_MARKDOWN_MARKERS = (
    f"**Notebook specification:** DIMER Notebook Specification {NOTEBOOK_SPEC} — **standalone** (§4)",
    "**Mode:** `",
    "**Run all:**",
    "**Bring Your Own Data:**",
    "**This notebook is standalone.**",
    "**Learning objectives:**",
    "## Prerequisites",
    "Do not upload confidential or restricted",
    "- **External access:** the Hugging Face Hub",
    "## 1. Install the pinned runtime",
    "## 2. Pipeline code (carried verbatim from",
    "## 3. Pin, stage and verify the model",
    "## Interpretation and limits",
    "A successful default run **proves** that",
    "without the repository being",
    "It does **not** establish benchmark superiority",
    "## References",
    f"- Repository model card: https://github.com/kurtvalcorza/{REPO_NAME}/blob/main/MODEL_CARD.md",
)
# Patterns that must never appear in tutorial code (comment-stripped), in any cell.
FORBIDDEN_PATTERNS = (
    ("credential in clone URL", re.compile(r"https://[^/'\"\s]*@github\.com/|x-access-token:")),
    ("repository clone (ST1)", re.compile(r"\bgit\b[^\n]*\bclone\b|github\.com")),
    ("editable self-install", re.compile(r"""['"](?:-e|--editable)['"]|pip install (?:-e|--editable)\b""")),
    ("repository package import (ST1)", re.compile(rf"^\s*(?:from|import)\s+{PACKAGE}\b", re.M)),
    ("mutable model reference (MOD14)", re.compile(r"revision\s*=\s*['\"](?:main|latest)['\"]")),
    ("trust_remote_code enabled", re.compile(r"trust_remote_code\s*[=:]\s*True")),
    (
        "unsafe deserialization",
        re.compile(
            r"\bpickle\.load"
            r"|\btorch\.load\s*\((?![^)]*weights_only\s*=\s*True)"
            r"|weights_only\s*=\s*False"
            r"|getattr\(\s*torch\s*,\s*['\"]load['\"]"
        ),
    ),
    ("archive extractall", re.compile(r"\.extractall\s*\(")),
    ("notebook magic or shell escape", re.compile(r"(?m)^\s*[%!]|get_ipython\(\)")),
)


class ValidationError(AssertionError):
    """Raised for any release-asset defect; the message names the file and rule."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _cell_source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def _strip_comments(source: str) -> str:
    """Return the source without comment tokens (string contents are preserved)."""
    out: list[str] = []
    last_row, last_col = 1, 0
    lines = source.splitlines(keepends=True)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return source
    for token in tokens:
        (srow, scol), (erow, ecol) = token.start, token.end
        if srow > last_row:
            out.append(lines[last_row - 1][last_col:] if last_row - 1 < len(lines) else "")
            for row in range(last_row, srow - 1):
                out.append(lines[row])
            last_row, last_col = srow, 0
        if srow - 1 < len(lines):
            out.append(lines[srow - 1][last_col:scol])
        if token.type != tokenize.COMMENT:
            out.append(token.string)
        last_row, last_col = erow, ecol
    return "".join(out)


def _assignment_targets(node: ast.AST):
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign | ast.AugAssign | ast.NamedExpr | ast.For | ast.comprehension):
        targets = [node.target]
    elif isinstance(node, ast.withitem) and node.optional_vars is not None:
        targets = [node.optional_vars]
    else:
        return []
    names = []
    for target in targets:
        for sub in ast.walk(target):
            if isinstance(sub, ast.Name):
                names.append(sub.id)
    return names


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    _check(spec is not None and spec.loader is not None, f"tools/{name}.py is required")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _package_identity() -> tuple[str, str]:
    """Read MODEL_ID / MODEL_REVISION from the package source without importing torch."""
    text = _read(ROOT / "src" / PACKAGE / "pipeline.py")
    model_id = re.search(r'^MODEL_ID = "([^"]+)"$', text, re.M)
    revision = re.search(r'^MODEL_REVISION = "([^"]+)"$', text, re.M)
    _check(
        model_id is not None and revision is not None,
        "pipeline.py must define MODEL_ID and MODEL_REVISION",
    )
    _check(SHA40.match(revision.group(1)) is not None, "MODEL_REVISION must be a 40-hex immutable commit")
    _check(model_id.group(1) == EXPECTED_MODEL_ID, f"MODEL_ID drifted from {EXPECTED_MODEL_ID}")
    return model_id.group(1), revision.group(1)


def validate_model_card() -> None:
    path = ROOT / "MODEL_CARD.md"
    text = _read(path)
    _check(text.startswith("---\n"), "MODEL_CARD.md must start with YAML front matter")
    front = text.split("---", 2)[1]
    for key in ("license:", "model_card_spec:", "base_model:"):
        _check(key in front, f"MODEL_CARD.md missing front-matter field: {key}")
    _check('model_card_spec: "1.1"' in front, "MODEL_CARD.md model_card_spec must be 1.1")
    _check(f"base_model: {EXPECTED_MODEL_ID}" in front, "MODEL_CARD.md base_model must equal MODEL_ID")
    _check(not PLACEHOLDER.search(text), "MODEL_CARD.md contains placeholder/scaffolding text")
    _check(not UNSUPPORTED_CLAIMS.search(text), "MODEL_CARD.md makes an unsupported release/benchmark claim")
    h1 = re.findall(r"(?m)^# (?!#)(.+)$", text)
    _check(len(h1) == 1, f"MODEL_CARD.md must contain exactly one H1, got {len(h1)}")
    found = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            found.append((len(match.group(1)), match.group(2).strip()))
    positions = []
    for heading in REQUIRED_CARD_HEADINGS:
        matches = [
            index
            for index, item in enumerate(found)
            if item[0] == heading[0] and item[1].casefold() == heading[1].casefold()
        ]
        _check(len(matches) == 1, f"required model-card heading missing/duplicated: {heading}")
        positions.append(matches[0])
    _check(positions == sorted(positions), "required model-card headings are out of order")
    _check("## Immutable provenance" in text, "MODEL_CARD.md must carry an '## Immutable provenance' section")


def validate_identity_consistency() -> None:
    """The immutable upstream identity must be the same string in every document."""
    model_id, revision = _package_identity()
    for name in ("README.md", "MODEL_CARD.md", "docs/WEIGHTS.md"):
        text = _read(ROOT / name)
        _check(model_id in text, f"{name} must name the upstream model `{model_id}`")
        _check(revision in text, f"{name} must cite the immutable revision {revision}")
        other = re.findall(r"\b[0-9a-f]{40}\b", text)
        stray = sorted({sha for sha in other if sha != revision and sha not in KNOWN_SHAS})
        _check(not stray, f"{name} cites an unexpected 40-hex revision: {stray}")


# --- weight-facts check (fleet rollout 2026-09-24) ---
# Every SHA-256 digest and byte count quoted in the weight prose must come from a committed
# weights/*/dimer-base-manifest.json, or be declared below with a label saying what it describes
# (dataset files, upstream files that are not staged, origin checkpoints, totals). Declared entries
# that no document cites any more are rejected, so the allowlist cannot go stale.
WEIGHT_DOCS = ("README.md", "MODEL_CARD.md", "docs/WEIGHTS.md")
EXTERNAL_WEIGHT_BYTES: dict[int, str] = {
    5_789_324: "60 pinned iNaturalist tutorial photographs, total",
    37_404_064: "FLUX LoRA adapter.safetensors from the tutorial run (380 tensors, float32)",
    9_524_648_584: "FLUX.1-schnell T5 text encoder (text_encoder_2), two shards together",
    23_782_494_032: "FLUX.1-schnell transformer, three bf16 shards together",
    23_782_506_688: "single-file flux1-schnell.safetensors on the unsloth mirror at 9df3faa (not staged)",
}
EXTERNAL_WEIGHT_DIGESTS: dict[str, str] = {}
_DIGEST = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{64}(?![0-9a-fA-F])")
_GROUPED = r"(\d{1,3}(?:[,\u202f\u00a0 ]\d{3})+|\d+)"
_BYTE_COUNT = re.compile(r"(?<![\d,\-])" + _GROUPED + r"\s*bytes\b|totalBytes`?\s*" + _GROUPED)


def _manifest_facts(root: Path = ROOT) -> tuple[set[str], set[int]]:
    digests: set[str] = set()
    sizes: set[int] = set()
    for path in sorted(root.glob("weights/*/dimer-base-manifest.json")):
        manifest = json.loads(_read(path))
        sizes.add(manifest["totalBytes"])
        for entry in manifest["files"]:
            digests.add(entry["sha256"])
            sizes.add(entry["bytes"])
    return digests, sizes


def validate_weight_facts(root: Path = ROOT) -> None:
    """Every SHA-256 and byte count quoted in the weight prose must come from a manifest or a labelled allowlist entry."""
    digests, sizes = _manifest_facts(root)
    _check(bool(digests), "no weights/*/dimer-base-manifest.json found to check weight facts against")
    cited_digests: set[str] = set()
    cited_sizes: set[int] = set()
    for name in WEIGHT_DOCS:
        path = root / name
        if not path.exists():
            continue
        text = _read(path)
        found_digests = set(_DIGEST.findall(text))
        found_sizes = {int(re.sub(r"[,\u202f\u00a0 ]", "", m.group(1) or m.group(2))) for m in _BYTE_COUNT.finditer(text)}
        cited_digests |= found_digests
        cited_sizes |= found_sizes
        bad_digests = sorted(found_digests - digests - set(EXTERNAL_WEIGHT_DIGESTS))
        _check(not bad_digests, f"{name} cites SHA-256 digests absent from every manifest and from EXTERNAL_WEIGHT_DIGESTS: {bad_digests}")
        bad_sizes = sorted(found_sizes - sizes - set(EXTERNAL_WEIGHT_BYTES))
        _check(not bad_sizes, f"{name} cites byte counts absent from every manifest and from EXTERNAL_WEIGHT_BYTES: {bad_sizes}")
    stale = sorted(set(EXTERNAL_WEIGHT_BYTES) - cited_sizes) + sorted(set(EXTERNAL_WEIGHT_DIGESTS) - cited_digests)
    _check(not stale, f"EXTERNAL_WEIGHT_* entries no weight document cites any more: {stale}")


# --- end weight-facts check ---

def validate_release_status() -> None:
    """STATUS.md, README.md and tutorials/README.md must agree on one status token."""
    status = _read(ROOT / "STATUS.md")
    match = re.search(r"Current status: \*\*(Candidate|Release-grade)\b", status)
    _check(match is not None, "STATUS.md must declare 'Current status: **Candidate**' or '**Release-grade**'")
    token = match.group(1)
    readme = _read(ROOT / "README.md")
    _check("## Release status" in readme, "README.md must have a '## Release status' section")
    section = readme.split("## Release status", 1)[1]
    _check(section.lstrip().startswith(f"**{token}"), f"README.md release status must open with **{token}**")
    # The capstone section carries its own status (candidate until its combined hosted run is recorded).
    registry = _read(ROOT / "tutorials" / "README.md").split(CAPSTONE_SECTION, 1)[0].replace("**", "")
    _check(f"| {token}" in registry, f"tutorials/README.md must record the {token} status")
    other = [t for t in STATUS_TOKENS if t != token]
    for name, text in (("README.md", section.replace("**", "")), ("tutorials/README.md", registry)):
        for stale in other:
            _check(f"| {stale}" not in text, f"{name} carries a conflicting status token")
    if token == "Candidate":
        _check(
            "docs/release-verification.md" in registry or "release-verification" in registry,
            "tutorials/README.md must point Candidate notebooks at docs/release-verification.md",
        )
    for name in ("README.md", "STATUS.md", "tutorials/README.md", "docs/release-verification.md"):
        text = _read(ROOT / name)
        _check(not PLACEHOLDER.search(text), f"{name} contains placeholder text")
        _check(not UNSUPPORTED_CLAIMS.search(text), f"{name} makes an unsupported release/benchmark claim")
    verification = _read(ROOT / "docs" / "release-verification.md")
    _check(
        "## Recorded executions" in verification,
        "docs/release-verification.md must have '## Recorded executions'",
    )


def _validate_notebook_structure(path: Path, notebook: dict) -> tuple[list[tuple[int, str, ast.Module]], str]:
    _check(notebook.get("nbformat") == 4, f"{path.name}: nbformat must be 4")
    dimer = notebook.get("metadata", {}).get("dimer")
    _check(isinstance(dimer, dict), f"{path.name}: metadata.dimer block is required")
    profile = dimer.get("notebook_profile")
    _check(profile in ALLOWED_PROFILES, f"{path.name}: invalid metadata.dimer.notebook_profile {profile!r}")
    _check(profile == EXPECTED_PROFILE, f"{path.name}: profile {profile!r} != declared {EXPECTED_PROFILE!r}")
    spec = dimer.get("notebook_spec", dimer.get("notebook_spec_version"))
    _check(spec == NOTEBOOK_SPEC, f"{path.name}: metadata.dimer must declare notebook spec version '{NOTEBOOK_SPEC}'")
    _check(dimer.get("notebook_mode") in ("REFERENCE", "GUIDED", "WORKSHOP"), f"{path.name}: metadata.dimer.notebook_mode must declare a §3.3 pedagogical mode")
    _check(dimer.get("standalone") is True, f"{path.name}: metadata.dimer.standalone must be true (ST6)")
    generated = dimer.get("generated_from")
    _template = _load_tool("notebook_template").TEMPLATE
    _check(isinstance(generated, dict), f"{path.name}: metadata.dimer.generated_from is required (ST5)")
    _check(generated.get("repository") == REPO_NAME, f"{path.name}: generated_from.repository must be {REPO_NAME}")
    _check(
        generated.get("module") == f"{_template.get('package_dir', f'src/{PACKAGE}')}/{_template.get('entry_module', 'pipeline.py')}",
        f"{path.name}: generated_from.module must name the template entry module",
    )
    _pkg_dir = ROOT / _template.get("package_dir", f"src/{PACKAGE}")
    _order = _load_tool("build_notebook")._module_order(_pkg_dir, list(_template.get("modules", ["pipeline.py"])))
    module_sha = hashlib.sha256("".join(_read(_pkg_dir / m) for m in _order).encode("utf-8")).hexdigest()
    _check(
        generated.get("module_sha256") == module_sha,
        f"{path.name}: generated_from.module_sha256 does not match src/ (PAR4: regenerate the notebook)",
    )
    _check(bool(generated.get("generator")), f"{path.name}: generated_from.generator is required")
    _check(
        not str(generated.get("revision", "")).endswith("+working-tree"),
        f"{path.name}: generated_from.revision is a working-tree label; commit the package modules, then regenerate (FS-m2)",
    )
    _check(
        generated.get("module_sha256_per_file") == {f"{_template.get('package_dir', f'src/{PACKAGE}')}/{m}": hashlib.sha256(_read(_pkg_dir / m).encode("utf-8")).hexdigest() for m in _order},
        f"{path.name}: generated_from.module_sha256_per_file does not match src/ (FS-m2: regenerate the notebook)",
    )
    cells = notebook.get("cells", [])
    _check(
        bool(cells) and cells[0].get("cell_type") == "markdown",
        f"{path.name}: first cell must be markdown",
    )
    code_cells: list[tuple[int, str, ast.Module]] = []
    markdown_parts: list[str] = []
    for index, cell in enumerate(cells):
        source = _cell_source(cell)
        if cell.get("cell_type") == "markdown":
            markdown_parts.append(source)
            continue
        _check(cell.get("cell_type") == "code", f"{path.name}: unexpected cell type at {index}")
        _check(cell.get("execution_count") is None, f"{path.name}: code cell {index} has execution_count")
        _check(not cell.get("outputs"), f"{path.name}: code cell {index} persists outputs")
        _check(
            index > 0 and cells[index - 1].get("cell_type") == "markdown",
            f"{path.name}: code cell {index} lacks a preceding explanatory markdown cell",
        )
        for line in source.splitlines():
            _check(not line.lstrip().startswith(("%", "!")), f"{path.name}: cell {index} uses a magic")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise ValidationError(f"{path.name}: code cell {index} does not compile: {exc}") from exc
        code_cells.append((index, source, tree))
    markdown = "\n".join(markdown_parts)
    raw_code = "\n".join(source for _, source, _ in code_cells)
    _check(not PLACEHOLDER.search(raw_code + markdown), f"{path.name}: placeholder text found")
    _check(not UNSUPPORTED_CLAIMS.search(markdown), f"{path.name}: unsupported release/benchmark claim")
    return code_cells, markdown


def _validate_gates(path: Path, code_cells: list[tuple[int, str, ast.Module]]) -> None:
    """Each BYOD gate is assigned exactly once, to the constant False, on a Colab form line."""
    for gate in BYOD_GATES:
        assignments = []
        for index, source, tree in code_cells:
            lines = source.splitlines()
            for node in ast.walk(tree):
                if gate in _assignment_targets(node):
                    line = lines[node.lineno - 1] if node.lineno - 1 < len(lines) else ""
                    assignments.append((index, node, line))
        _check(
            len(assignments) == 1,
            f"{path.name}: {gate} must be assigned exactly once, found {len(assignments)}",
        )
        index, node, line = assignments[0]
        is_false = (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.value, ast.Constant)
            and node.value.value is False
        )
        _check(is_false, f"{path.name}: {gate} must be assigned the constant False (cell {index})")
        _check("# @param" in line, f"{path.name}: {gate} must be a Colab form parameter (`# @param`)")
    for index, _source, tree in code_cells:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                _check(
                    not any(alias.name.startswith("google.colab") for alias in node.names),
                    f"{path.name}: google.colab must only be imported inside the BYOD gate (cell {index})",
                )


def _validate_embedded_modules(path: Path, notebook: dict, build) -> list[int]:
    """PAR1: one tagged cell per carried module, in dependency order, each equal to its module after
    the documented rewrites (generator /2 multi-module carrier; ST2 applied per module)."""
    tagged = [
        (index, cell)
        for index, cell in enumerate(notebook.get("cells", []))
        if cell.get("cell_type") == "code" and cell.get("metadata", {}).get("dimer", {}).get("embedded_module")
    ]
    template = _load_tool("notebook_template").TEMPLATE
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    context = build.load_context(ROOT, template, recorded)
    expected_rels = context["module_rels"]
    _check(
        [cell["metadata"]["dimer"]["embedded_module"] for _, cell in tagged] == expected_rels,
        f"{path.name}: the cells tagged metadata.dimer.embedded_module must be exactly {expected_rels}, in order (ST2)",
    )
    for (index, cell), module, rel in zip(
        tagged, context["modules"], context["module_rels"], strict=True
    ):
        _check(
            cell["metadata"]["dimer"].get("module_sha256") == context["per_module_sha256"][rel],
            f"{path.name}: cell {index} module_sha256 tag does not match {rel}",
        )
        # FS-M4: a carried cell is the module plus the generator's one Infrastructure title line, collapsed.
        _check(
            _cell_source(cell).startswith(build.CARRIED_TITLE_PREFIX) and cell.get("metadata", {}).get("cellView") == "form",
            f"{path.name}: carried module cell {index} must start with the generator's Infrastructure title and be collapsed (cellView: form)",
        )
        _check(
            build.strip_carried_title(_cell_source(cell)).rstrip("\n") + "\n" == context["embedded"][module],
            f"{path.name}: embedded module cell {index} differs from {rel} (PAR1); regenerate the notebook",
        )
    return [index for index, _ in tagged]


def _validate_identity(
    path: Path, code_cells: list[tuple[int, str, ast.Module]], embedded: list[int], revision: str
) -> None:
    """Identity constants are bound in the carried module only; nothing outside rebinds them."""
    for index, _source, tree in code_cells:
        if index in embedded:
            continue
        for node in ast.walk(tree):
            rebound = [name for name in _assignment_targets(node) if name in IDENTITY_NAMES]
            _check(not rebound, f"{path.name}: {rebound} must not be rebound outside the module cell (cell {index})")
    outside = "\n".join(source for index, source, _ in code_cells if index not in embedded)
    manifest_block = re.search(r"^MANIFEST = (\{.*?^\})$", outside, re.M | re.S)
    _check(manifest_block is not None, f"{path.name}: model cell must carry an inline MANIFEST literal (ST3)")
    outside_without_manifest = outside.replace(manifest_block.group(0), "")
    _check(
        revision not in outside_without_manifest,
        f"{path.name}: the model revision may appear only in the carried module and the inline manifest",
    )


def _validate_parity(path: Path, notebook: dict, code_cells: list[tuple[int, str, ast.Module]], build) -> None:
    """PAR2/PAR3: inline manifest and pins equal the repository's; the generator reproduces the file."""
    template = _load_tool("notebook_template").TEMPLATE
    code = "\n".join(source for _, source, _ in code_cells)
    manifest = json.loads(_read(ROOT / "weights" / template["weights_key"] / "dimer-base-manifest.json"))
    inline = re.search(r"^MANIFEST = (\{.*?^\})$", code, re.M | re.S)
    _check(inline is not None and json.loads(inline.group(1)) == manifest, f"{path.name}: inline MANIFEST != committed manifest (PAR2)")
    pins_block = re.search(r"^PINS = \[(.*?)^\]", code, re.M | re.S)
    _check(pins_block is not None, f"{path.name}: install cell must carry PINS = [...] (ENV2)")
    _check(re.findall(r"'([^']+)'", pins_block.group(1)) == build._pins(ROOT), f"{path.name}: inline PINS != pyproject runtime pins (PAR2)")
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    rendered = build.to_bytes(build.render(ROOT, template, recorded))
    current = path.read_bytes().replace(b"\r\n", b"\n")  # autocrlf checkouts are CRLF
    _check(current == rendered, f"{path.name}: differs from tools/build_notebook.py output (PAR3); regenerate")


def _validate_bootstrap_guard(path: Path, code_cells: list[tuple[int, str, ast.Module]]) -> None:
    """The stale-import guard must actually raise: `if stale:` whose body raises RuntimeError."""
    raises = False
    for _, _, tree in code_cells:
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "stale":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Raise) and isinstance(sub.exc, ast.Call):
                        func = sub.exc.func
                        if isinstance(func, ast.Name) and func.id == "RuntimeError":
                            raises = True
    _check(raises, f"{path.name}: install cell must raise RuntimeError when already-imported packages change")


def _validate_notebook_content(
    path: Path, code_cells: list[tuple[int, str, ast.Module]], markdown: str, embedded: list[int]
) -> None:
    model_id, _revision = _package_identity()
    stripped = {index: _strip_comments(source) for index, source, _ in code_cells}
    code = "\n".join(stripped.values())
    outside = "\n".join(text for index, text in stripped.items() if index not in embedded)
    missing = [marker for marker in COMMON_CODE_MARKERS + CODE_MARKERS if marker not in code]
    _check(not missing, f"{path.name}: missing required source markers: {missing}")
    present = [label for label, pattern in FORBIDDEN_PATTERNS if pattern.search(code)]
    _check(not present, f"{path.name}: forbidden/insecure source: {present}")
    # The two kernel cells (FS-M1) download the pinned uv wheel themselves; every learner cell is still checked.
    kernel_cells = {index for index, source, _tree in code_cells if "# dimer: kernel cell" in source}
    learner = "\n".join(text for index, text in stripped.items() if index not in embedded and index not in kernel_cells)
    leaked = [marker for marker in FORBIDDEN_OUTSIDE_MODULE if marker in learner]
    _check(not leaked, f"{path.name}: direct library use outside the carried module cell (G2): {leaked}")
    # FS-M1: exactly two kernel cells (the isolated install and the router); everything else runs in the uv environment.
    kernel_raw = [source for _index, source, _tree in code_cells if "# dimer: kernel cell" in source]
    _check(len(kernel_raw) == 2, f"{path.name}: exactly two kernel cells (isolated install and router) are expected (FS-M1)")
    install = next((k for k in kernel_raw if "LOCK_TEXT = r" in k), "")
    for needed in ('"--managed-python"', '"--require-hashes"', '"--only-binary"', '":all:"', "UV_SHA256", "LOCK_SHA256", 'platform.machine() != "x86_64"'):
        _check(needed in install, f"{path.name}: the isolated install cell must use {needed} (FS-M1)")
    _check("_ip.input_transformers_cleanup.append(_route_to_isolated_runtime)" in "\n".join(kernel_raw), f"{path.name}: later cells must be routed to the isolated environment (FS-M1)")
    _check("module.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=package)" in "\n".join(kernel_raw), f"{path.name}: the worker's google.colab stubs must carry a module spec")
    titled = sum(1 for _index, source, _tree in code_cells if source.startswith("# @title Infrastructure:"))
    _check(titled == 7, f"{path.name}: install, router, runtime, three carried-module and model cells must carry an Infrastructure title (FS-M4), found {titled}")
    _check(
        f"pipe = {MODEL_LOAD_EXPR}" in outside,
        f"{path.name}: must load through {MODEL_LOAD_EXPR} (INF1)",
    )
    _validate_gates(path, code_cells)
    _validate_bootstrap_guard(path, code_cells)
    for filename in EXPECTED_OUTPUTS:
        _check(filename in code, f"{path.name}: must export {filename}")
    missing_md = [marker for marker in COMMON_MARKDOWN_MARKERS + MARKDOWN_MARKERS if marker not in markdown]
    _check(not missing_md, f"{path.name}: missing learner-facing markers: {missing_md}")
    stale = [marker for marker in STALE_MARKDOWN if marker in markdown]
    _check(not stale, f"{path.name}: stale learner-facing text: {stale}")
    _check("{{" not in markdown and "}}" not in markdown, f"{path.name}: markdown must not show doubled braces")
    short = [(marker, markdown.count(marker), least) for marker, least in GUIDED_MARKERS if markdown.count(marker) < max(least, 1)]
    _check(not short, f"{path.name}: guided layer incomplete (marker, found, needed): {short}")
    # FS-M2: "ceiling" is used only to say the real photographs are NOT one (or for the package's operational limits).
    loose = [m.start() for m in re.finditer(r"ceiling", markdown) if not markdown[: m.start()].endswith("not a ") and not markdown[: m.start()].endswith("operational ")]
    _check(not loose, f"{path.name}: markdown calls something a ceiling ({len(loose)} place(s)); the real photographs are a reference line, not a ceiling (FS-M2)")
    _check(f"**Profile:** `{EXPECTED_PROFILE}`" in markdown, f"{path.name}: markdown must state the profile")
    _check(f"https://huggingface.co/{model_id}" in markdown, f"{path.name}: references must link {model_id}")


def validate_notebooks() -> None:
    tutorials = ROOT / "tutorials"
    # Scientific capstone notebooks have their own generator, tests and verification record (tests/test_capstone.py,
    # docs/capstone-verification.md); this validator governs the pipeline's primary tutorial only.
    notebooks = sorted(p for p in tutorials.glob("*.ipynb") if p.name not in CAPSTONE_NOTEBOOKS)
    _check(len(notebooks) == 1, f"exactly one primary tutorial notebook is expected, found {len(notebooks)}")
    path = notebooks[0]
    _check(path.name == NOTEBOOK_NAME, f"tutorial notebook must be named {NOTEBOOK_NAME}, found {path.name}")
    build = _load_tool("build_notebook")
    notebook = json.loads(_read(path))
    code_cells, markdown = _validate_notebook_structure(path, notebook)
    embedded = _validate_embedded_modules(path, notebook, build)
    _model_id, revision = _package_identity()
    _validate_identity(path, code_cells, embedded, revision)
    _validate_parity(path, notebook, code_cells, build)
    _validate_notebook_content(path, code_cells, markdown, embedded)
    registry = _read(tutorials / "README.md")
    _check(f"`{path.name}`" in registry, f"{path.name} missing from tutorials/README.md")
    _check(f"`{EXPECTED_PROFILE}`" in registry, f"tutorials/README.md must record `{EXPECTED_PROFILE}`")
    _check(
        f"DIMER Notebook Specification {NOTEBOOK_SPEC}" in registry,
        "tutorials/README.md must name the notebook spec version",
    )
    _check("standalone" in registry.lower(), "tutorials/README.md must record that the notebook is standalone")


def validate_all() -> list[str]:
    validate_model_card()
    validate_identity_consistency()
    validate_weight_facts()
    validate_release_status()
    validate_notebooks()
    return ["model-card", "identity-consistency", "weight-facts", "release-status", "notebook+parity"]


def main() -> int:
    passed = validate_all()
    print(f"release asset validation: PASS ({', '.join(passed)})")
    print("NOTE: static source validation only; not clean-runtime execution evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
