"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

This template configures an E2E text-to-image fine-tuning workflow: the pinned FLUX.1 [schnell] snapshot
(diffusers layout, staged from an ungated mirror and digest-verified against the committed manifest) and the CLIP
scorer are staged and verified, 60 pinned CC0 iNaturalist bird photographs are fetched, validated and split, every
prompt is encoded once with CLIP-L and T5-XXL and the encoders are released, the 12 B transformer is loaded 4-bit,
the frozen model is scored (held-out flow-matching loss, CLIP-scored four-step generations) beside a leave-one-out
real-photo reference, a bounded QLoRA fine-tuning runs, the held-out scores are read again in a paired comparison,
a new prompt is rendered, and the adapter is exported and reloaded into a fresh 4-bit pipeline.

Review fixes (Notebook Review Framework v1, review PR #9, FS-M1..M4 / FS-m1..m5): the runtime is the fleet's uv
isolated environment (no in-kernel install, no restart); the real photographs are a LEAVE-ONE-OUT reference line
(`real_photo_reference`, each photo excluded from its own reference) and nothing is called a ceiling; Section 5
releases every GPU resident a previous pass left (the reloaded pipeline, the scorer) before the encoders load, and
Section 9 releases the reloaded pipeline, so a BYOD re-run fits a 15 GB T4; the guided layer (audience, how to use,
roadmap, contract, predictions, worked answers, activity, troubleshooting, glossary, conclusion) is added and the
setup cells are labelled Infrastructure and collapsed; BYOD states its real minimum, takes a path field, names every
refused file and reports dropped duplicates; Sections 6 and 7 start from the pretrained base (`reset_to_pretrained`);
the images are shown inline.

Code cells are written with single braces and escaped by ``_py`` for the generator's ``str.format`` pass; ``@STEM@``
becomes the output stem. Stage and closing markdown is formatted too, so it contains no literal braces.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "flux-schnell-generation-pipeline"


def _py(code: str) -> str:
    """Escape a code cell for the generator's ``str.format`` pass; ``@STEM@`` stands for ``{stem}``."""
    return code.replace("{", "{{").replace("}", "}}").replace("@STEM@", "{stem}")


BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/flux_schnell_generation_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-black--forest--labs%2FFLUX.1--schnell-ffcc4d?style=flat",
        "https://huggingface.co/black-forest-labs/FLUX.1-schnell",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-black--forest--labs%2Fflux-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/black-forest-labs/flux",
    ),
    ("Licence", "https://img.shields.io/badge/weights-Apache--2.0-blue.svg", "https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/LICENSE.md"),
]

# Worked answers quote the recorded Kaggle Tesla T4 run of the previous notebook version (blob 52226812, 2026-09-20):
# its model stages are this version's, except that it printed the real-photo reference similarity with each photo
# inside its own reference (88.66); the leave-one-out value 57.7 is derived exactly from that run's per-image numbers.
RECORD = "the recorded Kaggle T4 run of the previous notebook version (same model stages, 2026-09-20)"

_DATA_CODE = _py(
    """import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

USE_BYOD = False  # @param {type:"boolean"}
BYOD_PATH = ''  # @param {type:"string"}

os.makedirs('outputs', exist_ok=True)
if USE_BYOD:
    if BYOD_PATH:
        byod_path = Path(BYOD_PATH)
        if not byod_path.exists():
            raise ValueError(f'BYOD_PATH {BYOD_PATH!r} does not exist: give the path of a .zip, or of a folder, holding captions.csv and the images')
        file_name = byod_path.name
    else:
        try:
            from google.colab import files
        except ImportError:
            raise ValueError('The upload dialog exists only on Google Colab: set BYOD_PATH to a .zip or a folder on this machine instead.') from None
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise ValueError(f'Upload exactly one .zip holding captions.csv and the images (got {len(uploaded)} files; a cancelled upload gives 0). Outside Colab, set BYOD_PATH instead.')
        file_name, payload = next(iter(uploaded.items()))
        byod_path = Path('work') / file_name
        byod_path.parent.mkdir(parents=True, exist_ok=True)
        byod_path.write_bytes(payload)
    byod_records = load_byod_dataset(byod_path)
    duplicates = duplicate_images(byod_records)
    print({'byod_records': len(byod_records), 'duplicates_dropped': len(duplicates), 'duplicate_pairs': duplicates[:10]})
    splits = split_dataset(byod_records, seed=0)
    data_source = 'BYOD (' + file_name + ')'
    captions_csv = 'outputs/@STEM@_byod_captions.csv'
else:
    splits = fetch_sample_dataset(cache_dir='weights/inat-birds')
    data_source = SAMPLE_LABEL_SOURCE
    captions_csv = 'outputs/@STEM@_sample_captions.csv'
train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']

dataset_report = dataset_manifest({'train': train_records, 'validation': val_records, 'test': test_records})
print({'data_source': data_source, 'splits': {k: v['n_records'] for k, v in dataset_report['splits'].items()}, 'captions': dataset_report['splits']['train']['n_captions'], 'disjoint': dataset_report['disjoint']})
print({'shorter_side': dataset_report['splits']['train']['shorter_side'], 'centre_cropped': dataset_report['splits']['train']['centre_cropped'], 'digest': dataset_report['digest'][:16] + '...'})
print({'first_test_record': validate_inputs(test_records[0]), 'caption': test_records[0]['caption']})
prompts = sample_prompts(train_records)
print({'prompts': prompts})
written_csv = write_dataset_csv(test_records, captions_csv)
print({'captions_csv': str(written_csv)})

print({'validation': INPUT_SCHEMA['validation']})
probes = {
    'missing caption': [{'id': r['id'], 'image': r['image']} for r in train_records[:4]],
    'image too small': [{**train_records[0], 'image': Image.new('RGB', (200, 200))}, *train_records[1:4]],
    'duplicate id': [train_records[0], *train_records[:4]],
}
for name, records in probes.items():
    try:
        validate_dataset(records)
        print({'probe': name, 'verdict': 'accepted'})
    except (TypeError, ValueError) as exc:
        print({'probe': name, 'rejected': str(exc)[:110]})"""
)

_ENCODE_CODE = _py(
    """import gc
import time

NEW_PROMPT = 'a photo of a House Finch (Haemorhous mexicanus) perched on a snow-covered branch in winter'


def gpu_memory_gb():
    return round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None


def release_gpu_residents():
    \"\"\"Free everything an earlier pass left on the GPU before the 9.7 GB of text encoders load: the transformer of
    `pipe`, every other pipeline this notebook created (the Section 9 `reloaded` one) and the CLIP scorer.\"\"\"
    released = {}
    if pipe.transformer is not None:
        released['pipe.transformer'] = pipe.release_transformer()
    for name in [n for n, v in globals().items() if isinstance(v, FluxSchnellPipeline) and v is not pipe]:
        globals()[name].release_transformer()
        del globals()[name]
        released[name] = True
    if 'scorer' in globals():
        del globals()['scorer']
        released['scorer'] = True
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return released


def reset_to_pretrained():
    \"\"\"Sections 6, 7 and 10 start from the pretrained base: if an earlier pass trained the adapter, release that
    transformer so the next load is the pinned base with an untrained LoRA (its B matrices are zero).\"\"\"
    if pipe.adapter is None:
        return
    pipe.release_transformer()
    print({'released': 'the adapted transformer', 'reason': 'an earlier Section 7 had trained the adapter in memory; the next load is the pretrained base'})


released_before_encoding = release_gpu_residents()
if released_before_encoding:
    print({'released_before_encoding': released_before_encoding, 'gpu_memory_gb': gpu_memory_gb()})
all_prompts = sample_prompts(train_records + val_records + test_records) + [NEW_PROMPT]
encode_report = pipe.encode_prompts(all_prompts)
print({**encode_report, 'gpu_memory_gb_with_encoders': gpu_memory_gb()})
cache = pipe.export_prompt_cache()
print({'embeddings_finite': all(bool(torch.isfinite(v['embeds']).all()) and bool(torch.isfinite(v['pooled']).all()) for v in cache.values()), 'embeds_shape': tuple(next(iter(cache.values()))['embeds'].shape), 'pooled_shape': tuple(next(iter(cache.values()))['pooled'].shape)})
released = pipe.release_text_encoder()
print({'text_encoders_released': released, 'gpu_memory_gb_after_release': gpu_memory_gb(), 'device': pipe.device, 'compute_dtype': pipe.compute_dtype})"""
)

_FROZEN_CODE = _py(
    """STEPS = 4  # @param {type:"integer"}
IMAGES_PER_PROMPT = 1  # @param {type:"integer"}
MAX_GENERATION_PROMPTS = 12  # @param {type:"integer"}
EVAL_SEED = 0


def grid(images, path, columns=6):
    tiles = [im.resize((256, 256)) for im in images]
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new('RGB', (256 * columns, 256 * rows), 'white')
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (256 * (i % columns), 256 * (i // columns)))
    sheet.save(path)
    return path


reset_to_pretrained()
load_report = pipe.load_transformer()
print({**load_report, 'quantization': QUANTIZATION, 'parameters': count_parameters(pipe.transformer), 'lora_tensors': len(lora_parameter_names(pipe.transformer)), 'gpu_memory_gb': gpu_memory_gb()})
scorer = ClipScorer(weights_dir=SCORER_WEIGHTS_DIR, device=pipe.device)
t0 = time.perf_counter()
frozen_val = pipe.evaluate(val_records, seed=EVAL_SEED)
frozen_test = pipe.evaluate(test_records, seed=EVAL_SEED)
print({'frozen_flow_matching_mse': {'validation': frozen_val['flow_matching_mse'], 'test': frozen_test['flow_matching_mse']}, 'by_sigma_test': frozen_test['by_sigma'], 'seconds': round(time.perf_counter() - t0, 1)})

if len(prompts) > MAX_GENERATION_PROMPTS:
    print({'warning': f'{len(prompts)} distinct training captions: generating for the first {MAX_GENERATION_PROMPTS} only (each image takes several seconds per model on a T4). Raise MAX_GENERATION_PROMPTS to score more.'})
generation_prompts = [p for p in prompts[:MAX_GENERATION_PROMPTS] for _ in range(IMAGES_PER_PROMPT)]
frozen_generation = pipe.generate(generation_prompts, seed=1000, steps=STEPS)
print({'generated': len(frozen_generation['images']), 'steps': frozen_generation['steps'], 'guidance_scale': frozen_generation['guidance_scale'], 'precision': frozen_generation['precision'], 'seconds': frozen_generation['seconds'], 'adapted': frozen_generation['model']['adapted']})
frozen_scores = score_generations(scorer, frozen_generation['images'], references=test_records)
# The real held-out photographs scored the same way, LEAVE-ONE-OUT: each photo's reference is the other test photos of
# its caption, never itself. A reference line to read the generations against, not a ceiling.
real_reference = real_photo_reference(scorer, test_records)
print({'frozen_generations': {k: frozen_scores[k] for k in ('clip_prompt_similarity', 'label_accuracy', 'reference_similarity')}})
print({'real_photo_reference_leave_one_out': {k: real_reference.get(k) for k in ('clip_prompt_similarity', 'label_accuracy', 'reference_similarity')}, 'photos_without_a_reference': real_reference['n_without_reference']})
for entry in frozen_scores['per_image'][::IMAGES_PER_PROMPT]:
    print({'prompt': entry['prompt'][:42], 'clip': entry['clip_prompt_similarity'], 'nearest': entry['nearest_prompt'][13:40], 'correct': entry['correct'], 'reference_similarity': entry.get('reference_similarity')})
frozen_grid = grid([g['image'] for g in frozen_generation['images']], 'outputs/@STEM@_frozen_grid.jpg')
print({'grid': str(frozen_grid), 'columns': 'generation prompts in the order printed above', 'gpu_memory_gb': gpu_memory_gb()})
display(Image.open(frozen_grid))"""
)

_ADAPT_CODE = _py(
    """EPOCHS = 3  # @param {type:"integer"}
LEARNING_RATE = 1e-4  # @param {type:"number"}
BATCH_SIZE = 1  # @param {type:"integer"}


def report(entry):
    row = {'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4), 'val_flow_matching_mse': entry['val_loss']}
    if 'note' in entry:
        row['note'] = entry['note']
    print(row)


# A re-run (an optional experiment) starts again from the pretrained base, so epoch 0 is the frozen model.
reset_to_pretrained()
t0 = time.perf_counter()
adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, seed=EVAL_SEED, progress=report)
adapt_seconds = round(time.perf_counter() - t0, 1)
print({'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'steps': adapt_result['n_steps'], 'best_epoch': adapt_result['best_epoch'], 'precision': adapt_result['precision'], 'seconds': adapt_seconds, 'peak_gpu_memory_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2)})"""
)

_COMPARE_CODE = _py(
    """if pipe.adapter is None:
    raise RuntimeError('Section 8 compares the adapted model with the frozen one, and no adapted model is loaded: run Section 7 first (Section 9 releases the adapted transformer after exporting it).')
adapted_val = pipe.evaluate(val_records, seed=EVAL_SEED)
adapted_test = pipe.evaluate(test_records, seed=EVAL_SEED)
adapted_generation = pipe.generate(generation_prompts, seed=1000, steps=STEPS)
adapted_scores = score_generations(scorer, adapted_generation['images'], references=test_records)
comparison = {
    'flow_matching_mse_validation': {'frozen': frozen_val['flow_matching_mse'], 'adapted': adapted_val['flow_matching_mse']},
    'flow_matching_mse_test': {'frozen': frozen_test['flow_matching_mse'], 'adapted': adapted_test['flow_matching_mse']},
    'flow_matching_mse_test_by_sigma': {s: {'frozen': frozen_test['by_sigma'][s], 'adapted': adapted_test['by_sigma'][s]} for s in adapted_test['by_sigma']},
    'clip_prompt_similarity': {'frozen': frozen_scores['clip_prompt_similarity'], 'adapted': adapted_scores['clip_prompt_similarity'], 'real_photo_reference': real_reference['clip_prompt_similarity']},
    'label_accuracy': {'frozen': frozen_scores['label_accuracy'], 'adapted': adapted_scores['label_accuracy'], 'real_photo_reference': real_reference['label_accuracy']},
    'reference_similarity': {'frozen': frozen_scores.get('reference_similarity'), 'adapted': adapted_scores.get('reference_similarity'), 'real_photo_reference': real_reference.get('reference_similarity')},
}
for name, row in comparison.items():
    print({name: row})
print({'real_photo_reference': real_reference['reference_kind'], 'reading': real_reference['reading']})
for before, after in zip(frozen_scores['per_image'][::IMAGES_PER_PROMPT], adapted_scores['per_image'][::IMAGES_PER_PROMPT]):
    print({'prompt': before['prompt'][:42], 'reference_similarity': {'frozen': before.get('reference_similarity'), 'adapted': after.get('reference_similarity')}, 'correct': {'frozen': before['correct'], 'adapted': after['correct']}})
adapted_grid = grid([g['image'] for g in adapted_generation['images']], 'outputs/@STEM@_adapted_grid.jpg')
pair_grid = grid([g['image'] for g in frozen_generation['images']] + [g['image'] for g in adapted_generation['images']], 'outputs/@STEM@_frozen_vs_adapted.jpg', columns=len(generation_prompts))
print({'grid': str(adapted_grid), 'side_by_side': str(pair_grid), 'rows': 'top: frozen, bottom: adapted; same prompt and seed in each column'})
display(Image.open(pair_grid))
evaluation_report = {
    'model': {'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY, 'staging': {'repo': STAGING_ID, 'revision': STAGING_REVISION}, 'quantization': QUANTIZATION, 'compute_dtype': pipe.compute_dtype},
    'scorer': frozen_scores['scorer'],
    'data_source': data_source,
    'dataset': dataset_report,
    'generation': {'steps': STEPS, 'guidance_scale': GUIDANCE_SCALE, 'images_per_prompt': IMAGES_PER_PROMPT, 'prompts': len(generation_prompts) // IMAGES_PER_PROMPT, 'seed': 1000},
    'frozen': {'validation': frozen_val, 'test': frozen_test, 'generations': frozen_scores},
    'adapted': {'validation': adapted_val, 'test': adapted_test, 'generations': adapted_scores},
    'real_photo_reference': real_reference,
    'comparison': comparison,
    'adaptation': {k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')},
    'history': adapt_result['history'],
    'adaptation_seconds': adapt_seconds,
}
with open('outputs/@STEM@_evaluation_report.json', 'w', encoding='utf-8') as f:
    json.dump(evaluation_report, f, indent=2)
best = adapt_result['history'][adapt_result['best_epoch']]
assert best['val_loss'] <= adapt_result['history'][0]['val_loss']
assert abs(adapted_val['flow_matching_mse'] - best['val_loss']) < 1e-4
print({'test_flow_matching_mse_change': round(adapted_test['flow_matching_mse'] - frozen_test['flow_matching_mse'], 6), 'note': 'held-out observation, not asserted'})
print({'report': 'outputs/@STEM@_evaluation_report.json'})"""
)

_EXPORT_CODE = _py(
    """import platform
import shutil

if pipe.adapter is None:
    raise RuntimeError('Section 9 exports the adapted model, and no adapted model is loaded: run Section 7 (and Section 8) first.')
new_generation = pipe.generate([NEW_PROMPT, NEW_PROMPT], seed=2000, steps=STEPS)
new_scores = score_generations(scorer, new_generation['images'])
print({'new_prompt': NEW_PROMPT, 'clip_prompt_similarity': new_scores['clip_prompt_similarity'], 'seconds': new_generation['seconds'], 'note': 'sanity check, not an evaluation'})
for i, entry in enumerate(new_generation['images']):
    entry['image'].save(f'outputs/@STEM@_new_prompt_{i}.png')
display(Image.open(grid([g['image'] for g in new_generation['images']], 'outputs/@STEM@_new_prompt_grid.jpg', columns=2)))
before = pipe.generate([prompts[0]], seed=3000, steps=STEPS)['images'][0]['image']

artifact_dir = Path('outputs/@STEM@_adapter')
shutil.rmtree(artifact_dir, ignore_errors=True)
pipe.save_artifact(artifact_dir, metadata={'tutorial': '@STEM@', 'data_source': data_source})
artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))
print({'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...', 'quantization': artifact_manifest['base_model']['quantization']})

print({'adapted_transformer_released': pipe.release_transformer(), 'gpu_memory_gb': gpu_memory_gb()})
reloaded = FluxSchnellPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device, prompt_cache=cache)
reloaded_test = reloaded.evaluate(test_records, seed=EVAL_SEED)
after = reloaded.generate([prompts[0]], seed=3000, steps=STEPS)['images'][0]['image']
parity = {'flow_matching_mse_diff': round(abs(reloaded_test['flow_matching_mse'] - adapted_test['flow_matching_mse']), 8), 'mean_abs_pixel_diff': round(float(np.abs(np.asarray(before, dtype=np.float32) - np.asarray(after, dtype=np.float32)).mean()), 4)}
reloaded_best_epoch = reloaded.adapter['best_epoch']
print({'reload_parity': parity, 'reloaded_best_epoch': reloaded_best_epoch, 'gpu_memory_gb_with_reloaded': gpu_memory_gb()})
assert parity['flow_matching_mse_diff'] < 1e-5 and parity['mean_abs_pixel_diff'] < 1.0
# The parity check is done: release the reloaded pipeline so nothing stays resident for a later pass (a BYOD re-run
# loads the 9.7 GB text encoders in Section 5, which do not fit beside a second 4-bit transformer on a 15 GB GPU).
reloaded.release_transformer()
del reloaded
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print({'reloaded_pipeline_released': True, 'gpu_memory_gb': gpu_memory_gb()})

result_payload = {
    'notebook_source': NOTEBOOK_SOURCE,
    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],
    'module_sha256_per_file': NOTEBOOK_SOURCE['module_sha256_per_file'],
    'model': {**evaluation_report['model'], 'model_license': MODEL_LICENSE, 'device': pipe.device, 'precision': frozen_generation['precision'], 'source': pipe.source},
    'scorer': {**evaluation_report['scorer'], 'license': SCORER_LICENSE},
    'provenance': {
        'snapshots': {'model': len(MANIFEST['files']), 'scorer': len(SCORER_MANIFEST['files'])},
        'staging': MANIFEST['staging'],
        'safetensors_only': True,
        'remote_code_executed': False,
        'text_encoders_released_before_training': released,
        'transformer_loaded_after_encoding': load_report,
        'data_base_url': CORPUS_BASE_URL,
        'data_license': CORPUS_LICENSE,
    },
    'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'diffusers': diffusers.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__, 'bitsandbytes': bitsandbytes.__version__, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    'data_source': data_source,
    'comparison': comparison,
    'real_photo_reference': {'kind': real_reference['reference_kind'], 'reading': real_reference['reading']},
    'artifact': {'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes']},
    'reload_parity': parity,
    'reloaded_best_epoch': reloaded_best_epoch,
}
with open('outputs/@STEM@_result.json', 'w', encoding='utf-8') as f:
    json.dump(result_payload, f, indent=2)

print('outputs/:')
for path in sorted(Path('outputs').rglob('*')):
    if path.is_file():
        print(f'  - {path.as_posix()} ({path.stat().st_size / 1024:.1f} KB)')"""
)

_ACTIVITY_CODE = _py(
    """RUN_ACTIVITY = False  # @param {type:"boolean"}
ACTIVITY_STEPS = (1, 2, 4)

if not RUN_ACTIVITY:
    print('Activity not run: set RUN_ACTIVITY = True and run this cell. It writes only under outputs/activity/ and changes no earlier result.')
else:
    reset_to_pretrained()
    activity_load = pipe.load_transformer()
    activity_dir = Path('outputs/activity')
    activity_dir.mkdir(parents=True, exist_ok=True)
    activity_prompts = prompts[:MAX_GENERATION_PROMPTS]
    activity_rows = {}
    activity_images = []
    for steps in ACTIVITY_STEPS:
        generation = pipe.generate(activity_prompts, seed=1000, steps=steps)
        scores = score_generations(scorer, generation['images'], references=test_records)
        activity_rows[steps] = {'clip_prompt_similarity': scores['clip_prompt_similarity'], 'label_accuracy': scores['label_accuracy'], 'reference_similarity': scores.get('reference_similarity'), 'seconds': generation['seconds']}
        activity_images += [g['image'] for g in generation['images']]
        print({'steps': steps, **activity_rows[steps]})
    activity_grid = grid(activity_images, activity_dir / '@STEM@_steps.jpg', columns=len(activity_prompts))
    print({'grid': str(activity_grid), 'rows': [f'{s} step(s)' for s in ACTIVITY_STEPS], 'model': 'the pretrained base (untrained LoRA)', 'real_photo_reference': {k: real_reference.get(k) for k in ('clip_prompt_similarity', 'label_accuracy', 'reference_similarity')}})
    display(Image.open(activity_grid))
    with open(activity_dir / '@STEM@_steps.json', 'w', encoding='utf-8') as f:
        json.dump({'model': 'pretrained base, untrained LoRA', 'seed': 1000, 'prompts': activity_prompts, 'rows': activity_rows, 'load': activity_load}, f, indent=2)
    print({'activity_transformer_released': pipe.release_transformer(), 'gpu_memory_gb': gpu_memory_gb()})"""
)

TEMPLATE = {
    "package": "flux_schnell_generation_pipeline",
    "repo_name": REPO,
    "stem": "flux_schnell_generation",
    "notebook_name": "flux_schnell_generation_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "infrastructure_labels": True,
    "collapse_model_cell": True,
    "isolated_runtime": True,
    # The fleet's uv isolated-environment mechanism (ast-audio-classification-pipeline / bioclip2-biodiversity-pipeline;
    # the same uv wheel as esm2-protein-pipeline): managed CPython, a size- and SHA-256-verified uv wheel, and a lock
    # compiled from the pyproject pins with `uv pip compile pyproject.toml --python-version 3.12 --python-platform
    # x86_64-manylinux_2_28 --generate-hashes --only-binary :all: -o tutorials/requirements-colab.lock.txt`. The
    # capstone's lab.lock is not reused: it takes torch from the CUDA 12.6 index and lacks peft, torchao and torchaudio.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "run_all": (
        "Selecting **Run all** in a fresh **GPU** runtime (a 15 GB T4 is enough; see the Prerequisites) builds an isolated "
        "environment from the hash-locked pins (torch, diffusers, transformers, peft, bitsandbytes, accelerate, sentencepiece, "
        "safetensors, huggingface-hub, numpy, pillow; the kernel's own packages are left alone, so no restart is needed), stages "
        "and digest-verifies two pinned snapshots — the 33.7 GB FLUX.1 [schnell] diffusers-layout snapshot (transformer, CLIP-L and "
        "T5-XXL encoders, tokenizers, VAE, scheduler), fetched from an ungated mirror and checked byte for byte against the "
        "manifest the repository committed, and a 0.6 GB CLIP scorer — fetches 60 CC0 iNaturalist bird photographs as "
        "digest-verified JPEGs (6 MB, no credential), validates them and splits them 36 / 12 / 12 by seed, encodes every prompt "
        "with the two text encoders and releases them, loads the 12 B transformer 4-bit (NF4) with an untrained LoRA adapter "
        "attached, scores the frozen model — the held-out flow-matching loss on the validation and test photographs, and six "
        "four-step generations scored by CLIP against their prompts and the held-out photographs, beside the same scores on the "
        "real photographs (a leave-one-out reference line) — runs a bounded QLoRA fine-tuning (3 epochs over 36 images), scores "
        "the adapted model on identical inputs, renders a new prompt, exports the adapter as safetensors with a manifest, releases "
        "the adapted transformer, reloads the artifact into a fresh 4-bit pipeline to verify parity, and releases that pipeline "
        "too. The default path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no "
        "configuration edit (NOTEBOOK_SPEC 2.2 §5). Measured times come from the previous notebook version, whose model stages "
        "are the same: a Kaggle Tesla T4 run took 2,912 s of cell time (2026-09-20) — 536 s staging the 34 GB of snapshots and "
        "about 36 minutes for Sections 4–9, 17 of them fine-tuning — with a pinned in-kernel install this version no longer does. "
        "Building the isolated environment (PyTorch with its CUDA libraries) comes on top and usually takes a few minutes (an "
        "estimate; no run of this version is recorded yet)."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4, select that cell and choose **Runtime → Run "
        "after** to supply your own captioned photographs: a zip (or, with `BYOD_PATH`, a zip or a folder on any runtime) holding "
        "`captions.csv` (columns `id`, `file`, `caption`) and the image files (JPEG or PNG, shorter side 256..4096 px). **Minimum:** "
        "at least six distinct images, at least one caption with three or more images (it supplies the held-out records), and "
        "at least four records left for training — six images of one caption is the smallest dataset that works. Your records "
        "are split **stratified within each caption** (each caption with three or more images gives about 20 % to validation and "
        "test and keeps the rest for training) and flow through the same contract — validation, prompt encoding, frozen "
        "baseline, adaptation, held-out evaluation, generation, artifact export and reload parity. Section 5 first releases "
        "everything the previous pass left on the GPU (the transformer, the reloaded pipeline, the scorer), so the text encoders "
        "fit again, and Sections 6 and 7 start from the pretrained base. The expected schema, the limits and the privacy guidance "
        "are stated in the Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is optional and "
        "never part of the default path."
    ),
    "pipeline_class": "FluxSchnellPipeline",
    "model_load": "FluxSchnellPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, use_lora=True)",
    "weights_key": "flux1-schnell",
    "modules": ["pipeline.py", "samples.py", "metrics.py"],
    "entry_module": "pipeline.py",
    # generator /2: the two weights directories derive from one shared root; the second pinned snapshot (CLIP scorer)
    # is carried, staged and verified by the model cell.
    "rewrites": [
        [
            r"^_WEIGHTS_ROOT = Path\(__file__\)[^\n]*$",
            '_WEIGHTS_ROOT = Path.cwd() / "weights"  # standalone rewrite (build_notebook.py): working-directory-relative',
        ]
    ],
    "extra_weights": [
        {
            "key": "clip-vit-b-32-laion2b",
            "var": "SCORER_MANIFEST",
            "dir": "SCORER_WEIGHTS_DIR",
            "identity": ["SCORER_ID", "SCORER_REVISION"],
            "stage": "stage_missing_scorer_files",
            "verify": "verify_scorer_snapshot",
        },
    ],
    "model_host": {
        "name": "the Hugging Face Hub (the upstream identity is `black-forest-labs/FLUX.1-schnell`; the bytes are served by the ungated mirror `unsloth/FLUX.1-schnell` at `9df3faa7…`, see Section 3)",
        "reference_url": "https://huggingface.co/black-forest-labs/FLUX.1-schnell",
        "revision_label": "upstream revision",
    },
    "model_cell_note": (
        "The bytes themselves come from the ungated mirror `STAGING_ID` = `unsloth/FLUX.1-schnell` at `STAGING_REVISION` = `9df3faa7…` "
        "(`verify_snapshot` checks that the manifest names exactly that source): the upstream repository is click-through gated and hides "
        "its LFS digests behind the gate, so what makes the mirror trustworthy here is the manifest — every byte loaded is one whose "
        "SHA-256 the repository committed, and the 23 files match the upstream tree file for file and byte for byte in size. There is "
        "no fallback to a different download and no remote model code is executed; the model classes come from `diffusers`, "
        "`transformers`, `peft` and `bitsandbytes` on PyPI. `from_pretrained` loads the VAE, both tokenizers and the scheduler config "
        "and leaves the transformer for Section 6: a 4-bit 12 B transformer (6.6 GB) and the two 16-bit text encoders (9.7 GB) do not "
        "fit a 16 GB GPU together, so prompts are encoded first."
    ),
    "runtime_imports": ["torch", "diffusers", "transformers", "peft", "bitsandbytes"],
    "title": "FLUX.1 [schnell] — DIMER E2E text-to-image QLoRA fine-tuning tutorial (standalone)",
    "badges": BADGES,
    "capability": "few-step text-to-image generation with a 12 B-parameter rectified-flow transformer loaded 4-bit, held-out flow-matching-loss and CLIP-scored evaluation, and bounded QLoRA fine-tuning to a set of captioned photographs",
    "intro": (
        "FLUX.1 [schnell] (Black Forest Labs, 2024) is a 12 B-parameter rectified-flow transformer distilled to render an image in one "
        "to four sampling steps with no classifier-free guidance. A CLIP-L encoder gives one pooled vector, a T5-XXL encoder gives 256 "
        "token embeddings, 19 double-stream and 38 single-stream transformer blocks predict the *velocity* of a 16-channel latent — the "
        "FLUX VAE's 8× compressed image, packed 2×2 into 64-channel tokens — and the VAE decodes the latent to pixels. Everything comes "
        "from one 33.7 GB snapshot in the diffusers layout, plus, for evaluation only, a CLIP ViT-B/32 scorer.\n\n"
        "Three things about this row are handled in the open. **The transformer does not fit a 16 GB GPU in any 16-bit format** "
        "(23.8 GB as shipped), so it is loaded 4-bit — bitsandbytes NF4 with double quantisation, float16 compute — at about 6.6 GB, "
        "and the LoRA is trained over that quantised base (QLoRA); the frozen numbers are therefore the 4-bit model's numbers, not the "
        "bfloat16 checkpoint's. **The text encoders (9.7 GB in float16) do not fit beside it**, so Section 5 encodes every prompt the "
        "notebook will ever use once, keeps the embeddings, and releases both encoders before the transformer loads; the reloaded "
        "pipeline in Section 9 adopts the same embeddings rather than loading them again. **Generation has no ground truth**, so the "
        "notebook reads three kinds of number and says what each is: the held-out *flow-matching loss* (the training objective, "
        "measured on photographs the model never trained on, with identical noise for the frozen and the adapted model), CLIP scores "
        "of generated images (prompt alignment, which species CLIP thinks it sees, and closeness to the held-out real photographs), "
        "and the same CLIP scores on the real photographs themselves — a **reference line, not a ceiling**: each photo is compared "
        "with the other held-out photos of its species, never with itself, and generated images can score above it. None of these "
        "is a human judgement of image quality.\n\n"
        "**Who this is for.** A learner who knows basic Python and PyTorch, has met the idea of a train/validation/test split, and "
        "wants to see how a large pretrained image generator is adapted to a narrow domain and measured honestly when there is no "
        "ground truth: a loss on held-out photographs, instrument readings from a second model, a reference line from real photos, "
        "and an adapter that reloads with verified parity. No prior experience with diffusion or flow models, quantisation or LoRA "
        "is assumed; each term is explained where it is first used and again in the **Glossary** at the end. A CUDA GPU of at least "
        "15 GB is required.\n\n"
        "**Input → Model → Output.**\n\n"
        "| | Generation (Sections 6, 8, 9) | Adaptation (Section 7) |\n"
        "|---|---|---|\n"
        "| Input | a text prompt (a caption of 1..1000 characters) and a seed | captioned photographs `id`, `image`, `caption`, split into training, validation and test |\n"
        "| Model | FLUX.1 [schnell], 4-bit transformer + VAE, prompts pre-encoded by CLIP-L and T5-XXL; four Euler steps, no guidance | the same 4-bit transformer with a rank-8 LoRA on the image-stream attention projections; only the LoRA is trained |\n"
        "| Output | a 512 × 512 image, scored by a separate CLIP model: prompt similarity, nearest caption, similarity to the held-out photographs | the held-out flow-matching loss before and after, the same CLIP scores before and after, and a 37 MB safetensors adapter that reloads with identical results |\n\n"
        "**How to use this notebook.** Choose a GPU runtime (Colab T4 or better), then **Runtime → Run all**. Sections 1–3 are "
        "**infrastructure** — the isolated environment, the carried code and the model verification — and can be run without study; "
        "their code is collapsed. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be "
        "edited; the defaults reproduce the default path. Each learner section asks you to **predict** before it runs; the next "
        "section opens with **What to notice** and a collapsible **Check your reasoning** block with a worked answer from a recorded "
        "run. Each optional experiment names the field to change and the cell to re-run from (**Runtime → Run after**). Section 10 "
        "is a **change-one-thing activity**, off by default. **Troubleshooting**, a **Glossary** and a **Conclusion** template are "
        "at the end. Your notes are optional.\n\n"
        "**Roadmap:** *core concepts* — 4 a captioned dataset and its split → 5 prompts become embeddings, and the encoders leave "
        "the GPU → 6 the frozen model: a held-out loss, CLIP readings and the real-photo reference line; *evaluation practice* — 7 "
        "bounded QLoRA fine-tuning → 8 the paired held-out comparison; *engineering* — 9 a new prompt, export, reload parity → 10 "
        "**change one thing: the number of sampling steps** → conclude."
    ),
    "learning_objectives": (
        "by the end you should be able to (1) explain why the text encoders and the 4-bit transformer take turns on a 16 GB GPU, "
        "from the memory figures Section 5 prints (Section 5); (2) say what the held-out flow-matching loss measures and why the "
        "frozen and adapted values are a paired comparison (Sections 6 and 8); (3) read a CLIP prompt similarity, a nearest-caption "
        "accuracy and a reference similarity, and say why the real photographs are a reference line, not a ceiling "
        "(Section 6); (4) state which parameters QLoRA trains and how the kept epoch is chosen (Section 7); (5) judge from the "
        "per-caption rows what six generated images per model do and do not show (Section 8); (6) check that an exported adapter "
        "reproduces the evaluated model (Section 9); and (7) predict, measure and explain how the number of sampling steps changes "
        "the CLIP readings (Section 10)."
    ),
    "exclusions": (
        "FLUX.1 [dev] or [pro], guidance-distillation or classifier-free guidance (schnell has none), ControlNet, inpainting or "
        "image-to-image conditioning, resolutions other than 512², full or 16-bit fine-tuning, DreamBooth identifiers, safety filtering "
        "of prompts or images, human preference or FID/KID benchmarks, prompt engineering, and any claim that a CLIP score or a "
        "flow-matching loss measures image quality. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Learner:** basic Python and PyTorch; what a train/validation/test split protects against. The notebook explains rectified flow, velocity, noise levels, the flow-matching loss, latents and the VAE, NF4 quantisation, LoRA and QLoRA, CLIP scores, the leave-one-out reference and reload parity where they are first used; the Glossary repeats them.",
        "- **Runtime:** a fresh **Linux x86_64 GPU** runtime — Google Colab T4 or better, Kaggle, or a Linux Jupyter server with a CUDA GPU of at least 15 GB. Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the kernel's own Python version does not matter, and a Windows or macOS kernel is not supported (Section 1 stops with that message). The CLIP-L and T5-XXL encoders run in float16 (9.7 GB) while they encode prompts and are then released; the transformer is loaded 4-bit (NF4, about 6.6 GB) with float16 compute and the LoRA parameters in float32; the FLUX VAE stays in float32 (0.34 GB). CPU-only runtimes are not supported: bitsandbytes 4-bit needs CUDA and the 12 B transformer would need 48 GB of RAM in float32. **Disk:** about 36 GB for the two snapshots plus several GB for the isolated environment (PyTorch with its CUDA libraries). **Host RAM:** every model load streams its weights to the GPU one tensor at a time (`device_map`), so a standard Colab VM (12.7 GiB) is meant to be enough; this repository's capstone notebook ran the same loading pattern on such a VM, but this notebook has not yet been run on Colab.",
        "- **Knowledge:** what a latent diffusion or flow-matching model does at inference (noise → latent → image), why a distilled few-step model uses no classifier-free guidance, what 4-bit weight quantisation changes, what a LoRA adapter changes and what it does not, and why a training loss is not a quality score. Each is explained again where it is used.",
        "- **Weights:** the transformer, both text encoders, the VAE and the CLIP scorer are all safetensors; nothing is unpickled and no Hub-hosted code is executed — the model classes come from `diffusers`, `transformers`, `peft` and `bitsandbytes` on PyPI. FLUX.1 [schnell] is released under the Apache-2.0 licence; the scorer is MIT. The upstream Hub repository is gated by a click-through, so the bytes are fetched from an ungated, unmodified mirror and verified against the manifest committed in this repository (Section 3).",
        "- **Data contract:** a record is `{id, image, caption}` — an RGB image with shorter side 256..4096 px (resized so the shorter side is 512 px and centre-cropped to 512 × 512; the crop is reported) and a caption of 1..1000 characters (truncated to 256 T5 tokens and 77 CLIP tokens; T5 truncations are reported). Validation is structural: nothing checks that a caption describes its image or that the model can render it.",
        "- **BYOD file (Section 4):** a zip holding `captions.csv` (header `id,file,caption`, UTF-8) and the images; a `file` value is the image's path inside the zip (relative to the archive root or to the folder holding `captions.csv`) or, when unique, its file name. On Colab the upload dialog opens; on any runtime set `BYOD_PATH` to a zip or a folder instead. **Minimum:** six distinct images, at least one caption with three or more images, and four records left for training — six images of one caption is the smallest dataset that works. Pixel-identical duplicates are dropped and reported by id. Generation and scoring use one image per distinct training caption, capped by `MAX_GENERATION_PROMPTS` (12) in Section 6 with a printed warning. Your test captions are written to `outputs/flux_schnell_generation_byod_captions.csv` (the sample's to `outputs/flux_schnell_generation_sample_captions.csv`, which also shows the expected shape).",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — photographs of identifiable people, licensed stock images or client material are exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the Hub, the default path fetches 60 pinned photographs (about 6 MB) from the public iNaturalist open-data bucket `inaturalist-open-data.s3.amazonaws.com` over HTTPS, digest-verified before decoding; every photo is CC0 and its observation page is recorded.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Sample photographs, validation and splits\n\n"
                "The default dataset is 60 research-grade iNaturalist photographs of six common North American birds — 10 per "
                "species, one per observer per species, every one CC0 — fetched by photo id from the open-data bucket and "
                "refused on any byte-size or SHA-256 mismatch (`fetch_corpus`). Each photo's caption is generated from its "
                "species by one template, so the adaptation teaches the generator what six names look like in this kind of "
                "photograph. `build_sample_dataset` draws a seeded split stratified by species — 6 / 2 / 2 per species for "
                "training, validation and test — and `dataset_manifest` validates every split, checks that no image appears "
                "twice and records a digest.\n\n"
                "Look for: 36 / 12 / 12 records, six distinct captions, a shorter side around 300..500 px (every photo is "
                "centre-cropped to 512²), a written `outputs/{stem}_sample_captions.csv` in the shape BYOD expects, and "
                "three refusal probes — a missing caption, a 200 px image, a duplicate id — each rejected before the model runs.\n\n"
                "**Predict before running:** each species gets two test photographs. Section 6 compares every real test photo with "
                "the *other* test photos of its species. How many photos is each real photo compared with, and what does that "
                "suggest about how steady that reference number will be?"
            ),
            "code": _DATA_CODE,
        },
        {
            "md": (
                "## 5. Encode every prompt, then release the text encoders\n\n"
                "**What to notice (Section 4):** 36 / 12 / 12 records, six captions, and three refusals printed before any model "
                "ran. Every caption appears in all three splits: the split is stratified *within* each species, so the held-out "
                "photos test new photographs of known captions, not new captions.\n\n"
                "<details><summary>Check your reasoning</summary>One. With two test photos per species and the photo itself "
                "excluded, each real photo's reference is the single other test photo of its species, so the real-photo reference "
                "similarity in Section 6 is a cosine between two photographs and swings with how alike those two happen to be. In "
                + RECORD
                + " the per-species values range from about 38 to 84 (derived from that run's numbers). The generated images "
                "are compared with the mean of both test photos, which is a smoother target.</details>\n\n"
                "`pipe.encode_prompts` loads CLIP-L and T5-XXL from the verified snapshot (float16 on CUDA, streamed to the GPU one "
                "tensor at a time), tokenises each distinct prompt — 77 CLIP tokens for the pooled vector, 256 T5 tokens for the "
                "sequence — runs both encoders once per prompt and keeps the (256, 4096) embeddings and the (768,) pooled vector on "
                "the CPU. Encoded here: the six training captions (which are also the validation and test captions and the "
                "generation prompts) and one new prompt for Section 9. schnell is guidance-distilled, so there is no negative prompt "
                "to encode. `release_text_encoder` then drops the 9.7 GB of encoders so the 4-bit transformer, the VAE, the scorer and "
                "a training graph fit on a 16 GB GPU.\n\n"
                "**Every pass starts with an empty GPU.** On a re-run (a BYOD run, or an experiment re-run from here), "
                "`release_gpu_residents()` first frees what the previous pass left: the transformer of `pipe`, the `reloaded` "
                "pipeline of Section 9 if it still exists, and the CLIP scorer (Section 6 loads a new one). The two never share "
                "the GPU with the encoders. This cell also defines `reset_to_pretrained()`, which Sections 6, 7 and 10 call so "
                "that a *frozen* number is never read from an adapted model.\n\n"
                "Look for: the encoders loading in about a minute from the bfloat16 shards, seven prompts encoded in seconds, no "
                "truncation, finite embeddings, and the GPU memory falling back to well under 1 GB after the release.\n\n"
                "**Predict before running:** how much GPU memory will the encoders occupy while they encode, and how much will "
                "remain once they are released, given that the embeddings are kept on the CPU?"
            ),
            "code": _ENCODE_CODE,
        },
        {
            "md": (
                "## 6. Load the 4-bit transformer; the frozen model's held-out loss and CLIP-scored generations\n\n"
                "**What to notice (Section 5):** the memory with the encoders loaded and after the release. On a re-run, the "
                "`released_before_encoding` line lists what an earlier pass had left on the GPU.\n\n"
                "<details><summary>Check your reasoning</summary>In "
                + RECORD
                + " the GPU held 12.15 GB with both encoders loaded and 0.35 GB after the release — the float32 VAE and "
                "allocator overhead; the seven prompts' embeddings are on the CPU. That 12 GB is why the 6.6 GB transformer cannot "
                "load until the encoders are gone, and why a re-run must first release whatever 4-bit transformer is still "
                "resident.</details>\n\n"
                "`pipe.load_transformer` reads the three bfloat16 shards (23.8 GB) and quantises every linear projection of the 57 "
                "blocks to 4-bit NF4 with double quantisation as it loads — about 6.6 GB on the GPU with float16 compute — checks the "
                "parameter count against the checkpoint (11,891,178,560), freezes everything and attaches the untrained LoRA adapter "
                "(`use_lora=True`; its B matrices start at zero, so until Section 7 this is the pretrained model). "
                "`reset_to_pretrained()` runs first, so an experiment that re-runs this cell after Section 7 scores the pretrained "
                "model, not the adapted one. Two kinds of number are read here and kept for the comparison.\n\n"
                "**Held-out flow-matching loss** (`pipe.evaluate`): each held-out photograph is VAE-encoded and packed, mixed with a "
                "seeded noise tensor at five fixed noise levels σ = 0.1, 0.3, 0.5, 0.7 and 0.9 as x_σ = (1 − σ)·x₀ + σ·ε, and the "
                "transformer's velocity prediction is scored against the rectified-flow target ε − x₀ (MSE over the tokens). It is the "
                "training objective measured on photographs the model never trains on; the same seed gives the same latents, noise and "
                "noise levels later, so the adapted number is a paired comparison, not a re-draw.\n\n"
                "**CLIP-scored generations** (`pipe.generate` + `score_generations`): one image per training caption at fixed seeds "
                "(4 Euler steps, no guidance, 512²), scored by the frozen CLIP ViT-B/32 on prompt alignment (cosine × 100), zero-shot "
                "label accuracy (which of the six captions is nearest — an argmax with no threshold) and similarity to the mean embedding "
                "of the held-out real photographs of that species.\n\n"
                "**The real-photo reference line** (`real_photo_reference`) scores the 12 real test photographs the same way, "
                "**leave-one-out**: each photo's reference similarity is measured against the other test photos of its caption, "
                "never against itself. It is a reference to read the generations against, **not a ceiling**, and generated images "
                "can score above it: a generator conditioned on the prompt can match the prompt's CLIP embedding more closely than a "
                "photograph with its own background and pose, and each generation is compared with the mean of the two test photos "
                "while each real photo is compared with one. Only label accuracy reads naturally as a target, because the captions "
                "describe exactly these photographs.\n\n"
                "Look for: the load taking a few minutes, a flow-matching MSE around 0.5..1.5, label accuracy at or below the real "
                "photographs', and the first grid of generated birds displayed below the numbers (also written to "
                "`outputs/{stem}_frozen_grid.jpg`).\n\n"
                "**Predict before running:** on CLIP prompt similarity, will the frozen model's generated birds score above or "
                "below the real photographs? And on reference similarity?"
            ),
            "code": _FROZEN_CODE,
        },
        {
            "md": (
                "## 7. Bounded QLoRA fine-tuning\n\n"
                "**What to notice (Section 6):** the frozen generations next to the real-photo reference line, metric by metric, "
                "and the per-species rows: which birds did CLIP name correctly?\n\n"
                "<details><summary>Check your reasoning</summary>Above, on both. In "
                + RECORD
                + " the frozen generations scored prompt similarity 31.08 against the real photographs' 30.25 and reference "
                "similarity 71.95 against a leave-one-out real-photo value of 57.7, while label accuracy was 0.667 against the real "
                "photos' 0.917. That run printed 88.66 for the real photos' reference similarity because each photo was then "
                "included in its own reference; excluding it gives 57.7. So the reference line is not a ceiling on two of the three "
                "readings: the generator is conditioned on the prompt and is compared with a two-photo mean, while each photo is "
                "compared with one other photo. Label accuracy is the reading on which the frozen model sat clearly below the real "
                "photographs.</details>\n\n"
                "`pipe.adapt` trains the 380 LoRA tensors (rank 8, 9,338,880 parameters — 0.08 % of the transformer) that `peft` "
                "attached to the image-stream query, key, value and output projections of all 57 blocks, and nothing else; the 4-bit "
                "base, the VAE and the text encoders are frozen. Each step takes one training photograph's packed latent (VAE-encoded "
                "once, seeded), draws a noise level σ uniformly from (0, 1) and a noise tensor (both seeded), mixes x_σ = (1 − σ)·x₀ + "
                "σ·ε, and minimises the MSE between the predicted velocity and ε − x₀; AdamW at a fixed learning rate on float32 LoRA "
                "masters, gradient-norm clipping at 1.0, float16 autocast with loss scaling and gradient checkpointing (the 1,280-token "
                "activations of 57 blocks would not otherwise fit). Epoch 0 records the frozen model's validation loss, and the epoch "
                "with the lowest validation flow-matching loss is kept. `reset_to_pretrained()` runs first, so re-running this cell "
                "with a different `EPOCHS` or `LEARNING_RATE` trains again from the pretrained base (`adapt` refuses to continue from "
                "an already trained adapter).\n\n"
                "Watch the validation loss from epoch 0; three epochs over 36 images (108 steps) took 17 minutes on a T4 in "
                + RECORD
                + ", most of it the per-epoch validation pass. The training loss is a noisy per-step average over random noise "
                "levels and is not the quality signal — the paired held-out numbers in Section 8 are.\n\n"
                "**Predict before running:** will the validation loss fall at every epoch? Will the training loss?"
            ),
            "code": _ADAPT_CODE,
        },
        {
            "md": (
                "## 8. Held-out evaluation: the paired comparison\n\n"
                "**What to notice (Section 7):** the validation loss per epoch, the kept epoch and the peak GPU memory of training.\n\n"
                "<details><summary>Check your reasoning</summary>In "
                + RECORD
                + " the validation loss fell at every epoch (0.834 → 0.595 → 0.534 → 0.524) and epoch 3 was kept, while the "
                "training loss went 0.624 → 0.669 → 0.569: up, then down. Each training step draws its own random noise level, and "
                "the loss depends strongly on it (Section 6 prints the per-σ loss), so a per-epoch training average is noisy; the "
                "validation loss uses the same fixed noise every time and is the signal used to choose the epoch. Peak training "
                "memory was 12.23 GB.</details>\n\n"
                "The test photographs were never used for training or epoch selection. The adapted model is scored exactly as "
                "the frozen model was in Section 6 — the same seed, so the same latents, noise and noise levels, and the same "
                "prompt/seed pairs for generation — and the table puts the frozen, the adapted and the real-photo reference numbers "
                "side by side. The cell asserts only what the procedure guarantees — the kept epoch's validation loss is no higher "
                "than the frozen model's (epoch 0) and the re-scored validation loss matches the history — and prints the test "
                "comparison without a directional assertion: a lower test flow-matching MSE is what to look for, not what is "
                "promised. The frozen and adapted images are displayed side by side (top row frozen, bottom row adapted; each "
                "column one prompt and seed). Six images per model from one seeded run give no dispersion estimate; these are "
                "sample-sanity numbers that show the adaptation contract works, not a benchmark, and CLIP agreement is not a human "
                "judgement of quality.\n\n"
                "**Predict before running:** will every species' reference similarity rise after adaptation?"
            ),
            "code": _COMPARE_CODE,
        },
        {
            "md": (
                "## 9. A new prompt, artifact export and fresh reload\n\n"
                "**What to notice (Section 8):** the per-species rows, not only the means — and whether the bottom row of the "
                "side-by-side grid looks more like the photographs of each species than the top row.\n\n"
                "<details><summary>Check your reasoning</summary>Not every one. In "
                + RECORD
                + " the test flow-matching MSE fell from 0.795 to 0.507 and was lower at every noise level, label accuracy "
                "rose from 0.667 to 0.833 and the mean reference similarity from 71.95 to 75.01 — but two of the six species "
                "moved the other way (Chipping Sparrow 70.24 → 68.58, Dark-eyed Junco 69.16 → 68.69). With one image per species "
                "and one seed, a single image can move a species' number either way; the paired loss on 12 photographs at five "
                "noise levels is the steadier evidence.</details>\n\n"
                "The adapted model renders `NEW_PROMPT` — a composition that appears in no training caption — at two seeds; the "
                "CLIP prompt similarity is printed as a sanity check, not an evaluation, and both images are displayed. One more "
                "image of the first training prompt is rendered and kept for the parity check.\n\n"
                "`pipe.save_artifact` writes the 380 trained tensors (about 37 MB in float32) as `adapter.safetensors` with a "
                "`manifest.json` recording the artifact format, the base model's id, revision, licence, staging source and "
                "quantisation, the LoRA configuration, the tensor names, the file size and SHA-256, the training configuration and "
                "the epoch history (OUT8). Two 4-bit transformers do not fit a 16 GB GPU, so the adapted pipeline then **releases** "
                "its transformer (`release_transformer`, which also forgets the adapter) before `FluxSchnellPipeline.from_artifact` "
                "re-verifies both snapshots, checks the manifest, the LoRA scope and the digest **before** deserialising, loads a fresh "
                "4-bit transformer with the adapter attached and overlays the tensors — a new object from files, not the in-memory "
                "model (VER2). The fresh pipeline adopts the prompt embeddings already encoded (so the text encoders are not loaded "
                "again), and the cell asserts that it reproduces the held-out flow-matching loss recorded in Section 8 and the same "
                "image for the same prompt and seed (VER4: a mean absolute pixel difference below 1 on the 0..255 scale — same device, "
                "same kernels, same deterministic NF4 quantisation of the same bytes). Once parity is checked the reloaded pipeline is "
                "**released** too, so no 4-bit transformer stays resident for a later pass. The result file records the per-module "
                "SHA-256 of the carried code next to the source revision.\n\n"
                "**Predict before running:** will the reloaded pipeline reproduce the adapted model's numbers exactly, or only "
                "closely?"
            ),
            "code": _EXPORT_CODE,
        },
        {
            "md": (
                "## 10. Your turn — change one thing: the number of sampling steps\n\n"
                "**What to notice (Section 9):** the reload parity line, the GPU memory with the reloaded pipeline and after its "
                "release, and the list of files under `outputs/`.\n\n"
                "<details><summary>Check your reasoning</summary>Exactly, on the same runtime. In "
                + RECORD
                + " both differences were 0.0: the same bytes quantised the same way, the same embeddings and the same kernels "
                "give the same arithmetic. On a different GPU, driver or library build the numbers would differ slightly, which is "
                "why parity is checked in one session. That run held 7.48 GB with the reloaded pipeline resident and did not "
                "release it; this version releases it, and the printed memory should fall back to about 1 GB (the VAE and the CLIP scorer).</details>\n\n"
                "**Predict → Change one thing → Run → Observe → Explain.**\n\n"
                "1. **Predict:** schnell is distilled to render in one to four steps. With one or two steps instead of four, will "
                "the CLIP prompt similarity and the label accuracy of the pretrained model's images go up, down or stay put? Write "
                "your guess down.\n"
                "2. **Change one thing:** set `RUN_ACTIVITY = True` in the cell below and nothing else. It renders the same "
                "prompts with the same seed at 1, 2 and 4 steps with the **pretrained base** (an untrained LoRA; "
                "`reset_to_pretrained()` runs first).\n"
                "3. **Run:** run this cell only (it loads the transformer again, a few minutes, and releases it at the end). It "
                "writes only under `outputs/activity/` and changes no earlier result.\n"
                "4. **Observe:** one row per step count — prompt similarity, label accuracy, reference similarity and seconds — and a "
                "grid with one row per step count, each column one prompt.\n"
                "5. **Explain:** what did fewer steps cost on each reading, what did they save in time, and how far apart must two "
                "rows be before six images make the difference believable?\n\n"
                "<details><summary>Check your reasoning</summary>No run of this activity is recorded yet, so compare your "
                "prediction with what the cell printed rather than with a number here. Two things to check either way: the "
                "4-step row should reproduce Section 6's frozen numbers (same model, prompts, seed and steps) when the runtime is "
                "the same, which makes it the control row; and the time should fall roughly in proportion to the steps, because "
                "every step is one pass of the 12 B transformer. Distillation is what lets few steps work at all, so a small "
                "change between rows is a plausible outcome, and with six images per row a difference of one correct label is "
                "0.167 of label accuracy.</details>"
            ),
            "code": _ACTIVITY_CODE,
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "**What to notice (Section 10):** if you ran the activity, whether the 4-step row matches Section 6 and how the "
        "readings and the seconds changed with fewer steps.\n\n"
        "Read your own Section 8 before concluding; the claim below is what the run is designed to test, not a result stated in "
        "advance. If the test flow-matching loss fell and the generations moved towards the held-out photographs, a LoRA of nine "
        "million parameters trained for about 17 minutes on 36 photographs over a 4-bit 12 B base has taught the "
        "rectified-flow transformer something about a narrow visual domain from a handful of captioned images on a 16 GB GPU, "
        "measured on identical inputs before and after, and the artifact that carries the change is 37 MB. Check the "
        "per-species rows: in the recorded run of the previous version two of the six species' reference similarities fell "
        "while the mean rose.\n\n"
        "The numbers are sample-sanity evidence. A flow-matching loss is the training objective, not a quality score; CLIP similarity "
        "and CLIP's nearest-caption vote are a frozen model's opinion, not a human judgement, and CLIP itself has biases about what a "
        "species name looks like; six images per model from one seeded run give no dispersion estimate; and nothing here measures "
        "aesthetics, diversity, artefacts or prompt fidelity beyond the six captions. The real photographs are a **reference "
        "line, not a ceiling**: generated images can score above them on prompt similarity and reference similarity, so a "
        "generation that beats a photograph on a CLIP reading is not a better bird picture. Every number is the **4-bit** model's: NF4 "
        "quantisation changes the base's outputs relative to the bfloat16 checkpoint by an amount this notebook does not measure, and "
        "an adapter trained over the quantised base is meant to be served over it (the artifact manifest records `nf4`). Fine-tuning "
        "on a narrow domain can also erode the model elsewhere — the new prompt in Section 9 is a sanity check on one composition, "
        "not a test of generality.\n\n"
        "**Pretraining overlap.** The 60 photographs are public iNaturalist images. Photographs like them — possibly these ones — "
        "may be in the web-scale data behind the LAION-2B CLIP scorer and in FLUX's undisclosed training set, so both the "
        "frozen generator's and the scorer's familiarity with these species can be inflated; nothing here measures that.\n\n"
        "**Run-to-run variability.** The seeds fix the data split, the noise, the noise levels and the generation seeds, not "
        "the arithmetic: 4-bit matrix multiplications, attention and float16 autocast are not bit-reproducible across GPU "
        "models, drivers and library builds. A run on another runtime gives slightly different losses, images and CLIP "
        "readings — enough to flip a borderline nearest-caption vote — while reload parity within one session stays exact.\n\n"
        "Three things to carry to real data. **Captions are the contract:** the adapter learns the association between the "
        "caption text and the images; a caption that does not describe its image, or one caption for very different images, "
        "teaches noise. **Held-out photographs, known captions:** the split is stratified within each caption, so every "
        "held-out caption also appears in training and the held-out loss measures generalisation to new photographs of known "
        "captions, not to new captions; a caption with fewer than three images is used for training only. **Licences travel "
        "with the outputs:** FLUX.1 [schnell] is Apache-2.0 and the training photographs here are CC0 — with your own data, the "
        "rights to the images and to what the adapter produces are yours to establish.\n\n"
        "A successful default run **proves** that the pipeline modules carried in this standalone notebook (their per-module "
        "SHA-256 values are in the result file) can stage a pinned safetensors snapshot from a mirror and digest-verify it "
        "against a committed manifest, fetch and validate digest-pinned real photographs, encode prompts and release the "
        "encoders, load a 12 B transformer 4-bit, execute bounded QLoRA fine-tuning, evaluate the frozen and the adapted model "
        "on identical held-out inputs beside a leave-one-out real-photo reference, and emit the shown machine-readable artifacts "
        "— without the repository being reachable. It does **not** establish benchmark superiority, production fitness, or "
        "image quality beyond the checks shown.\n\n"
        "**Next experiments** (each starts after the default Run all; Sections 6 and 7 always start again from the pretrained "
        "base):\n\n"
        "1. **Sampling steps:** the Section 10 activity (`RUN_ACTIVITY = True`), run that cell only.\n"
        "2. **Training length or rate:** in Section 7 set `EPOCHS` or `LEARNING_RATE`, then select Section 7 and choose "
        "**Runtime → Run after** (Sections 7–10 are recomputed; the frozen numbers of Section 6 stay valid). Watch the "
        "validation loss for the epoch where it turns.\n"
        "3. **Steps for the whole comparison:** in Section 6 set `STEPS = 1` or `2`, then **Run after** from Section 6. The "
        "frozen and adapted models are both scored at that step count.\n"
        "4. **A less noisy label accuracy:** in Section 6 set `IMAGES_PER_PROMPT = 2`, then **Run after** from Section 6.\n"
        "5. **Your own captioned photographs:** in Section 4 set `USE_BYOD = True` (upload, or set `BYOD_PATH`), then **Run "
        "after** from Section 4. Section 5 frees the GPU first; compare the adapted numbers with your photographs' "
        "leave-one-out reference line.\n\n"
        "## Troubleshooting\n\n"
        "- **Section 1 stops with \"needs a Linux x86_64 runtime\".** The locked environment is built from manylinux wheels; use Colab, "
        "Kaggle or a Linux Jupyter server with a CUDA GPU.\n"
        "- **Section 1 fails to download `uv`, Python or a package.** The runtime needs `pypi.org`, `files.pythonhosted.org` and the "
        "python-build-standalone release host. Re-run the cell; a size or SHA-256 mismatch is refused on purpose.\n"
        "- **Section 3 is slow or stops while downloading.** It fetches about 34 GB from the Hub; an anonymous download can be "
        "rate-limited. Run Section 3 again: files already staged and verified are not fetched twice. A size or SHA-256 "
        "mismatch is refused; delete the named file under `weights/` and run Section 3 again.\n"
        "- **\"No space left on device\".** The snapshots need about 36 GB and the isolated environment several more; free disk "
        "or use a runtime with more of it.\n"
        "- **\"The isolated environment's Python process exited\".** Usually out of host or GPU memory. Restart the session and "
        "choose **Run all** again.\n"
        "- **CUDA out of memory.** The encoders (Section 5) and a 4-bit transformer never fit together. Re-run from Section 5 "
        "(**Run after**), which releases every resident first; if it persists, restart the session and choose **Run all**. "
        "Keep `BATCH_SIZE = 1` on a 16 GB GPU.\n"
        "- **Section 4: \"Upload exactly one .zip … (got 0 files)\".** The upload was cancelled or empty. Run Section 4 again and "
        "pick one zip, or set `BYOD_PATH`.\n"
        "- **Section 4 refuses a BYOD dataset.** The message names the file or the rule — a member missing from the zip, an "
        "unreadable image, a side outside 256..4096 px, a missing column, or too few images for the split (six distinct "
        "images, one caption with three or more). Fix the zip and run Section 4 again.\n"
        "- **Section 8 or 9: \"no adapted model is loaded\".** Section 9 released the adapted transformer after exporting it; "
        "run Section 7 first (**Run after** from Section 7).\n"
        "- **Section 9's parity assertion fails.** The export or reload is broken; do not use that artifact. Run Sections 7–9 again.\n\n"
        "## Glossary\n\n"
        "- **Rectified flow / velocity:** the model learns to move a noisy latent towards a clean one along a straight line; "
        "at each step it predicts the direction and speed of that move (the *velocity*), whose target is ε − x₀.\n"
        "- **Noise level σ:** how much noise is mixed in, x_σ = (1 − σ)·x₀ + σ·ε; σ = 0 is the clean latent, σ = 1 pure noise.\n"
        "- **Flow-matching loss:** the mean squared error between the predicted and the target velocity; the training objective, "
        "here also measured on held-out photographs at five fixed σ.\n"
        "- **Latent / VAE / packed tokens:** the VAE compresses a 512 × 512 image to a 16-channel 64 × 64 latent and decodes it "
        "back; FLUX packs 2 × 2 latent patches into 64-channel tokens.\n"
        "- **Guidance distillation:** schnell was trained to produce in one to four steps without classifier-free guidance, so "
        "there is no negative prompt and the guidance scale is 0.\n"
        "- **NF4 / double quantisation:** a 4-bit number format for weights (NormalFloat), with the quantisation constants "
        "themselves quantised; the transformer shrinks from 23.8 GB to about 6.6 GB.\n"
        "- **LoRA / QLoRA:** low-rank matrices A and B added beside frozen weight matrices; only A and B are trained. QLoRA is "
        "LoRA trained over a quantised base.\n"
        "- **CLIP prompt similarity:** the cosine (× 100) between a CLIP image embedding and the CLIP text embedding of its prompt.\n"
        "- **Zero-shot label accuracy:** the share of images whose nearest caption, by CLIP cosine among the dataset's captions, "
        "is their own (an argmax, no threshold).\n"
        "- **Reference similarity:** the cosine (× 100) between an image and the mean embedding of the held-out real photographs of its caption.\n"
        "- **Leave-one-out reference:** the same readings on the real photographs, each compared with the others of its caption "
        "and never with itself; a reference line, not an upper bound.\n"
        "- **Held-out / paired comparison:** photographs never used for training or epoch selection, scored before and after "
        "with identical latents, noise and noise levels.\n"
        "- **Adapter / safetensors / reload parity:** the saved trained tensors / a code-free tensor file format / the check that "
        "base + adapter reproduces the in-memory model's loss and image.\n"
        "- **Isolated environment:** the separate Python environment Section 1 builds from hash-locked packages, where every "
        "later cell runs.\n\n"
        "## Conclusion (your notes)\n\n"
        "Fill in from the numbers this run printed; keep each claim to what the evidence shows.\n\n"
        "- **Task:** which photographs, which captions, which split sizes?\n"
        "- **Principal result:** the test flow-matching MSE before and after, with `n` photographs and five noise levels.\n"
        "- **Instrument readings:** prompt similarity, label accuracy and reference similarity, frozen and adapted, beside "
        "the leave-one-out real-photo reference line. Which reading moved, and on how many images?\n"
        "- **Uncertainty or failure mode:** which species moved the other way? What did your Section 10 run change?\n"
        "- **Limitations:** what does this run *not* show (see Interpretation and limits)?\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/flux-schnell-generation-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/flux-schnell-generation-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weights notes (identity, mirror staging, byte-identity evidence): https://github.com/kurtvalcorza/flux-schnell-generation-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Hugging Face model repository (identity of record): https://huggingface.co/black-forest-labs/FLUX.1-schnell (revision `{MODEL_REVISION}`)\n"
        "- Ungated mirror the bytes are staged from: https://huggingface.co/unsloth/FLUX.1-schnell (revision `9df3faa7ae3b6ddf0b2b69bb78616372897cc65c`)\n"
        "- Black Forest Labs (2024). Announcing Black Forest Labs — FLUX.1: https://blackforestlabs.ai/announcing-black-forest-labs/ ; reference code: https://github.com/black-forest-labs/flux\n"
        "- Liu, X., Gong, C., Liu, Q. (2023). Flow straight and fast: Learning to generate and transfer data with rectified flow. ICLR: https://arxiv.org/abs/2209.03003\n"
        "- Sauer, A., et al. (2024). Fast high-resolution image synthesis with latent adversarial diffusion distillation: https://arxiv.org/abs/2403.12015\n"
        "- Dettmers, T., et al. (2023). QLoRA: Efficient finetuning of quantized LLMs. NeurIPS: https://arxiv.org/abs/2305.14314\n"
        "- Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. ICLR: https://arxiv.org/abs/2106.09685\n"
        "- Cherti, M., et al. (2023). Reproducible scaling laws for contrastive language-image learning. CVPR (the LAION CLIP scorer): https://arxiv.org/abs/2212.07143\n"
        "- DIMER Notebook Specification 2.2 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)\n"
    ),
}
