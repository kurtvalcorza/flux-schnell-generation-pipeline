"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

This template configures an E2E text-to-image fine-tuning workflow: the pinned FLUX.1 [schnell] snapshot
(diffusers layout, staged from an ungated mirror and digest-verified against the committed manifest) and the CLIP
scorer are staged and verified, 60 pinned CC0 iNaturalist bird photographs are fetched, validated and split, every
prompt is encoded once with CLIP-L and T5-XXL and the encoders are released, the 12 B transformer is loaded 4-bit,
the frozen model is scored (held-out flow-matching loss, CLIP-scored four-step generations) against the real-photo
ceiling, a bounded QLoRA fine-tuning runs in the kernel, the held-out scores are read again in a paired comparison,
a new prompt is rendered, and the adapter is exported and reloaded into a fresh 4-bit pipeline.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "flux-schnell-generation-pipeline"

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

TEMPLATE = {
    "package": "flux_schnell_generation_pipeline",
    "repo_name": REPO,
    "stem": "flux_schnell_generation",
    "notebook_name": "flux_schnell_generation_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh **GPU** runtime (a 16 GB T4 is enough; see the Prerequisites) installs the pinned "
        "dependencies (torch, diffusers, transformers, peft, bitsandbytes, accelerate, sentencepiece, safetensors, huggingface-hub, "
        "numpy, pillow), stages and digest-verifies two pinned snapshots — the 33.7 GB FLUX.1 [schnell] diffusers-layout snapshot "
        "(transformer, CLIP-L and T5-XXL encoders, tokenizers, VAE, scheduler), fetched from an ungated mirror and checked byte for "
        "byte against the manifest the repository committed, and a 0.6 GB CLIP scorer — fetches 60 CC0 iNaturalist bird photographs "
        "as digest-verified JPEGs (6 MB, no credential), validates them and splits them 36 / 12 / 12 by seed, encodes every prompt "
        "with the two text encoders and releases them, loads the 12 B transformer 4-bit (NF4) with an untrained LoRA adapter "
        "attached, scores the frozen model — the held-out flow-matching loss on the validation and test photographs, and six "
        "four-step generations scored by CLIP against their prompts, the held-out photographs and the real-photo ceiling — runs a "
        "bounded QLoRA fine-tuning (3 epochs over 36 images), scores the adapted model on identical inputs, renders a new prompt, "
        "exports the adapter as safetensors with a manifest, releases the adapted transformer and reloads the artifact into a fresh "
        "4-bit pipeline to verify parity. The default path needs no repository clone, no DIMER worker or service, no credential, "
        "no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5). On a T4 the whole path takes about an hour of model "
        "time after the 34 GB of downloads; the timings recorded on the release run are in `docs/release-verification.md`."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "captioned photographs as a zip holding `captions.csv` (columns `id`, `file`, `caption`) beside the image files (JPEG or "
        "PNG, shorter side 256..4096 px; at least four images, and at least one caption with three or more images so a held-out "
        "record exists). Your records are split by caption into training, validation and test sets and flow through the same "
        "contract — validation, prompt encoding, frozen baseline, adaptation, held-out evaluation, generation, artifact export "
        "and reload parity. Because the text encoders and the transformer do not share the GPU, re-running from Section 4 releases "
        "the transformer before your prompts are encoded (Section 5 does this). The expected schema, the ceilings and the privacy "
        "guidance are stated in the Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is optional "
        "and never part of the default path."
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
        "and the same CLIP scores on the real photographs themselves — the ceiling. None of these is a human judgement of image quality."
    ),
    "learning_objectives": (
        "install the pinned runtime; inspect the carried pipeline, dataset and scorer modules; stage and digest-verify a pinned snapshot "
        "served by a mirror against a committed manifest; fetch, validate and split a small real captioned-photograph dataset; encode "
        "prompts with two text encoders and release them; load a 12 B transformer 4-bit; read a held-out flow-matching loss and "
        "CLIP-scored few-step generations against a real-photo ceiling; run a bounded QLoRA fine-tuning of a rectified-flow transformer "
        "with explicit hyperparameters; compare the adapted and frozen models on identical held-out inputs; render a new prompt; and "
        "export a safetensors adapter that reloads against the pinned 4-bit base with verified parity."
    ),
    "exclusions": (
        "FLUX.1 [dev] or [pro], guidance-distillation or classifier-free guidance (schnell has none), ControlNet, inpainting or "
        "image-to-image conditioning, resolutions other than 512², full or 16-bit fine-tuning, DreamBooth identifiers, safety filtering "
        "of prompts or images, human preference or FID/KID benchmarks, prompt engineering, and any claim that a CLIP score or a "
        "flow-matching loss measures image quality. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported **GPU** runtime (Google Colab T4 or better, or a Jupyter kernel with a CUDA GPU of at least 15 GB, bitsandbytes support and Python 3.12). The CLIP-L and T5-XXL encoders run in float16 (9.7 GB) while they encode prompts and are then released; the transformer is loaded 4-bit (NF4, about 6.6 GB) with float16 compute and the LoRA parameters in float32; the FLUX VAE stays in float32 (0.34 GB). CPU-only runtimes are not supported for this notebook: bitsandbytes 4-bit needs CUDA and the 12 B transformer would need 48 GB of RAM in float32. About 36 GB of disk is needed for the two snapshots.",
        "- **Knowledge:** what a latent diffusion or flow-matching model does at inference (noise → latent → image), why a distilled few-step model uses no classifier-free guidance, what 4-bit weight quantisation changes, what a LoRA adapter changes and what it does not, and why a training loss is not a quality score.",
        "- **Weights:** the transformer, both text encoders, the VAE and the CLIP scorer are all safetensors; nothing is unpickled and no Hub-hosted code is executed — the model classes come from `diffusers`, `transformers`, `peft` and `bitsandbytes` on PyPI. FLUX.1 [schnell] is released under the Apache-2.0 licence; the scorer is MIT. The upstream Hub repository is gated by a click-through, so the bytes are fetched from an ungated, unmodified mirror and verified against the manifest committed in this repository (Section 3).",
        "- **Data contract:** a record is `{{id, image, caption}}` — an RGB image with shorter side 256..4096 px (resized so the shorter side is 512 px and centre-cropped to 512 × 512; the crop is reported) and a caption of 1..1000 characters (truncated to 256 T5 tokens and 77 CLIP tokens; T5 truncations are reported). Validation is structural: nothing checks that a caption describes its image or that the model can render it.",
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
                "photograph. `build_sample_dataset` draws a seeded stratified split — 6 / 2 / 2 per species for training, "
                "validation and test — and `dataset_manifest` validates every split, checks that no image appears twice and "
                "records a digest.\n\n"
                "Look for: 36 / 12 / 12 records, six distinct captions, a shorter side around 300..500 px (every photo is "
                "centre-cropped to 512²), a written `outputs/{stem}_sample_captions.csv` in the shape BYOD expects, and "
                "three refusal probes — a missing caption, a 200 px image, a duplicate id — each rejected before the model runs."
            ),
            "code": (
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "import numpy as np\n"
                "from PIL import Image\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    splits = split_dataset(load_byod_dataset(byod_path), seed=0)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "else:\n"
                "    splits = fetch_sample_dataset(cache_dir='weights/inat-birds')\n"
                "    data_source = SAMPLE_LABEL_SOURCE\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n\n"
                "dataset_report = dataset_manifest({{'train': train_records, 'validation': val_records, 'test': test_records}})\n"
                "print({{'data_source': data_source, 'splits': {{k: v['n_records'] for k, v in dataset_report['splits'].items()}}, 'captions': dataset_report['splits']['train']['n_captions'], 'disjoint': dataset_report['disjoint']}})\n"
                "print({{'shorter_side': dataset_report['splits']['train']['shorter_side'], 'centre_cropped': dataset_report['splits']['train']['centre_cropped'], 'digest': dataset_report['digest'][:16] + '...'}})\n"
                "print({{'first_test_record': validate_inputs(test_records[0]), 'caption': test_records[0]['caption']}})\n"
                "prompts = sample_prompts(train_records)\n"
                "print({{'prompts': prompts}})\n"
                "sample_csv = write_dataset_csv(test_records, 'outputs/{stem}_sample_captions.csv')\n"
                "print({{'sample_csv': str(sample_csv)}})\n\n"
                "print({{'validation': INPUT_SCHEMA['validation']}})\n"
                "probes = {{\n"
                "    'missing caption': [{{'id': r['id'], 'image': r['image']}} for r in train_records[:4]],\n"
                "    'image too small': [{{**train_records[0], 'image': Image.new('RGB', (200, 200))}}, *train_records[1:4]],\n"
                "    'duplicate id': [train_records[0], *train_records[:4]],\n"
                "}}\n"
                "for name, records in probes.items():\n"
                "    try:\n"
                "        validate_dataset(records)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Encode every prompt, then release the text encoders\n\n"
                "`pipe.encode_prompts` loads CLIP-L and T5-XXL from the verified snapshot (float16 on CUDA), tokenises each distinct "
                "prompt — 77 CLIP tokens for the pooled vector, 256 T5 tokens for the sequence — runs both encoders once per prompt and "
                "keeps the (256, 4096) embeddings and the (768,) pooled vector on the CPU. Encoded here: the six training captions (which "
                "are also the validation and test captions and the generation prompts) and one new prompt for Section 9. schnell is "
                "guidance-distilled, so there is no negative prompt to encode. `release_text_encoder` then drops the 9.7 GB of encoders so "
                "the 4-bit transformer, the VAE, the scorer and a training graph fit on a 16 GB GPU. If the transformer is already "
                "resident (a BYOD re-run), it is released first — the two never share the GPU.\n\n"
                "Look for: the encoders loading in about a minute from the bfloat16 shards, seven prompts encoded in seconds, no "
                "truncation, finite embeddings, and the GPU memory falling back to nearly zero after the release."
            ),
            "code": (
                "import time\n\n"
                "NEW_PROMPT = 'a photo of a House Finch (Haemorhous mexicanus) perched on a snow-covered branch in winter'\n\n"
                "def gpu_memory_gb():\n"
                "    return round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None\n\n"
                "if pipe.transformer is not None:\n"
                "    print({{'transformer_released_for_encoding': pipe.release_transformer()}})\n"
                "all_prompts = sample_prompts(train_records + val_records + test_records) + [NEW_PROMPT]\n"
                "encode_report = pipe.encode_prompts(all_prompts)\n"
                "print({{**encode_report, 'gpu_memory_gb_with_encoders': gpu_memory_gb()}})\n"
                "cache = pipe.export_prompt_cache()\n"
                "print({{'embeddings_finite': all(bool(torch.isfinite(v['embeds']).all()) and bool(torch.isfinite(v['pooled']).all()) for v in cache.values()), 'embeds_shape': tuple(next(iter(cache.values()))['embeds'].shape), 'pooled_shape': tuple(next(iter(cache.values()))['pooled'].shape)}})\n"
                "released = pipe.release_text_encoder()\n"
                "print({{'text_encoders_released': released, 'gpu_memory_gb_after_release': gpu_memory_gb(), 'device': pipe.device, 'compute_dtype': pipe.compute_dtype}})"
            ),
        },
        {
            "md": (
                "## 6. Load the 4-bit transformer; the frozen model's held-out loss and CLIP-scored generations\n\n"
                "`pipe.load_transformer` reads the three bfloat16 shards (23.8 GB) and quantises every linear projection of the 57 "
                "blocks to 4-bit NF4 with double quantisation as it loads — about 6.6 GB on the GPU with float16 compute — checks the "
                "parameter count against the checkpoint (11,891,178,560), freezes everything and attaches the untrained LoRA adapter "
                "(`use_lora=True`; its B matrices start at zero, so until Section 7 this is the pretrained model). Two kinds of number "
                "are read here and kept for the comparison.\n\n"
                "**Held-out flow-matching loss** (`pipe.evaluate`): each held-out photograph is VAE-encoded and packed, mixed with a "
                "seeded noise tensor at five fixed noise levels σ ∈ {{0.1, 0.3, 0.5, 0.7, 0.9}} as x_σ = (1 − σ)·x₀ + σ·ε, and the "
                "transformer's velocity prediction is scored against the rectified-flow target ε − x₀ (MSE over the tokens). It is the "
                "training objective measured on photographs the model never trains on; the same seed gives the same latents, noise and "
                "noise levels later, so the adapted number is a paired comparison, not a re-draw.\n\n"
                "**CLIP-scored generations** (`pipe.generate` + `score_generations`): one image per training caption at fixed seeds "
                "(4 Euler steps, no guidance, 512²), scored by the frozen CLIP ViT-B/32 on prompt alignment (cosine × 100), zero-shot "
                "label accuracy (which of the six captions is nearest — an argmax with no threshold) and similarity to the mean embedding "
                "of the held-out real photographs of that species. `real_photo_baseline` scores the real test photographs the same way: "
                "the ceiling these numbers could reach. Look for: the load taking a few minutes, a flow-matching MSE around 0.5..1.5, "
                "label accuracy below the real photographs' 1.0 for at least some species, and a first grid of six generated birds."
            ),
            "code": (
                "STEPS = 4  # @param {{type:\"integer\"}}\n"
                "IMAGES_PER_PROMPT = 1  # @param {{type:\"integer\"}}\n"
                "EVAL_SEED = 0\n\n"
                "def grid(images, path, columns=6):\n"
                "    tiles = [im.resize((256, 256)) for im in images]\n"
                "    rows = (len(tiles) + columns - 1) // columns\n"
                "    sheet = Image.new('RGB', (256 * columns, 256 * rows), 'white')\n"
                "    for i, tile in enumerate(tiles):\n"
                "        sheet.paste(tile, (256 * (i % columns), 256 * (i // columns)))\n"
                "    sheet.save(path)\n"
                "    return path\n\n"
                "load_report = pipe.load_transformer()\n"
                "print({{**load_report, 'quantization': QUANTIZATION, 'parameters': count_parameters(pipe.transformer), 'lora_tensors': len(lora_parameter_names(pipe.transformer)), 'gpu_memory_gb': gpu_memory_gb()}})\n"
                "scorer = ClipScorer(weights_dir=SCORER_WEIGHTS_DIR, device=pipe.device)\n"
                "t0 = time.perf_counter()\n"
                "frozen_val = pipe.evaluate(val_records, seed=EVAL_SEED)\n"
                "frozen_test = pipe.evaluate(test_records, seed=EVAL_SEED)\n"
                "print({{'frozen_flow_matching_mse': {{'validation': frozen_val['flow_matching_mse'], 'test': frozen_test['flow_matching_mse']}}, 'by_sigma_test': frozen_test['by_sigma'], 'seconds': round(time.perf_counter() - t0, 1)}})\n\n"
                "generation_prompts = [p for p in prompts for _ in range(IMAGES_PER_PROMPT)]\n"
                "frozen_generation = pipe.generate(generation_prompts, seed=1000, steps=STEPS)\n"
                "print({{'generated': len(frozen_generation['images']), 'steps': frozen_generation['steps'], 'guidance_scale': frozen_generation['guidance_scale'], 'precision': frozen_generation['precision'], 'seconds': frozen_generation['seconds'], 'adapted': frozen_generation['model']['adapted']}})\n"
                "frozen_scores = score_generations(scorer, frozen_generation['images'], references=test_records)\n"
                "real_ceiling = real_photo_baseline(scorer, test_records)\n"
                "print({{'frozen_generations': {{k: frozen_scores[k] for k in ('clip_prompt_similarity', 'label_accuracy', 'reference_similarity')}}}})\n"
                "print({{'real_photo_ceiling': {{k: real_ceiling[k] for k in ('clip_prompt_similarity', 'label_accuracy', 'reference_similarity')}}}})\n"
                "for entry in frozen_scores['per_image'][::IMAGES_PER_PROMPT]:\n"
                "    print({{'prompt': entry['prompt'][:42], 'clip': entry['clip_prompt_similarity'], 'nearest': entry['nearest_prompt'][13:40], 'correct': entry['correct'], 'reference_similarity': entry['reference_similarity']}})\n"
                "print({{'grid': str(grid([g['image'] for g in frozen_generation['images']], 'outputs/{stem}_frozen_grid.jpg')), 'gpu_memory_gb': gpu_memory_gb()}})"
            ),
        },
        {
            "md": (
                "## 7. Bounded QLoRA fine-tuning\n\n"
                "`pipe.adapt` trains the 380 LoRA tensors (rank 8, 9,338,880 parameters — 0.08 % of the transformer) that `peft` "
                "attached to the image-stream query, key, value and output projections of all 57 blocks, and nothing else; the 4-bit "
                "base, the VAE and the text encoders are frozen. Each step takes one training photograph's packed latent (VAE-encoded "
                "once, seeded), draws a noise level σ uniformly from (0, 1) and a noise tensor (both seeded), mixes x_σ = (1 − σ)·x₀ + "
                "σ·ε, and minimises the MSE between the predicted velocity and ε − x₀; AdamW at a fixed learning rate on float32 LoRA "
                "masters, gradient-norm clipping at 1.0, float16 autocast with loss scaling and gradient checkpointing (the 1,280-token "
                "activations of 57 blocks would not otherwise fit). Epoch 0 records the frozen model's validation loss, and the epoch "
                "with the lowest validation flow-matching loss is kept.\n\n"
                "Watch the validation loss from epoch 0; three epochs over 36 images (108 steps) take on the order of half an hour on "
                "a T4, most of it the per-epoch validation pass. The training loss is a noisy per-step average over random noise "
                "levels and is not the quality signal — the paired held-out numbers in Section 8 are."
            ),
            "code": (
                "EPOCHS = 3  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-4  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 1  # @param {{type:\"integer\"}}\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4), 'val_flow_matching_mse': entry['val_loss']}}\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, seed=EVAL_SEED, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'steps': adapt_result['n_steps'], 'best_epoch': adapt_result['best_epoch'], 'precision': adapt_result['precision'], 'seconds': adapt_seconds, 'peak_gpu_memory_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2)}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation: the paired comparison\n\n"
                "The test photographs were never used for training or epoch selection. The adapted model is scored exactly as "
                "the frozen model was in Section 6 — the same seed, so the same latents, noise and noise levels, and the same six "
                "prompt/seed pairs for generation — and the table puts the frozen, the adapted and the real-photo numbers side by "
                "side. The cell asserts only what the procedure guarantees — the kept epoch's validation loss is no higher than the "
                "frozen model's (epoch 0) and the re-scored validation loss matches the history — and prints the test comparison "
                "without a directional assertion: a lower test flow-matching MSE is what to look for, not what is promised. Look too "
                "for the generated birds moving towards the held-out photographs: higher reference similarity and label accuracy, "
                "and a second grid to compare with the first by eye. Six images per model from one seeded run give no dispersion "
                "estimate; these are sample-sanity numbers that show the adaptation contract works, not a benchmark, and CLIP "
                "agreement is not a human judgement of quality."
            ),
            "code": (
                "adapted_val = pipe.evaluate(val_records, seed=EVAL_SEED)\n"
                "adapted_test = pipe.evaluate(test_records, seed=EVAL_SEED)\n"
                "adapted_generation = pipe.generate(generation_prompts, seed=1000, steps=STEPS)\n"
                "adapted_scores = score_generations(scorer, adapted_generation['images'], references=test_records)\n"
                "comparison = {{\n"
                "    'flow_matching_mse_validation': {{'frozen': frozen_val['flow_matching_mse'], 'adapted': adapted_val['flow_matching_mse']}},\n"
                "    'flow_matching_mse_test': {{'frozen': frozen_test['flow_matching_mse'], 'adapted': adapted_test['flow_matching_mse']}},\n"
                "    'flow_matching_mse_test_by_sigma': {{s: {{'frozen': frozen_test['by_sigma'][s], 'adapted': adapted_test['by_sigma'][s]}} for s in adapted_test['by_sigma']}},\n"
                "    'clip_prompt_similarity': {{'frozen': frozen_scores['clip_prompt_similarity'], 'adapted': adapted_scores['clip_prompt_similarity'], 'real_photos': real_ceiling['clip_prompt_similarity']}},\n"
                "    'label_accuracy': {{'frozen': frozen_scores['label_accuracy'], 'adapted': adapted_scores['label_accuracy'], 'real_photos': real_ceiling['label_accuracy']}},\n"
                "    'reference_similarity': {{'frozen': frozen_scores['reference_similarity'], 'adapted': adapted_scores['reference_similarity'], 'real_photos': real_ceiling['reference_similarity']}},\n"
                "}}\n"
                "for name, row in comparison.items():\n"
                "    print({{name: row}})\n"
                "for before, after in zip(frozen_scores['per_image'][::IMAGES_PER_PROMPT], adapted_scores['per_image'][::IMAGES_PER_PROMPT]):\n"
                "    print({{'prompt': before['prompt'][:42], 'reference_similarity': {{'frozen': before['reference_similarity'], 'adapted': after['reference_similarity']}}, 'correct': {{'frozen': before['correct'], 'adapted': after['correct']}}}})\n"
                "print({{'grid': str(grid([g['image'] for g in adapted_generation['images']], 'outputs/{stem}_adapted_grid.jpg'))}})\n"
                "evaluation_report = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY, 'staging': {{'repo': STAGING_ID, 'revision': STAGING_REVISION}}, 'quantization': QUANTIZATION, 'compute_dtype': pipe.compute_dtype}},\n"
                "    'scorer': frozen_scores['scorer'],\n"
                "    'data_source': data_source,\n"
                "    'dataset': dataset_report,\n"
                "    'generation': {{'steps': STEPS, 'guidance_scale': GUIDANCE_SCALE, 'images_per_prompt': IMAGES_PER_PROMPT, 'seed': 1000}},\n"
                "    'frozen': {{'validation': frozen_val, 'test': frozen_test, 'generations': frozen_scores}},\n"
                "    'adapted': {{'validation': adapted_val, 'test': adapted_test, 'generations': adapted_scores}},\n"
                "    'real_photo_ceiling': real_ceiling,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report, f, indent=2)\n"
                "best = adapt_result['history'][adapt_result['best_epoch']]\n"
                "assert best['val_loss'] <= adapt_result['history'][0]['val_loss']\n"
                "assert abs(adapted_val['flow_matching_mse'] - best['val_loss']) < 1e-4\n"
                "print({{'test_flow_matching_mse_change': round(adapted_test['flow_matching_mse'] - frozen_test['flow_matching_mse'], 6), 'note': 'held-out observation, not asserted'}})\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. A new prompt, artifact export and fresh reload\n\n"
                "The adapted model renders `NEW_PROMPT` — a composition that appears in no training caption — at two seeds; the "
                "CLIP prompt similarity is printed as a sanity check, not an evaluation. One more image of the first training prompt "
                "is rendered and kept for the parity check.\n\n"
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
                "same kernels, same deterministic NF4 quantisation of the same bytes)."
            ),
            "code": (
                "import platform\n"
                "import shutil\n\n"
                "new_generation = pipe.generate([NEW_PROMPT, NEW_PROMPT], seed=2000, steps=STEPS)\n"
                "new_scores = score_generations(scorer, new_generation['images'])\n"
                "print({{'new_prompt': NEW_PROMPT, 'clip_prompt_similarity': new_scores['clip_prompt_similarity'], 'seconds': new_generation['seconds'], 'note': 'sanity check, not an evaluation'}})\n"
                "for i, entry in enumerate(new_generation['images']):\n"
                "    entry['image'].save(f'outputs/{stem}_new_prompt_{{i}}.png')\n"
                "before = pipe.generate([prompts[0]], seed=3000, steps=STEPS)['images'][0]['image']\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...', 'quantization': artifact_manifest['base_model']['quantization']}})\n\n"
                "print({{'adapted_transformer_released': pipe.release_transformer(), 'gpu_memory_gb': gpu_memory_gb()}})\n"
                "reloaded = FluxSchnellPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device, prompt_cache=cache)\n"
                "reloaded_test = reloaded.evaluate(test_records, seed=EVAL_SEED)\n"
                "after = reloaded.generate([prompts[0]], seed=3000, steps=STEPS)['images'][0]['image']\n"
                "parity = {{'flow_matching_mse_diff': round(abs(reloaded_test['flow_matching_mse'] - adapted_test['flow_matching_mse']), 8), 'mean_abs_pixel_diff': round(float(np.abs(np.asarray(before, dtype=np.float32) - np.asarray(after, dtype=np.float32)).mean()), 4)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch'], 'gpu_memory_gb': gpu_memory_gb()}})\n"
                "assert parity['flow_matching_mse_diff'] < 1e-5 and parity['mean_abs_pixel_diff'] < 1.0\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model': {{**evaluation_report['model'], 'model_license': MODEL_LICENSE, 'device': pipe.device, 'precision': frozen_generation['precision'], 'source': pipe.source}},\n"
                "    'scorer': {{**evaluation_report['scorer'], 'license': SCORER_LICENSE}},\n"
                "    'provenance': {{\n"
                "        'snapshots': {{'model': len(MANIFEST['files']), 'scorer': len(SCORER_MANIFEST['files'])}},\n"
                "        'staging': MANIFEST['staging'],\n"
                "        'safetensors_only': True,\n"
                "        'remote_code_executed': False,\n"
                "        'text_encoders_released_before_training': released,\n"
                "        'transformer_loaded_after_encoding': load_report,\n"
                "        'data_base_url': CORPUS_BASE_URL,\n"
                "        'data_license': CORPUS_LICENSE,\n"
                "    }},\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'diffusers': diffusers.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__, 'bitsandbytes': bitsandbytes.__version__, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}},\n"
                "    'data_source': data_source,\n"
                "    'comparison': comparison,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes']}},\n"
                "    'reload_parity': parity,\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_payload, f, indent=2)\n\n"
                "print('outputs/:')\n"
                "for path in sorted(Path('outputs').rglob('*')):\n"
                "    if path.is_file():\n"
                "        print(f'  - {{path.as_posix()}} ({{path.stat().st_size / 1024:.1f}} KB)')"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "A LoRA of nine million parameters trained for half an hour on 36 photographs over a 4-bit 12 B base lowers the held-out "
        "flow-matching loss on twelve photographs the model never saw and moves its four-step generations towards the held-out real "
        "photographs of the same species. That is the claim: the adaptation contract teaches a rectified-flow transformer a narrow "
        "visual domain from a handful of captioned images on a 16 GB GPU, the held-out objective is measured on identical inputs "
        "before and after, and the artifact that carries the change is 37 MB.\n\n"
        "The numbers are sample-sanity evidence. A flow-matching loss is the training objective, not a quality score; CLIP similarity "
        "and CLIP's nearest-caption vote are a frozen model's opinion, not a human judgement, and CLIP itself has biases about what a "
        "species name looks like; six images per model from one seeded run give no dispersion estimate; and nothing here measures "
        "aesthetics, diversity, artefacts or prompt fidelity beyond the six captions. Every number is the **4-bit** model's: NF4 "
        "quantisation changes the base's outputs relative to the bfloat16 checkpoint by an amount this notebook does not measure, and "
        "an adapter trained over the quantised base is meant to be served over it (the artifact manifest records `nf4`). Fine-tuning "
        "on a narrow domain can also erode the model elsewhere — the new prompt in Section 9 is a sanity check on one composition, "
        "not a test of generality.\n\n"
        "Three things to carry to real data. **Captions are the contract:** the adapter learns the association between the "
        "caption text and the images; a caption that does not describe its image, or one caption for very different images, "
        "teaches noise. **Hold out by caption, not by image:** the split keeps every caption's images across sets so the "
        "held-out loss measures generalisation within the domain; a caption with one image cannot be evaluated. **Licences "
        "travel with the outputs:** FLUX.1 [schnell] is Apache-2.0 and the training photographs here are CC0 — with your own "
        "data, the rights to the images and to what the adapter produces are yours to establish.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can stage a pinned safetensors snapshot from a mirror and digest-verify it against a committed manifest, fetch and "
        "validate digest-pinned real photographs, encode prompts and release the encoders, load a 12 B transformer 4-bit, execute "
        "bounded QLoRA fine-tuning, evaluate the frozen and the adapted model on identical held-out inputs with a real-photo ceiling, "
        "and emit the shown machine-readable artifacts — without the repository being reachable. It does **not** establish benchmark "
        "superiority, production fitness, or image quality beyond the checks shown.\n\n"
        "**Optional experiments (they do not affect the default path):** raise `EPOCHS` or `LEARNING_RATE` and watch the "
        "validation loss for the epoch where it turns; set `STEPS = 1` or `2` and read how few-step distillation trades prompt "
        "similarity for speed; set `IMAGES_PER_PROMPT = 2` for a less noisy label accuracy; or bring your own captioned "
        "photographs through BYOD and compare the real-photo ceiling with the adapted numbers.\n\n"
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
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)\n"
    ),
}
