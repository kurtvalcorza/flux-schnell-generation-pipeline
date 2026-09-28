"""Stage 4 (lab environment): generate exactly 32 FLUX.1 [schnell] candidates per defect class from fixed seeds."""
# ruff: noqa: E501
from __future__ import annotations

import contextlib
import gc
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdi_core as core  # noqa: E402

MODEL_ID = "black-forest-labs/FLUX.1-schnell"  # identity of record (gated upstream)
MODEL_REVISION = "741f7c3ce8b383c54771c7003378a50191e9efe9"
MODEL_LICENSE = "apache-2.0"
STAGING_ID = "unsloth/FLUX.1-schnell"  # ungated, unmodified mirror that serves the bytes
STAGING_REVISION = "9df3faa7ae3b6ddf0b2b69bb78616372897cc65c"
# (path, bytes, sha256) of the 23-file diffusers snapshot, from flux-schnell-generation-pipeline's committed manifest.
FILES = (
    ("model_index.json", 536, "24946df21ff25e210486b5f6b14208983a90c9c73f8d48cfa724c0e4e03f7201"),
    ("scheduler/scheduler_config.json", 274, "b129cebacf8f851867ec5c7c4d3f4bf787e232525a53becf4df5a72278a788d5"),
    ("text_encoder/config.json", 613, "d79d5c8c6ce85112a923d621a5412886ddbbb0636210fc0f72f450582e675542"),
    ("text_encoder/model.safetensors", 246144352, "893d67a23f4693ed42cdab4cbad7fe3e727cf59609c40da28a46b5470f9ed082"),
    ("text_encoder_2/config.json", 782, "9001e5a8ae0571a362f806b87b6105dd1a15c33dca237b606d2561164109beeb"),
    ("text_encoder_2/model-00001-of-00002.safetensors", 4994582224, "ec87bffd1923e8b2774a6d240c922a41f6143081d52cf83b8fe39e9d838c893e"),
    ("text_encoder_2/model-00002-of-00002.safetensors", 4530066360, "a5640855b301fcdbceddfa90ae8066cd9414aff020552a201a255ecf2059da00"),
    ("text_encoder_2/model.safetensors.index.json", 19885, "3bacec0f0cf392399d4a385908f67dd73df99c9e9cfee669f148858ba9fbdb0a"),
    ("tokenizer/merges.txt", 524619, "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"),
    ("tokenizer/special_tokens_map.json", 588, "2cdb3b8331a60c92fc1e55a13e9fd61fd2293c5a51275fdcccd62b780052530e"),
    ("tokenizer/tokenizer_config.json", 705, "6bdcee9ccce2a16ca2b4c0c5ed00b42c50ea225f4472a8c4c1e963a2902c2881"),
    ("tokenizer/vocab.json", 1059962, "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349"),
    ("tokenizer_2/special_tokens_map.json", 2543, "7a1985a994c41886db38c719d2a3d2f40606663cc19d7c5d6a85d349320e06d2"),
    ("tokenizer_2/spiece.model", 791656, "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86"),
    ("tokenizer_2/tokenizer.json", 2424235, "f5dfec163765e18e270537fe896c49f5fad74db1525641d9b255a3008b999596"),
    ("tokenizer_2/tokenizer_config.json", 20817, "1a3d2db64215ed77854dd4208aac5f8361c1b5471cabd19c0ef1472d1a895eb0"),
    ("transformer/config.json", 321, "397cfb92299488013ec3af6142a2a877366f8d2e44efbb4f3e33479e7960d3d0"),
    ("transformer/diffusion_pytorch_model-00001-of-00003.safetensors", 9962580296, "9b633dbe87316385c5b1c262bd4b5a01e3d955170661d63dcec8a01e89c0d820"),
    ("transformer/diffusion_pytorch_model-00002-of-00003.safetensors", 9949328904, "58b4434078f0c2567ddc54e3b5cbf39626ab55fbd9d5c22956e183668f535dec"),
    ("transformer/diffusion_pytorch_model-00003-of-00003.safetensors", 3870584832, "e2cbc25471ed5186e69a9b51098300cb2f612556453e38a372c851a220ed238d"),
    ("transformer/diffusion_pytorch_model.safetensors.index.json", 120822, "783f857a5872f069e75daf4a5abe5efd6ff9ec2f37d71159767910cebfe048a6"),
    ("vae/config.json", 774, "bc1e208f414a315365fbecf426838f43b87c9d5c051219e0968a56e2644b2998"),
    ("vae/diffusion_pytorch_model.safetensors", 167666902, "f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3"),
)
SIDE = 512
STEPS = 4
GUIDANCE_SCALE = 0.0  # schnell is guidance-distilled
MAX_T5_TOKENS = 256
PRECISION = "transformer nf4 (bitsandbytes, double quantisation, float16 compute); CLIP-L and T5-XXL float16; VAE float32"
SEED_BASE = 100_000  # candidate i of defect role r uses seed SEED_BASE * r + i


def stage_and_verify(root: Path) -> dict:
    from huggingface_hub import hf_hub_download

    fetched = 0
    for rel, size, _ in FILES:
        path = root / rel
        if not path.is_file() or path.stat().st_size != size:
            print(f"  fetching {rel} ({size / 1e9:.2f} GB) from {STAGING_ID}@{STAGING_REVISION[:8]}")
            hf_hub_download(STAGING_ID, rel, revision=STAGING_REVISION, local_dir=str(root))
            fetched += size
    for rel, size, digest in FILES:
        path = root / rel
        if path.stat().st_size != size or core.sha256_file(path) != digest:
            raise core.ContractError(f"{rel} fails its pinned digest; refusing to load (no fallback model, size or precision is used)")
    print(f"verified {len(FILES)} files against their pinned SHA-256 digests")
    return {"downloaded_bytes": fetched, "files": len(FILES), "total_bytes": sum(s for _, s, _ in FILES)}


def preload_nvidia_libraries() -> None:
    """Load the pip-installed CUDA libraries globally so bitsandbytes resolves them on images whose system CUDA is older."""
    import ctypes
    import os

    for site in sys.path:
        nvidia = Path(site) / "nvidia"
        if nvidia.is_dir():
            os.environ["LD_LIBRARY_PATH"] = ":".join([str(p) for p in nvidia.glob("*/lib")] + [os.environ.get("LD_LIBRARY_PATH", "")])
            for so in sorted(nvidia.rglob("lib*.so*")):
                with contextlib.suppress(OSError):
                    ctypes.CDLL(str(so), mode=getattr(ctypes, "RTLD_GLOBAL", 0))


def main(cfg_path: str) -> None:
    cfg = core.read_json(cfg_path)
    work, out = Path(cfg["work_dir"]), Path(cfg["out_dir"])
    order = core.read_json(out / "data_manifest.json")["class_order"]
    defects = order[1:]
    prompts_path = out / "prompts.json"
    prompts = core.read_json(prompts_path)["prompts"]
    rows = core.read_split_manifest(out / "split_manifest.csv")
    real_hashes = {r["pixel_sha256"] for r in rows if r["selected"]}  # generation review may consult selected training images only
    root = work / "weights" / "flux1-schnell"
    t0 = time.time()
    staging = stage_and_verify(root)
    stage_seconds = round(time.time() - t0, 1)

    preload_nvidia_libraries()
    import torch
    from diffusers import AutoencoderKL, BitsAndBytesConfig, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel
    from PIL import Image
    from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel

    if not torch.cuda.is_available():
        raise SystemExit("FLUX.1 [schnell] is loaded 4-bit with bitsandbytes and needs a CUDA GPU")
    fp16 = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
    tokenizer_2 = AutoTokenizer.from_pretrained(str(root / "tokenizer_2"))

    # 1) Text encoders only: encode every distinct prompt, keep the embeddings on the CPU, release the encoders.
    t1 = time.time()
    clip = CLIPTextModel.from_pretrained(str(root), subfolder="text_encoder", torch_dtype=fp16).to("cuda").eval()
    t5 = T5EncoderModel.from_pretrained(str(root), subfolder="text_encoder_2", torch_dtype=fp16).to("cuda").eval()
    cache, token_counts = {}, {}
    with torch.inference_mode():
        for p in prompts:
            text = p["text"]
            clip_ids = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
            pooled = clip(clip_ids.to("cuda"), output_hidden_states=False).pooler_output
            t5_batch = tokenizer_2(text, padding="max_length", max_length=MAX_T5_TOKENS, truncation=True, return_tensors="pt")
            embeds = t5(t5_batch.input_ids.to("cuda"), output_hidden_states=False)[0]
            cache[p["prompt_id"]] = (embeds[0].to("cpu"), pooled[0].to("cpu"))
            n_clip = len(tokenizer(text).input_ids)
            n_t5 = len(tokenizer_2(text).input_ids)
            token_counts[p["prompt_id"]] = {"clip_tokens": n_clip, "clip_limit": tokenizer.model_max_length, "clip_truncated": n_clip > tokenizer.model_max_length, "t5_tokens": n_t5, "t5_limit": MAX_T5_TOKENS, "t5_truncated": n_t5 > MAX_T5_TOKENS}
    encoder_peak = torch.cuda.max_memory_allocated() / 2**30
    encode_seconds = round(time.time() - t1, 1)
    del clip, t5
    gc.collect()
    torch.cuda.empty_cache()
    print(f"encoded {len(cache)} prompts in {encode_seconds} s (peak {encoder_peak:.2f} GiB); encoders released, {torch.cuda.memory_allocated() / 2**30:.2f} GiB still allocated")

    # 2) 4-bit transformer + float32 VAE; generation from the cached embeddings.
    torch.cuda.reset_peak_memory_stats()
    t2 = time.time()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=fp16)
    transformer = FluxTransformer2DModel.from_pretrained(str(root), subfolder="transformer", quantization_config=quant, torch_dtype=fp16).to("cuda").eval()
    vae = AutoencoderKL.from_pretrained(str(root), subfolder="vae", torch_dtype=torch.float32).to("cuda").eval()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(root), subfolder="scheduler")
    pipe = FluxPipeline(scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=tokenizer, text_encoder_2=None, tokenizer_2=tokenizer_2, transformer=transformer)
    pipe.set_progress_bar_config(disable=True)
    load_seconds = round(time.time() - t2, 1)
    print(f"4-bit transformer loaded in {load_seconds} s; {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated")

    gen_dir = out / "generated"
    shutil.rmtree(gen_dir, ignore_errors=True)
    manifest, seen = [], set()
    t3 = time.time()
    for role, label in enumerate(defects, start=1):
        class_prompts = [p for p in prompts if p["label"] == label]
        (gen_dir / label).mkdir(parents=True, exist_ok=True)
        for i in range(core.GENERATION_CANDIDATES):
            p = class_prompts[i % len(class_prompts)]
            seed = SEED_BASE * role + i
            cand_id = f"syn_{label}_{i:02d}"
            row = {"candidate_id": cand_id, "intended_label": label, "label_status": "intended by prompt; not an independently validated annotation", "prompt_id": p["prompt_id"], "prompt": p["text"], "phi4_guided": p["phi4_guided"], "source_image_ids": p["source_image_ids"], "seed": seed, "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "staging": f"{STAGING_ID}@{STAGING_REVISION}", "precision": PRECISION, "width": SIDE, "height": SIDE, "steps": STEPS, "guidance_scale": GUIDANCE_SCALE, "scheduler": "FlowMatchEulerDiscreteScheduler (upstream config)", "review_status": "not reviewed (canonical run)"}
            t = time.time()
            pixels = None
            try:
                embeds, pooled = cache[p["prompt_id"]]
                generator = torch.Generator(device="cpu").manual_seed(seed)
                with torch.inference_mode():
                    latents = pipe(prompt_embeds=embeds[None].to("cuda", fp16), pooled_prompt_embeds=pooled[None].to("cuda", fp16), num_inference_steps=STEPS, guidance_scale=GUIDANCE_SCALE, height=SIDE, width=SIDE, max_sequence_length=MAX_T5_TOKENS, generator=generator, output_type="latent").images
                    latents = FluxPipeline._unpack_latents(latents, SIDE, SIDE, pipe.vae_scale_factor).float()
                    latents = latents / vae.config.scaling_factor + vae.config.shift_factor
                    decoded = vae.decode(latents, return_dict=False)[0][0]
                pixels = decoded.permute(1, 2, 0).float().cpu().numpy()
            except Exception as exc:  # a failed attempt is recorded, never silently retried
                row["error"] = f"{type(exc).__name__}: {exc}"[:200]
            status, reason, digest = core.candidate_eligibility(pixels, side=SIDE, seen=seen, real=real_hashes)
            row.update(status=status, reason=reason, pixel_sha256=digest, seconds=round(time.time() - t, 2))
            if pixels is not None and np.isfinite(pixels).all():
                path = gen_dir / label / f"{cand_id}.png"
                Image.fromarray(core.to_uint8(pixels)).save(path)
                row.update(file=str(path.relative_to(out)), png_sha256=core.sha256_file(path))
            if status == "eligible":
                seen.add(digest)
            manifest.append(row)
            print(f"{cand_id} seed {seed} {status}{(' (' + reason + ')') if reason else ''} {row['seconds']} s")
    generate_seconds = round(time.time() - t3, 1)
    transformer_peak = torch.cuda.max_memory_allocated() / 2**30
    core.write_jsonl(out / "generation_manifest.jsonl", manifest)
    status = core.generation_status(manifest, defects)
    core.write_json(out / "generation_status.json", status)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for label in defects:
        items = [(Image.open(out / r["file"]), f"{r['candidate_id'][-2:]} {r['status']}\nintended: {label}") for r in manifest if r["intended_label"] == label and r.get("file")]
        core.contact_sheet(items, title=f"FLUX candidates, intended label '{label}' (prompt label, not a verified annotation)").save(figures / f"generated_{label}.png")
    with open(out / "human_review_template.csv", "w", encoding="utf-8") as handle:
        handle.write("candidate_id,intended_label,decision,reason\n")
        for r in manifest:
            handle.write(f"{r['candidate_id']},{r['intended_label']},,\n")
    summary = {"model": {"id": MODEL_ID, "revision": MODEL_REVISION, "license": MODEL_LICENSE, "staging": {"repo": STAGING_ID, "revision": STAGING_REVISION}}, "staging": staging, "stage_seconds": stage_seconds, "encode_seconds": encode_seconds, "encoder_peak_gpu_gib": round(encoder_peak, 2), "load_seconds": load_seconds, "generate_seconds": generate_seconds, "transformer_peak_gpu_gib": round(transformer_peak, 2), "token_counts": token_counts, "prompts_sha256": core.sha256_file(prompts_path), "generation_status": status, "precision": PRECISION, "steps": STEPS, "side": SIDE}
    del pipe, transformer, vae
    gc.collect()
    torch.cuda.empty_cache()
    if cfg["delete_model_weights_after_use"]:
        shutil.rmtree(root, ignore_errors=True)
        summary["weights_deleted_after_use"] = True
    core.write_json(work / "state" / "flux_stage.json", summary)
    print(core.canonical_json({"usable": status["usable"], "complete": status["complete"], "generate_seconds": generate_seconds, "encoder_peak_gpu_gib": summary["encoder_peak_gpu_gib"], "transformer_peak_gpu_gib": summary["transformer_peak_gpu_gib"]}))


if __name__ == "__main__":
    core.run_main(main, sys.argv[1])
