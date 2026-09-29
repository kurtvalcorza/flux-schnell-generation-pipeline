"""Stage 2 (phi4 environment): describe the selected training exemplars with Phi-4-multimodal-instruct (frozen, NF4)."""
# ruff: noqa: E501
from __future__ import annotations

import gc
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdi_core as core  # noqa: E402

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
MODEL_REVISION = "93f923e1a7727d1c4f446756212d9d3e8fcc5d81"  # immutable commit; the same pin as phi4-multimodal-pipeline
MODEL_LICENSE = "MIT"
# (path, bytes, sha256) for every file the image path needs, from the pipeline's committed snapshot manifest.
FILES = (
    ("LICENSE", 1141, "c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383"),
    ("added_tokens.json", 249, "d4f2aceb0f20b71dd1f4bcc7e052e4412946bf281840b8f83d39f259571af486"),
    ("config.json", 4631, "49e1c05f93d43d7f17715b779a2576235b019f587285d7d914e5b05156253f62"),
    ("configuration_phi4mm.py", 11014, "bd9609bd47ba0c87788011e5158a8bd3e1e93165a82a9b764eb7cb048006c949"),
    ("generation_config.json", 190, "757daa0d0e89171fe48fc3286341833e95b90bdb7dd3b02a2f8920fb09f85a38"),
    ("merges.txt", 2418348, "856ce61180bb689282eed6b3a6838bb1f438399be23aefe9d20eb379791fb4ad"),
    ("model-00001-of-00003.safetensors", 4997504848, "c46bb03332d82f6a3eaf85bd20af388dd4d4d68b198c2203c965c7381a466094"),
    ("model-00002-of-00003.safetensors", 4952333128, "b3e812c0c8acef4e7f5e34d6c9f77a7640ee4a2b93ea351921365ac62f19918d"),
    ("model-00003-of-00003.safetensors", 1199389232, "7be96b7339303752634b202d3f377bcf312a03046586eca6cea23347ace1e65a"),
    ("model.safetensors.index.json", 239890, "b67dbc7062e1ccf472faba4222d631dc42929c827fbdaed1ec8e34fe0601819a"),
    ("modeling_phi4mm.py", 116057, "e2b44eb7a66d6cc54524cee1ff9ba92d0658d435ea8900329ea0dbdb85c6439d"),
    ("preprocessor_config.json", 482, "9db19b9663fb86f04c0f11d3a9b7f65a19f13d4543fb4a15bd33f82b0c92d64f"),
    ("processing_phi4mm.py", 32775, "84914d3e12256b4e2186e040c9830c11408468b6774f42afe85e6f8de2626d50"),
    ("processor_config.json", 121, "798fc4cd09c067053af27f07f0d2b329b471c5b2eb923ceb06efa41dee660c05"),
    ("special_tokens_map.json", 473, "57491904f8680d4b52ed440f1f7ba48cad1c31ecf3eb453b03484e6ff4723ae8"),
    ("speech_conformer_encoder.py", 110521, "3742827e945732cc5deea4a95e14004da037044431a94e3f3fac26239e614e3a"),
    ("tokenizer.json", 15524479, "4c1b9f641d4f8b7247b8d5007dd3b6a9f6a87cb5123134fe0d326f14d10c0585"),
    ("tokenizer_config.json", 3248, "a5da2e45718db78924ad5135a58a80b0303596acf54a1dc5c912c98436ddcaf3"),
    ("vision_siglip_navit.py", 78218, "7d5c053341ee9c099126fe675d5dcdc0ed5c0246f92fffdec78a1ab2f804e28d"),
    ("vocab.json", 3910310, "6cb65a857824fa6615bb1782d95d882617a8bbce1da0317118586b36f39e98bd"),
)
# Upstream Python that transformers executes under trust_remote_code=True; each is digest-verified above first.
REMOTE_CODE_FILES = ("configuration_phi4mm.py", "modeling_phi4mm.py", "processing_phi4mm.py", "speech_conformer_encoder.py", "vision_siglip_navit.py")
# Language-model linears load 4-bit; the vision encoder, projector, embeddings and the checkpoint's LoRA stay float16.
NF4_SKIP_MODULES = ("lm_head", "embed_tokens", "embed_tokens_extend", "image_embed", "audio_embed", "img_processor", "encoder", "lora_A", "lora_B")
INSTRUCTION = (
    "Describe only what is visible in this grayscale close-up photograph of a surface, in at most three short sentences. "
    "Mention the texture, the illumination, and the shape and size of any visible mark. "
    "If you cannot tell what the material is, write 'material: unknown'. "
    "Do not guess the alloy, the manufacturing process or how severe any mark is."
)
MAX_NEW_TOKENS = 96
DECODING = {"do_sample": False, "strategy": "greedy", "max_new_tokens": MAX_NEW_TOKENS, "attention": "eager", "quantization": "nf4 (bitsandbytes, double quantisation, float16 compute)"}


def stage_and_verify(root: Path) -> dict:
    from huggingface_hub import hf_hub_download

    fetched = 0
    for rel, size, _ in FILES:
        path = root / rel
        if not path.is_file() or path.stat().st_size != size:
            print(f"  fetching {rel} ({size / 1e9:.2f} GB)")
            hf_hub_download(MODEL_ID, rel, revision=MODEL_REVISION, local_dir=str(root))
            fetched += size
    for rel, size, digest in FILES:
        path = root / rel
        if path.stat().st_size != size or core.sha256_file(path) != digest:
            raise core.ContractError(f"{MODEL_ID}@{MODEL_REVISION}: {rel} fails its pinned digest; refusing to load (no fallback model is used)")
    print(f"verified {len(FILES)} files against their pinned SHA-256 digests")
    return {"downloaded_bytes": fetched, "files": len(FILES), "total_bytes": sum(s for _, s, _ in FILES)}


def install_adapter_switch(model) -> None:
    """The checkpoint routes images through its built-in 'vision' LoRA and calls set_lora_adapter on every forward.
    Upstream's version also flips requires_grad on 4-bit tensors; this replacement only switches the active adapter."""
    from peft.tuners.tuners_utils import BaseTunerLayer

    layers = [m for m in model.modules() if isinstance(m, BaseTunerLayer)]

    def switch(name: str) -> None:
        for m in layers:
            m._active_adapter = [name]
            m._disable_adapters = False

    def unset() -> None:
        for m in layers:
            m._disable_adapters = True

    model.set_lora_adapter = switch
    model.unset_lora_adapter = unset


def main(cfg_path: str) -> None:
    cfg = core.read_json(cfg_path)
    work, out = Path(cfg["work_dir"]), Path(cfg["out_dir"])
    if not cfg["allow_phi4_remote_code"]:
        raise SystemExit("Phi-4-multimodal ships its model code as Python files in the model repository. Set ALLOW_PHI4_REMOTE_CODE = True after reviewing the trust boundary described above the cell.")
    rows = [r for r in core.read_split_manifest(out / "split_manifest.csv") if r["exemplar"]]
    if any(r["split"] != "train" or not r["selected"] for r in rows):
        raise core.ContractError("a captioning exemplar is not a selected training image; refusing to describe it")
    order = core.read_json(out / "data_manifest.json")["class_order"]
    rows.sort(key=lambda r: (order.index(r["label"]), r["budget_rank"]))
    data_root = work / "data" / ("sdi" if cfg["data_source"] == "bosch" else "byod")
    root = work / "weights" / "phi4-multimodal-instruct"
    t0 = time.time()
    staging = stage_and_verify(root)
    stage_seconds = round(time.time() - t0, 1)

    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, GenerationConfig

    if not torch.cuda.is_available():
        raise SystemExit("Phi-4 is loaded 4-bit with bitsandbytes and needs a CUDA GPU: Runtime > Change runtime type > T4 GPU")
    torch.manual_seed(0)
    t1 = time.time()
    processor = AutoProcessor.from_pretrained(str(root), trust_remote_code=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, llm_int8_skip_modules=list(NF4_SKIP_MODULES))
    model = AutoModelForCausalLM.from_pretrained(str(root), trust_remote_code=True, device_map="cuda", _attn_implementation="eager", torch_dtype=torch.float16, quantization_config=quant).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    install_adapter_switch(model)
    generation_config = GenerationConfig.from_pretrained(str(root))
    load_seconds = round(time.time() - t1, 1)
    print(f"loaded 4-bit in {load_seconds} s; GPU memory allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    prompt = f"<|user|><|image_1|>{INSTRUCTION}<|end|><|assistant|>"
    records = []
    for r in rows:
        path = data_root / r["relpath"]
        image = Image.open(path).convert("RGB")
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda")
        t = time.time()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, generation_config=generation_config)
        new_tokens = generated[:, inputs["input_ids"].shape[1]:]
        text = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        n = int(new_tokens.shape[1])
        eos = generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list | tuple) else [eos]) - {None}
        last = int(new_tokens[0, -1]) if n else None
        stop_reason = "end_of_sequence" if last in eos_ids else ("max_new_tokens" if n >= MAX_NEW_TOKENS else "other")
        records.append({
            "image_id": r["image_id"], "label": r["label"], "exemplar_rank": r["budget_rank"], "split": r["split"],
            "image_sha256": r["file_sha256"], "instruction": INSTRUCTION, "response": text, "new_tokens": n,
            "stop_reason": stop_reason, "eos_token_ids": sorted(eos_ids), "decoding": DECODING, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "label_shown_to_model": False, "seconds": round(time.time() - t, 2),
        })
        print(f"[{r['label']} #{r['budget_rank']}] {r['image_id']}: {text}")
    core.write_jsonl(out / "caption_records.jsonl", records)
    summary = {
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "license": MODEL_LICENSE, "remote_code_files": {rel: d for rel, _, d in FILES if rel in REMOTE_CODE_FILES}},
        "staging": staging, "stage_seconds": stage_seconds, "load_seconds": load_seconds,
        "peak_gpu_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "records": len(records), "decoding": DECODING,
    }
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    if cfg["delete_model_weights_after_use"]:
        shutil.rmtree(root, ignore_errors=True)
        summary["weights_deleted_after_use"] = True
    core.write_json(work / "state" / "phi4_stage.json", summary)
    print(core.canonical_json({k: summary[k] for k in ("stage_seconds", "load_seconds", "peak_gpu_allocated_gib", "records")}))


if __name__ == "__main__":
    core.run_main(main, sys.argv[1])
