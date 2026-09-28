"""Stage 3 (lab environment): compile Phi-4 descriptions into bounded FLUX prompts and freeze them."""
# ruff: noqa: E501
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdi_core as core  # noqa: E402


def main(cfg_path: str) -> None:
    cfg = core.read_json(cfg_path)
    out = Path(cfg["out_dir"])
    order = core.read_json(out / "data_manifest.json")["class_order"]
    records = core.read_jsonl(out / "caption_records.jsonl")
    selected = {r["image_id"] for r in core.read_split_manifest(out / "split_manifest.csv") if r["selected"] and r["split"] == "train"}
    outside = sorted({r["image_id"] for r in records} - selected)
    if outside:
        raise core.ContractError(f"caption records reference images outside the selected training budget: {outside}")
    for r in records:
        terms = core.extract_terms(r["response"])
        print(f"[{r['label']} #{r['exemplar_rank']}] matched terms: {terms}")
    prompts = core.compile_prompts(records, order)
    body = {"template_base": core.PROMPT_BASE, "vocabulary": core.VOCABULARY, "term_limits": core.TERM_LIMITS, "max_prompt_chars": core.MAX_PROMPT_CHARS, "prompts": prompts}
    core.write_json(out / "prompts.json", body)
    for p in prompts:
        tag = "Phi-4-guided" if p["phi4_guided"] else f"FALLBACK ({p['fallback_reason']})"
        print(f"{p['prompt_id']:<12} {tag}\n    {p['text']}")
    print(core.canonical_json({"prompts": len(prompts), "phi4_guided": sum(p["phi4_guided"] for p in prompts), "prompts_sha256": core.sha256_file(out / "prompts.json")}))


if __name__ == "__main__":
    core.run_main(main, sys.argv[1])
