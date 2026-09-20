"""Evaluation of generated images with a frozen CLIP scorer: prompt alignment, zero-shot label accuracy among
the dataset's captions, and similarity to the held-out real photographs of the same caption.

The scorer is the pinned `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` snapshot (MIT), loaded from its verified
directory; it never takes part in generation or training, so a change in these scores can only come from the
generator. Every number is a cosine similarity of L2-normalised CLIP embeddings (×100) or an accuracy derived
from one — tutorial sample-sanity evidence, not a benchmark, and not a human judgement of image quality.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .pipeline import SCORER_ID, SCORER_REVISION, SCORER_WEIGHTS_DIR, stage_missing_scorer_files, verify_scorer_snapshot


def _projected(features: Any) -> Any:
    """The projected CLIP embedding as a tensor: `transformers` 5 returns the encoder's model output (the projection
    written into `pooler_output`) from `get_text_features` / `get_image_features`, earlier releases the tensor."""
    return features if hasattr(features, "shape") else features.pooler_output


class ClipScorer:
    """Frozen CLIP image/text embedder on a verified snapshot."""

    def __init__(self, *, device: str | None = None, weights_dir: str | Path | None = None, allow_download: bool = False) -> None:
        root = Path(weights_dir) if weights_dir is not None else SCORER_WEIGHTS_DIR
        stage_missing_scorer_files(root, allow_download=allow_download)
        verify_scorer_snapshot(root)
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(str(root), torch_dtype=torch.float32).to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.processor = CLIPProcessor.from_pretrained(str(root))
        self.weights_dir = root
        self.identity = {"id": SCORER_ID, "revision": SCORER_REVISION}

    def image_embeddings(self, images: Sequence[Any]) -> Any:
        import torch

        out = []
        with torch.inference_mode():
            for start in range(0, len(images), 16):
                batch = self.processor(images=[im.convert("RGB") for im in images[start : start + 16]], return_tensors="pt")
                features = _projected(self.model.get_image_features(pixel_values=batch["pixel_values"].to(self.device)))
                out.append(torch.nn.functional.normalize(features.float(), dim=-1).cpu())
        return torch.cat(out)

    def text_embeddings(self, texts: Sequence[str]) -> Any:
        import torch

        with torch.inference_mode():
            batch = self.processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
            features = _projected(
                self.model.get_text_features(
                    input_ids=batch["input_ids"].to(self.device), attention_mask=batch["attention_mask"].to(self.device)
                )
            )
        return torch.nn.functional.normalize(features.float(), dim=-1).cpu()


def score_generations(
    scorer: ClipScorer,
    generated: Sequence[Mapping[str, Any]],
    *,
    references: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score `generated` records (`{prompt, image}`) with three measures:

    * `clip_prompt_similarity` — mean cosine(image, its prompt) × 100 (prompt alignment);
    * `label_accuracy` — the fraction of images whose nearest prompt among the distinct prompts is their own
      (zero-shot classification of the generated image among the dataset's captions; argmax rule);
    * `reference_similarity` — mean cosine(image, mean embedding of the real `references` with the same caption)
      × 100 when references are given (how close the generations sit to the held-out photographs)."""
    if not generated:
        raise ValueError("no generated records to score")
    started = time.perf_counter()
    prompts = list(dict.fromkeys(str(g["prompt"]) for g in generated))
    text = scorer.text_embeddings(prompts)
    images = scorer.image_embeddings([g["image"] for g in generated])
    index = {p: i for i, p in enumerate(prompts)}
    sims = images @ text.T  # (n_images, n_prompts)
    own = [float(sims[i, index[str(g["prompt"])]]) for i, g in enumerate(generated)]
    nearest = sims.argmax(dim=1).tolist()
    correct = [nearest[i] == index[str(g["prompt"])] for i, g in enumerate(generated)]
    per_image = []
    ref_means: dict[str, Any] = {}
    if references:
        ref_images = scorer.image_embeddings([r["image"] for r in references])
        for caption in prompts:
            rows = [i for i, r in enumerate(references) if str(r["caption"]) == caption]
            if rows:
                mean = ref_images[rows].mean(dim=0)
                ref_means[caption] = mean / mean.norm()
    ref_scores = []
    for i, g in enumerate(generated):
        entry = {
            "prompt": str(g["prompt"]),
            "seed": g.get("seed"),
            "clip_prompt_similarity": round(own[i] * 100, 3),
            "nearest_prompt": prompts[nearest[i]],
            "correct": bool(correct[i]),
        }
        if str(g["prompt"]) in ref_means:
            value = float(images[i] @ ref_means[str(g["prompt"])]) * 100
            entry["reference_similarity"] = round(value, 3)
            ref_scores.append(value)
        per_image.append(entry)
    report: dict[str, Any] = {
        "scorer": dict(scorer.identity),
        "n_images": len(generated),
        "n_prompts": len(prompts),
        "clip_prompt_similarity": round(sum(own) / len(own) * 100, 3),
        "label_accuracy": round(sum(correct) / len(correct), 4),
        "decision_rule": "argmax cosine similarity over the distinct prompts (no threshold)",
        "per_image": per_image,
        "seconds": round(time.perf_counter() - started, 2),
    }
    if ref_scores:
        report["reference_similarity"] = round(sum(ref_scores) / len(ref_scores), 3)
        report["n_references"] = len(references or [])
    return report


def real_photo_baseline(scorer: ClipScorer, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The same three measures on real photographs of the dataset (image = the record's own photo, prompt = its
    caption, references = the other records): the ceiling a generator could reach on these metrics."""
    generated = [{"prompt": r["caption"], "image": r["image"], "seed": None} for r in records]
    report = score_generations(scorer, generated, references=records)
    report["note"] = "real held-out photographs scored as if generated (references include each photo itself)"
    return report
