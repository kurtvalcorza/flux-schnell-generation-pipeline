"""Adaptation, evaluation and artifact tests on a stub transformer/VAE (torch required, no weights, no diffusers):
the flow-matching loss, the training loop, epoch selection, the transactional guarantee, the residency rule for the
text encoders and the artifact round trip with its refusals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from conftest import synthetic_records  # noqa: E402
from flux_schnell_generation_pipeline import LORA_TENSORS, FluxSchnellPipeline, lora_parameter_names  # noqa: E402
from flux_schnell_generation_pipeline import pipeline as pl  # noqa: E402

RANK, DIM, EMB = 2, 6, 8


class _StubTransformer(torch.nn.Module):
    """Names follow peft's layout on the FLUX blocks; the output depends on the LoRA tensors so training moves them."""

    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.checkpointing = False
        blocks = ("transformer_blocks.0", "transformer_blocks.1", "single_transformer_blocks.0", "single_transformer_blocks.1")
        for block in blocks:
            for target in ("to_q", "to_k"):
                stem = f"{block}.attn.{target}".replace(".", "__")
                self.register_parameter(f"{stem}__lora_A__default__weight", torch.nn.Parameter(torch.randn(RANK, DIM) * 0.1))
                self.register_parameter(f"{stem}__lora_B__default__weight", torch.nn.Parameter(torch.zeros(DIM, RANK)))

    def named_parameters(self, *args, **kwargs):  # type: ignore[override]
        for name, param in super().named_parameters(*args, **kwargs):
            yield name.replace("__", "."), param

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return {k.replace("__", "."): v for k, v in super().state_dict(*args, **kwargs).items()}

    def enable_gradient_checkpointing(self) -> None:
        self.checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.checkpointing = False

    def forward(
        self, hidden_states, timestep, guidance, pooled_projections, encoder_hidden_states, txt_ids, img_ids, return_dict
    ):
        assert guidance is None and not return_dict
        assert txt_ids.shape == (encoder_hidden_states.shape[1], 3) and img_ids.shape[1] == 3
        gain = 1.0
        params = dict(self.named_parameters())
        for name, param in params.items():
            if ".lora_B." in name:
                gain = gain + (param @ params[name.replace("lora_B", "lora_A")]).mean()
        pred = hidden_states * gain + 0.01 * encoder_hidden_states.mean() + 0.01 * pooled_projections.mean()
        pred = pred - timestep.view(-1, 1, 1) * 0.1
        return (pred,)


class _StubVAE:
    config = SimpleNamespace(scaling_factor=0.3611, shift_factor=0.1159)

    def encode(self, pixels):
        pooled = torch.nn.functional.adaptive_avg_pool2d(pixels, 4)[:, :2]  # (B, 2, 4, 4)
        latent = pooled.repeat(1, pl.LATENT_CHANNELS // 2, 1, 1)  # 16 latent channels, a 4×4 grid -> 4 tokens
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda generator=None: latent))


def _pipeline(monkeypatch, *, transformer: bool = True) -> FluxSchnellPipeline:
    stub = _StubTransformer()
    names = lora_parameter_names(stub)
    monkeypatch.setattr(pl, "LORA_PARAMETERS", sum(dict(stub.named_parameters())[n].numel() for n in names))
    pipe = FluxSchnellPipeline(
        vae=_StubVAE(),
        scheduler_config={"num_train_timesteps": 1000, "shift": 1.0, "use_dynamic_shifting": False},
        tokenizer=None,
        tokenizer_2=None,
        device="cpu",
        dtype=torch.float32,
        compute_dtype="float32",
        weights_dir=Path("unused"),
        source="stub",
        use_lora=True,
        transformer=stub if transformer else None,
    )
    generator = torch.Generator().manual_seed(0)
    for prompt in ("a photo of a red bird", "a photo of a blue bird"):
        pipe._prompt_cache[prompt] = {
            "embeds": torch.randn(4, EMB, generator=generator),
            "pooled": torch.randn(EMB, generator=generator),
            "tokens": 4,
        }
    return pipe


def test_stub_matches_the_contract_shape():
    names = lora_parameter_names(_StubTransformer())
    assert len(names) == 16 and all(".lora_" in n for n in names) and LORA_TENSORS == 380
    assert pl.count_parameters(_StubTransformer()) == 1 + 16 * RANK * DIM


def test_evaluate_is_paired_and_seeded(monkeypatch):
    pipe = _pipeline(monkeypatch)
    records = synthetic_records(4)
    first = pipe.evaluate(records, seed=3)
    second = pipe.evaluate(records, seed=3)
    assert first["flow_matching_mse"] == second["flow_matching_mse"] and first["per_record"] == second["per_record"]
    assert set(first["by_sigma"]) == {str(s) for s in pl.EVAL_SIGMAS} and first["adapted"] is False
    assert pipe.evaluate(records, seed=4)["flow_matching_mse"] != first["flow_matching_mse"]
    assert first["metric"].startswith("flow_matching_mse")


def test_flow_matching_target_is_velocity(monkeypatch):
    """At σ the transformer sees (1 − σ)·x₀ + σ·ε and is scored against ε − x₀ (checked through a spy transformer)."""
    pipe = _pipeline(monkeypatch)
    seen = {}

    def spy(hidden_states, timestep, **kwargs):
        seen.setdefault("calls", []).append((hidden_states.clone(), timestep.clone()))
        return (torch.zeros_like(hidden_states),)

    pipe.transformer.forward = spy  # nn.Module.__call__ resolves `forward` on the instance first
    records = synthetic_records(1)
    report = pipe.evaluate(records, seed=0)
    assert len(seen["calls"]) == len(pl.EVAL_SIGMAS)
    tokens = pl.pack_latents(pipe._latents(records, seed=0))
    for (noisy, sigma), expected in zip(seen["calls"], pl.EVAL_SIGMAS, strict=True):
        assert float(sigma[0]) == pytest.approx(expected)
        generator = torch.Generator(device="cpu").manual_seed(int(expected * 1_000))
        noise = torch.randn(tokens.shape, generator=generator)
        assert torch.allclose(noisy, (1 - expected) * tokens + expected * noise, atol=1e-6)
        # a zero prediction is scored against the velocity ε − x₀
        assert report["by_sigma"][str(expected)] == pytest.approx(float(((noise - tokens) ** 2).mean()), rel=1e-4)


def test_adapt_trains_only_lora_and_keeps_the_best_epoch(monkeypatch):
    pipe = _pipeline(monkeypatch)
    train, val = synthetic_records(6), synthetic_records(2, seed=50)
    before = pipe.evaluate(val, seed=0)["flow_matching_mse"]
    base_before = pipe.transformer.base.clone()
    seen = []
    result = pipe.adapt(train, val, epochs=3, lr=1e-2, seed=0, progress=seen.append)
    assert [e["epoch"] for e in seen] == [0, 1, 2, 3] and seen[0]["note"].startswith("frozen model")
    assert result["history"][0]["val_loss"] == before
    assert result["best_epoch"] == min(range(4), key=lambda i: result["history"][i]["val_loss"])
    assert result["n_trainable"] == pl.LORA_PARAMETERS and result["n_steps"] == 18
    assert result["precision"].startswith("nf4 base, float32 autocast") and result["quantization"] == "nf4"
    assert result["objective"].startswith("rectified-flow velocity MSE")
    assert torch.equal(pipe.transformer.base, base_before)
    assert pipe.adapter is not None and all(not p.requires_grad for p in pipe.transformer.parameters())
    assert pipe.transformer.checkpointing is False
    assert pipe.evaluate(val, seed=0)["flow_matching_mse"] == result["history"][result["best_epoch"]]["val_loss"]


def test_adapt_is_transactional_when_the_progress_callback_raises(monkeypatch):
    pipe = _pipeline(monkeypatch)
    initial = {k: v.clone() for k, v in pipe.transformer.state_dict().items()}

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        pipe.adapt(synthetic_records(4), None, epochs=2, progress=boom)
    assert pipe.adapter is None
    assert all(torch.equal(initial[k], v) for k, v in pipe.transformer.state_dict().items())
    assert all(not p.requires_grad for p in pipe.transformer.parameters())
    assert pipe.transformer.checkpointing is False


def test_adapt_refusals(monkeypatch):
    pipe = _pipeline(monkeypatch)
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(synthetic_records(4), epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(synthetic_records(4), lr=1.0)
    with pytest.raises(ValueError, match="batch_size"):
        pipe.adapt(synthetic_records(4), batch_size=8)
    with pytest.raises(ValueError, match="4..2000"):
        pipe.adapt(synthetic_records(3))
    with pytest.raises(ValueError, match="prompt not encoded"):
        pipe.adapt([{**r, "caption": "an unseen caption"} for r in synthetic_records(4)])
    pipe.use_lora = False
    with pytest.raises(ValueError, match="use_lora=True"):
        pipe.adapt(synthetic_records(4))
    with pytest.raises(ValueError, match="nothing to save"):
        pipe.save_artifact("unused")


def test_generate_and_steps_refusals(monkeypatch):
    pipe = _pipeline(monkeypatch)
    with pytest.raises(ValueError, match="steps must be an int in 1..8"):
        pipe.generate(["a photo of a red bird"], steps=0)
    with pytest.raises(ValueError, match="steps must be an int in 1..8"):
        pipe.generate(["a photo of a red bird"], steps=9)
    with pytest.raises(ValueError, match="prompt not encoded"):
        pipe.generate(["a photo of a green bird"])


def test_text_encoders_are_refused_while_the_transformer_is_resident(monkeypatch):
    pipe = _pipeline(monkeypatch)
    assert pipe.encode_prompts(["a photo of a red bird"]) == {"encoded": 0, "cached": 2}  # cached: no encoder needed
    with pytest.raises(ValueError, match="transformer is resident"):
        pipe.encode_prompts(["a photo of a green bird"])
    assert pipe.release_transformer() is True and pipe.transformer is None and pipe.release_transformer() is False
    cache = pipe.export_prompt_cache()
    fresh = _pipeline(monkeypatch)
    fresh._prompt_cache.clear()
    with pytest.raises(ValueError, match="unexpected shape"):
        fresh.import_prompt_cache(cache)  # the stub embeddings are not (256, 4096) / (768,)
    good = {"p": {"embeds": torch.zeros(pl.MAX_PROMPT_TOKENS, pl.T5_HIDDEN), "pooled": torch.zeros(pl.POOLED_DIM), "tokens": 3}}
    assert fresh.import_prompt_cache(good) == 1


def test_artifact_round_trip_and_refusals(tmp_path, monkeypatch):
    pipe = _pipeline(monkeypatch)
    records = synthetic_records(4)
    pipe.adapt(records, epochs=1, lr=1e-2)
    adapted = pipe.evaluate(records, seed=0)["flow_matching_mse"]
    out = pipe.save_artifact(tmp_path / "adapter", metadata={"tutorial": "test"})
    manifest = json.loads((out / pl.ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["format"] == pl.ARTIFACT_FORMAT and len(manifest["tensors"]) == 16
    assert manifest["metadata"] == {"tutorial": "test"}
    assert manifest["base_model"]["staging"] == {"repo": pl.STAGING_ID, "revision": pl.STAGING_REVISION}
    assert manifest["base_model"]["quantization"] == "nf4" and manifest["base_model"]["license"] == "apache-2.0"
    fresh = _pipeline(monkeypatch)
    assert fresh.evaluate(records, seed=0)["flow_matching_mse"] != adapted
    fresh.load_artifact(out)
    assert fresh.evaluate(records, seed=0)["flow_matching_mse"] == adapted and fresh.adapter["best_epoch"] == 1
    # digest mismatch
    (out / pl.ARTIFACT_WEIGHTS_NAME).write_bytes((out / pl.ARTIFACT_WEIGHTS_NAME).read_bytes() + b"\0")
    with pytest.raises(ValueError, match="digest or size"):
        _pipeline(monkeypatch).load_artifact(out)
    # tensor list outside the scope
    pipe.save_artifact(out)
    manifest = json.loads((out / pl.ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["tensors"] = manifest["tensors"][:-1] + ["base"]
    (out / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        _pipeline(monkeypatch).load_artifact(out)
    # a pipeline without the adapter attached refuses
    pipe.save_artifact(out)
    plain = _pipeline(monkeypatch)
    plain.use_lora = False
    with pytest.raises(ValueError, match="use_lora=True"):
        plain.load_artifact(out)
    digest = hashlib.sha256((out / pl.ARTIFACT_WEIGHTS_NAME).read_bytes()).hexdigest()
    assert digest == json.loads((out / pl.ARTIFACT_MANIFEST_NAME).read_text())["files"][0]["sha256"]
