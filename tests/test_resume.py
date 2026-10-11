"""Resume reproducibility: positional seeding makes restarts bit-exact.

The contract under test: every stochastic training draw (timestep, noise,
text-dropout mask, caption choice) is a pure function of position
(seed, rank, epoch, batch index / sample index), so a run resumed from any
checkpoint continues the unbroken run's draw sequence exactly.
"""

import random
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from iris3b.config import (
    Config,
    DataConfig,
    EMAConfig,
    OptimizerConfig,
    TextEncoderConfig,
    TrainConfig,
)
from iris3b.data.datasets import select_caption
from iris3b.registry import TEXT_ENCODERS
from iris3b.seeding import mix_seed
from iris3b.text.base import TextEncoder, TextEncoding
from iris3b.train.ckpt import load_checkpoint, save_checkpoint
from tiny_config import tiny_model_config


@TEXT_ENCODERS.register("stub")
class StubTextEncoder(TextEncoder):
    """Deterministic per-prompt embeddings; no downloads, no global RNG."""

    def __init__(self, cfg: TextEncoderConfig, device="cpu"):
        self.dim = cfg.dim
        self.max_length = cfg.max_length
        self.device = torch.device(device)

    def _embed(self, prompts: list[str]) -> TextEncoding:
        rows = []
        for p in prompts:
            gen = torch.Generator().manual_seed(mix_seed(len(p), *(ord(c) for c in p[:32])))
            rows.append(torch.randn(self.max_length, self.dim, generator=gen))
        emb = torch.stack(rows).to(self.device)
        mask = torch.ones(emb.shape[:2], dtype=torch.long, device=self.device)
        return TextEncoding(embeddings=emb, mask=mask)

    def encode(self, prompts: list[str]) -> TextEncoding:
        return self._embed(list(prompts))

    def null(self, negative_prompt: str = "") -> TextEncoding:
        return self._embed([negative_prompt])

    def to(self, device) -> "StubTextEncoder":
        self.device = torch.device(device)
        return self


def test_mix_seed_stable_and_order_sensitive():
    assert mix_seed(1, 2, 3) == mix_seed(1, 2, 3)
    assert mix_seed(1, 2, 3) != mix_seed(1, 3, 2)
    assert mix_seed(1, 2, 3) != mix_seed(3, 2, 1)
    assert mix_seed(0) != mix_seed(0, 0)
    for parts in [(0,), (7, 0, 0), (2**63, 5), (1, 2, 3)]:
        assert 0 <= mix_seed(*parts) < 2**63


def _multi_caption_cfg() -> DataConfig:
    return DataConfig(caption_fields=["a", "b", "c", "d"])


def test_caption_choice_is_positional_not_stream_based():
    info = {k: f"caption {k}" for k in "abcd"}
    cfg = _multi_caption_cfg()

    def pick(seed: int, epoch: int, idx: int) -> str:
        return select_caption(info, cfg, random.Random(mix_seed(seed, epoch, idx)))

    # same position key -> same caption, regardless of the global RNG stream
    random.seed(0)
    first = pick(7, 1, 123)
    random.seed(999)
    assert pick(7, 1, 123) == first

    # resampled across epochs, uniform-ish across fields within an epoch
    picks_e1 = [pick(7, 1, i) for i in range(200)]
    picks_e2 = [pick(7, 2, i) for i in range(200)]
    assert picks_e1 != picks_e2
    assert set(picks_e1) == {f"caption {k}" for k in "abcd"}


def _tiny_run_cfg(work_dir: Path, **train_overrides) -> Config:
    train = TrainConfig(
        batch_size=2,
        num_epochs=1,
        max_steps=6,
        mixed_precision="no",
        gradient_clip=0.5,
        text_dropout=0.5,
        seed=7,
        optimizer=OptimizerConfig(name="adamw", lr=1e-3, betas=[0.9, 0.95], auto_lr="none", warmup_steps=2),
        ema=EMAConfig(enabled=True, decay=0.999),
        save_every_steps=3,
        save_every_epochs=0,
        sample_every_steps=0,
        log_every=1000,
    )
    train = replace(train, **train_overrides)
    return Config(
        name="resume-test",
        work_dir=str(work_dir),
        report_to="none",
        model=tiny_model_config(),
        text_encoder=TextEncoderConfig(name="stub", dim=32, max_length=16),
        repa=replace(Config().repa, weight=0.0),
        data=DataConfig(type="synthetic", image_size=32, num_workers=0),
        train=train,
    )


def _run(cfg: Config) -> None:
    from accelerate.state import AcceleratorState

    from iris3b.train.trainer import Trainer

    AcceleratorState._reset_state()
    Trainer(cfg).run()


def _assert_tree_equal(a, b, path=""):
    assert type(a) is type(b), f"type mismatch at {path}: {type(a)} vs {type(b)}"
    if isinstance(a, dict):
        assert a.keys() == b.keys(), f"key mismatch at {path}"
        for k in a:
            _assert_tree_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"length mismatch at {path}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_tree_equal(x, y, f"{path}[{i}]")
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a, b), f"tensor mismatch at {path}"
    else:
        assert a == b, f"value mismatch at {path}: {a} vs {b}"


def test_resumed_run_is_bit_exact(tmp_path: Path):
    # unbroken run: 6 steps, checkpoints at 3 and 6
    dir_a = tmp_path / "unbroken"
    _run(_tiny_run_cfg(dir_a))
    ckpt_3 = dir_a / "checkpoints" / "epoch_1_step_3.pth"
    ckpt_6a = dir_a / "checkpoints" / "epoch_1_step_6.pth"
    assert ckpt_3.exists() and ckpt_6a.exists()

    # resumed run: restart from step 3, continue to 6
    dir_b = tmp_path / "resumed"
    _run(_tiny_run_cfg(dir_b, resume_from=str(ckpt_3)))
    ckpt_6b = dir_b / "checkpoints" / "epoch_1_step_6.pth"
    assert ckpt_6b.exists()

    pay_a = torch.load(ckpt_6a, map_location="cpu", weights_only=False)
    pay_b = torch.load(ckpt_6b, map_location="cpu", weights_only=False)
    for key in ("state_dict", "state_dict_ema", "optimizer"):
        assert key in pay_a and key in pay_b, f"missing payload key {key}"
        _assert_tree_equal(pay_a[key], pay_b[key], key)
    assert pay_a["step"] == pay_b["step"] == 6


def test_resumed_shape_batched_run_is_bit_exact(tmp_path: Path):
    """Exact resume through the ShapeBatchSampler (its state nests world_size under layout)."""

    def cfg_for(work_dir: Path, **train_overrides) -> Config:
        cfg = _tiny_run_cfg(work_dir, **train_overrides)
        data = replace(cfg.data, shape_policy="area", shape_align=cfg.model.patch_size)
        return replace(cfg, data=data)

    dir_a = tmp_path / "unbroken"
    _run(cfg_for(dir_a))
    ckpt_3 = dir_a / "checkpoints" / "epoch_1_step_3.pth"
    position = torch.load(ckpt_3, map_location="cpu", weights_only=False)["data_position"]
    assert position["kind"] == "shape_batches" and "world_size" not in position

    dir_b = tmp_path / "resumed"
    _run(cfg_for(dir_b, resume_from=str(ckpt_3)))
    pay_a = torch.load(dir_a / "checkpoints" / "epoch_1_step_6.pth", map_location="cpu", weights_only=False)
    pay_b = torch.load(dir_b / "checkpoints" / "epoch_1_step_6.pth", map_location="cpu", weights_only=False)
    for key in ("state_dict", "state_dict_ema", "optimizer"):
        _assert_tree_equal(pay_a[key], pay_b[key], key)
    assert pay_a["step"] == pay_b["step"] == 6


def test_load_checkpoint_optimizer_core_prefix(tmp_path: Path):
    torch.manual_seed(0)
    core1 = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 4))
    aux_repa = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 8), nn.SiLU(), nn.Linear(8, 2))
    opt1 = torch.optim.AdamW(
        list(core1.parameters()) + list(aux_repa.parameters()),
        lr=1e-3,
        betas=(0.9, 0.95),
    )
    aux_repa(core1(torch.randn(3, 4))).sum().backward()
    opt1.step()
    path = tmp_path / "phase1.pth"
    save_checkpoint(path, core1, opt1, epoch=1, step=3)

    torch.manual_seed(1)
    core2 = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 4))
    aux_new = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 8), nn.SiLU(), nn.Linear(8, 4))
    ncore = len(list(core2.parameters()))

    def fresh_opt() -> torch.optim.AdamW:
        return torch.optim.AdamW(
            list(core2.parameters()) + list(aux_new.parameters()),
            lr=1e-3,
            betas=(0.9, 0.95),
        )

    # Full replay "loads" (same tensor count) but pairs the old head's moments
    # with the new head; the shape mismatch detonates on the next step.
    opt_full = fresh_opt()
    load_checkpoint(path, core2, opt_full)
    aux_new(core2(torch.randn(3, 4))).sum().backward()
    with pytest.raises(RuntimeError):
        opt_full.step()

    # Core-prefix load: core moments carried exactly, new head left stateless.
    opt_core = fresh_opt()
    _, step = load_checkpoint(path, core2, opt_core, optimizer_prefix=ncore)
    assert step == 3
    saved = torch.load(path, map_location="cpu", weights_only=False)["optimizer"]["state"]
    for idx, p in enumerate(core2.parameters()):
        _assert_tree_equal(opt_core.state[p]["exp_avg"], saved[idx]["exp_avg"], f"core[{idx}]")
    assert all(p not in opt_core.state for p in aux_new.parameters())
    for p in list(core2.parameters()) + list(aux_new.parameters()):
        p.grad = None
    aux_new(core2(torch.randn(3, 4))).sum().backward()
    opt_core.step()
