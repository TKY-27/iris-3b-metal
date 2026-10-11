"""Representation alignment: plain REPA and iREPA.

iREPA (arXiv 2512.10794) differs from plain REPA in exactly two places, and both
live in ``iris3b.repa``:

1. the projector is a ``Conv2d(3x3)`` applied on the student *token grid* rather
   than a per-token MLP, so neighbouring patches can talk before alignment;
2. the teacher tokens are *spatially normalized* first — the per-channel mean
   over the token axis is subtracted, which deletes the global (image-level)
   component and leaves patch-to-patch contrast as the only alignment signal.

The teacher is always a local stub: ``torch.hub.load`` (or
``transformers.AutoModel.from_pretrained``) is monkeypatched, so no test here
can touch the network or a weights cache.
"""

from dataclasses import replace

import pytest
import torch
from torch import nn

from iris3b.config import RepaConfig
from iris3b.repa import REPALoss, _spatial_normalize


class StubHFTeacher(nn.Module):
    """Transformers-shaped DINOv3 stub: [CLS, registers, patches] rows."""

    class _Config:
        num_register_tokens = 4

    class _Output:
        def __init__(self, last_hidden_state):
            self.last_hidden_state = last_hidden_state

    def __init__(self, dim: int, patch: int = 16):
        super().__init__()
        self.config = self._Config()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.extras = nn.Parameter(torch.randn(1, 1 + self.config.num_register_tokens, dim))

    def forward(self, pixel_values: torch.Tensor):
        patches = self.proj(pixel_values).flatten(2).permute(0, 2, 1)
        extras = self.extras.expand(patches.shape[0], -1, -1)
        return self._Output(torch.cat([extras, patches], dim=1))


def stub_hf(monkeypatch, dim: int, patch: int = 16) -> list[str]:
    """Replace transformers.AutoModel.from_pretrained with a local factory."""
    import transformers

    calls: list[str] = []

    def fake_from_pretrained(repo, **kwargs):
        calls.append(repo)
        return StubHFTeacher(dim, patch)

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", staticmethod(fake_from_pretrained))
    return calls


class StubTeacher(nn.Module):
    """Local ViT-shaped teacher: a patch conv, never downloaded.

    ``forward_features`` answers in the DINOv2 dict form so the production
    ``x_norm_patchtokens`` branch is the one under test.
    """

    def __init__(self, dim: int, patch: int = 14):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.proj(x).flatten(2).permute(0, 2, 1)
        return {"x_norm_patchtokens": tokens}


def stub_hub(monkeypatch, dim: int, patch: int = 14) -> list[tuple]:
    """Replace torch.hub.load with a local factory; returns the call log."""
    calls: list[tuple] = []

    def fake_load(repo, entry, **kwargs):
        calls.append((repo, entry, kwargs))
        return StubTeacher(dim, patch)

    monkeypatch.setattr(torch.hub, "load", fake_load)
    return calls


def tiny_cfg(**over) -> RepaConfig:
    base = RepaConfig(teacher_dim=16, proj_hidden_dim=24, teacher_image_size=224)
    return replace(base, **over)


# --------------------------------------------------------------------------
# 1. spatial normalization: the mechanism that makes iREPA not-REPA
# --------------------------------------------------------------------------


def test_spatial_normalize_deletes_the_global_component_at_gamma_one():
    torch.manual_seed(0)
    # a strong per-channel global component: the thing iREPA removes
    tokens = torch.randn(3, 256, 32) * 2.0 + torch.randn(1, 1, 32) * 5.0

    y = _spatial_normalize(tokens, 1.0)
    assert y.shape == tokens.shape
    # zero mean per channel over the TOKEN axis (dim=1), for every sample
    assert y.mean(dim=1).abs().max().item() < 1e-4
    # ...and unit-ish variance on the same axis
    torch.testing.assert_close(y.var(dim=1, unbiased=False), torch.ones(3, 32), rtol=0, atol=1e-3)
    # the batch/channel axes are NOT normalized: this is per-sample, per-channel
    assert y.mean(dim=(0, 1)).abs().max().item() < 1e-4


def test_spatial_normalize_at_gamma_zero_keeps_the_mean():
    """gamma=0 is plain per-channel scaling — the REPA-like degenerate case."""
    torch.manual_seed(1)
    tokens = torch.randn(2, 64, 8) + 3.0

    y0 = _spatial_normalize(tokens, 0.0)
    scale = torch.sqrt(tokens.var(dim=1, keepdim=True, unbiased=False) + 1e-6)
    torch.testing.assert_close(y0, tokens / scale)
    # the mean survives (only rescaled), which is exactly what gamma=1 destroys
    torch.testing.assert_close(y0.mean(dim=1), tokens.mean(dim=1) / scale.squeeze(1))
    assert y0.mean(dim=1).abs().min().item() > 1e-2
    # gamma is a pure shift, so the spread is identical either way
    torch.testing.assert_close(
        y0.var(dim=1, unbiased=False),
        _spatial_normalize(tokens, 1.0).var(dim=1, unbiased=False),
    )


def test_spatial_norm_gamma_interpolates_linearly():
    torch.manual_seed(2)
    tokens = torch.randn(2, 64, 8) + 3.0
    half = _spatial_normalize(tokens, 0.5)
    expected = (_spatial_normalize(tokens, 0.0) + _spatial_normalize(tokens, 1.0)) / 2.0
    torch.testing.assert_close(half, expected)


# --------------------------------------------------------------------------
# 2. projector construction
# --------------------------------------------------------------------------


def test_irepa_builds_a_conv3x3_projector(monkeypatch):
    stub_hub(monkeypatch, 16)
    loss = REPALoss(tiny_cfg(variant="irepa"), student_dim=32)

    proj = loss.projector
    assert isinstance(proj, nn.Conv2d)
    assert (proj.in_channels, proj.out_channels) == (32, 16)
    assert proj.kernel_size == (3, 3) and proj.padding == (1, 1)
    assert proj.stride == (1, 1) and proj.bias is not None


def test_repa_builds_the_three_layer_mlp_projector(monkeypatch):
    stub_hub(monkeypatch, 16)
    loss = REPALoss(tiny_cfg(variant="repa"), student_dim=32)

    proj = loss.projector
    assert isinstance(proj, nn.Sequential)
    assert [type(m) for m in proj] == [nn.Linear, nn.SiLU, nn.Linear, nn.SiLU, nn.Linear]
    assert (proj[0].in_features, proj[0].out_features) == (32, 24)
    assert (proj[2].in_features, proj[2].out_features) == (24, 24)
    assert (proj[4].in_features, proj[4].out_features) == (24, 16)


def test_unknown_variant_raises_before_the_teacher_is_touched(monkeypatch):
    calls = stub_hub(monkeypatch, 16)
    with pytest.raises(ValueError, match="unknown repa variant 'irepa2'"):
        REPALoss(tiny_cfg(variant="irepa2"), student_dim=32)
    assert calls == []  # cheap failure: no hub load on a typo'd variant


def test_teacher_load_failure_disables_the_loss_instead_of_raising(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no network")

    monkeypatch.setattr(torch.hub, "load", boom)
    loss = REPALoss(tiny_cfg(variant="irepa"), student_dim=32)
    assert loss.available is False and loss._teacher_holder == ()


# --------------------------------------------------------------------------
# 3. _project on the token grid
# --------------------------------------------------------------------------


def test_project_maps_a_token_grid_to_teacher_dim(monkeypatch):
    stub_hub(monkeypatch, 16)
    loss = REPALoss(tiny_cfg(variant="irepa"), student_dim=32)

    out = loss._project(torch.randn(2, 256, 32), (16, 16))
    assert out.shape == (2, 256, 16)

    # the conv really sees a row-major grid: an explicit reference agrees
    tokens = torch.randn(1, 16, 32)
    grid = tokens.permute(0, 2, 1).reshape(1, 32, 4, 4)
    ref = loss.projector(grid).flatten(2).permute(0, 2, 1)
    torch.testing.assert_close(loss._project(tokens, (4, 4)), ref)


def test_project_accepts_a_rectangular_grid(monkeypatch):
    """A 2x8 grid is a real shape, not an error: the caller states it."""
    stub_hub(monkeypatch, 16)
    loss = REPALoss(tiny_cfg(variant="irepa"), student_dim=32)
    tokens = torch.randn(1, 16, 32)
    ref = loss.projector(tokens.permute(0, 2, 1).reshape(1, 32, 2, 8))
    ref = ref.flatten(2).permute(0, 2, 1)
    torch.testing.assert_close(loss._project(tokens, (2, 8)), ref)
    # ...and it is NOT the same as pretending the 16 tokens are 4x4
    assert not torch.allclose(loss._project(tokens, (2, 8)), loss._project(tokens, (4, 4)))


def test_project_rejects_a_grid_that_contradicts_the_token_count(monkeypatch):
    stub_hub(monkeypatch, 16)
    irepa = REPALoss(tiny_cfg(variant="irepa"), student_dim=32)
    with pytest.raises(ValueError, match="token count 255 does not match student grid 16x16"):
        irepa._project(torch.randn(2, 255, 32), (16, 16))

    # the MLP variant has no grid assumption; the contrast is the point
    mlp = REPALoss(tiny_cfg(variant="repa"), student_dim=32)
    assert mlp._project(torch.randn(2, 255, 32), (16, 16)).shape == (2, 255, 16)


# --------------------------------------------------------------------------
# 4. forward: bounded cosine, gradients only into the projector
# --------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["repa", "irepa"])
def test_forward_is_a_bounded_cosine_that_trains_only_the_projector(monkeypatch, variant):
    torch.manual_seed(3)
    stub_hub(monkeypatch, 16)
    loss = REPALoss(tiny_cfg(variant=variant), student_dim=32)
    teacher = loss._teacher_holder[0]
    assert all(not p.requires_grad for p in teacher.parameters())

    images = torch.randn(2, 3, 64, 64).clamp(-1.0, 1.0)
    tokens = torch.randn(2, 256, 32, requires_grad=True)

    value = loss(images, tokens, (16, 16))
    assert value.shape == () and torch.isfinite(value)
    assert -1.0 <= value.item() <= 1.0

    value.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in loss.projector.parameters())
    assert all(p.grad is None for p in teacher.parameters())
    assert tokens.grad is not None  # the student stream is trained through it


def test_perfectly_aligned_tokens_score_minus_one(monkeypatch):
    """Sanity on the sign convention: the loss is a NEGATIVE cosine."""
    stub_hub(monkeypatch, 16)
    loss = REPALoss(tiny_cfg(variant="irepa"), student_dim=16)
    images = torch.randn(1, 3, 64, 64).clamp(-1.0, 1.0)
    with torch.no_grad():
        teacher_tokens = loss._teacher_tokens(images, (16, 16))
    aligned = _spatial_normalize(teacher_tokens, loss.cfg.spatial_norm_gamma)

    monkeypatch.setattr(loss, "_project", lambda t, _grid: t)
    assert loss(images, aligned, (16, 16)).item() == pytest.approx(-1.0, abs=1e-5)
    assert loss(images, -aligned, (16, 16)).item() == pytest.approx(1.0, abs=1e-5)


def test_irepa_actually_normalizes_the_teacher_and_repa_does_not(monkeypatch):
    """The two variants must not agree when only the teacher path differs."""
    torch.manual_seed(6)
    stub_hub(monkeypatch, 16)
    images = torch.randn(2, 3, 64, 64).clamp(-1.0, 1.0)
    tokens = torch.randn(2, 256, 16)

    seen: list[float] = []
    for variant in ("repa", "irepa"):
        loss = REPALoss(tiny_cfg(variant=variant), student_dim=16)
        monkeypatch.setattr(loss, "_project", lambda t, _grid: t)
        seen.append(loss(images, tokens, (16, 16)).item())
    assert seen[0] != pytest.approx(seen[1], abs=1e-4)


# --------------------------------------------------------------------------
# 5. equal student/teacher grids need no resampling
# --------------------------------------------------------------------------


def test_no_token_resize_when_both_grids_are_16x16(monkeypatch):
    """student 256/16 = 16, DINOv2 teacher 224/14 = 16 -> equal-grid branch."""
    stub_hub(monkeypatch, 16, patch=14)

    def forbidden(*a, **k):
        raise AssertionError("_resize_token_grid must not run on the 256px path")

    monkeypatch.setattr("iris3b.repa._resize_token_grid", forbidden)

    loss = REPALoss(tiny_cfg(variant="irepa"), student_dim=32)
    images = torch.randn(2, 3, 256, 256).clamp(-1.0, 1.0)
    with torch.no_grad():
        assert loss._teacher_tokens(images, (16, 16)).shape[1] == 256
    assert torch.isfinite(loss(images, torch.randn(2, 256, 32), (16, 16)))


def test_the_resize_guard_is_not_vacuous(monkeypatch):
    """Mismatched grids DO take the resize branch, so the no-op test has teeth."""
    import iris3b.repa as repa_mod

    stub_hub(monkeypatch, 16, patch=28)  # 224/28 -> 8x8 = 64 teacher tokens

    calls: list[tuple] = []
    real = repa_mod._resize_token_grid

    def spy(tokens, src, dst):
        calls.append((src, dst))
        return real(tokens, src, dst)

    monkeypatch.setattr(repa_mod, "_resize_token_grid", spy)

    loss = REPALoss(tiny_cfg(variant="irepa", teacher_patch_size=28), student_dim=32)
    images = torch.randn(2, 3, 64, 64).clamp(-1.0, 1.0)
    assert torch.isfinite(loss(images, torch.randn(2, 256, 32), (16, 16)))
    # the student grid was downsampled onto the teacher's
    assert calls == [((16, 16), (8, 8))]


def test_rectangular_student_resamples_on_its_own_grid(monkeypatch):
    """A 16x64 grid has 1024 tokens; a sqrt() would have called it 32x32."""
    import iris3b.repa as repa_mod

    stub_hub(monkeypatch, 16, patch=14)  # 224/14 -> 16x16 = 256 teacher tokens
    calls: list[tuple] = []
    real = repa_mod._resize_token_grid

    def spy(tokens, src, dst):
        calls.append((src, dst))
        return real(tokens, src, dst)

    monkeypatch.setattr(repa_mod, "_resize_token_grid", spy)
    loss = REPALoss(tiny_cfg(variant="repa"), student_dim=32)
    images = torch.randn(2, 3, 256, 1024).clamp(-1.0, 1.0)
    assert torch.isfinite(loss(images, torch.randn(2, 1024, 32), (16, 64)))
    assert calls == [((16, 64), (16, 16))]


def test_teacher_match_student_keeps_a_patch_level_target(monkeypatch):
    """With the flag on, no resampling happens at ANY student grid."""
    from dataclasses import replace

    stub_hub(monkeypatch, 16, patch=14)

    def forbidden(*a, **k):
        raise AssertionError("a matched teacher must never resample")

    monkeypatch.setattr("iris3b.repa._resize_token_grid", forbidden)
    cfg = replace(tiny_cfg(variant="repa"), teacher_match_student=True)
    loss = REPALoss(cfg, student_dim=32)
    images = torch.randn(2, 3, 512, 256).clamp(-1.0, 1.0)
    assert torch.isfinite(loss(images, torch.randn(2, 32 * 16, 32), (32, 16)))


# --------------------------------------------------------------------------
# 6. the real trainer path: captured student features must be a square grid
# --------------------------------------------------------------------------


def test_trainmodel_feeds_a_square_patch_grid_into_the_conv_projector(monkeypatch):
    """The one way iREPA could have died at step 0: non-square student tokens.

    ``TrainModel`` hands ``out.features[repa_layer]`` straight to the projector.
    If MM-DiT's captured stream carried the text tokens too, the count would not
    be square and the conv path would raise where the MLP path never would.
    """
    from iris3b.models import IrisDiT
    from iris3b.train.trainer import TrainModel
    from tiny_config import tiny_model_config

    torch.manual_seed(7)
    model_cfg = tiny_model_config(depth=4, patch_size=8, text_len=16, repa_layer=2)
    model = IrisDiT(model_cfg)

    stub_hub(monkeypatch, 16)
    repa = REPALoss(tiny_cfg(variant="irepa"), student_dim=model_cfg.hidden_size)
    train_model = TrainModel(model, repa)

    x = torch.randn(2, 3, 32, 32)  # 32/8 -> 4x4 = 16 patch tokens
    clean = x.clamp(-1.0, 1.0)
    t = torch.tensor([100.0, 400.0])
    y = torch.randn(2, model_cfg.text_len, model_cfg.text_dim)

    out = train_model(x, t, y, clean=clean)
    tokens = out.features[model_cfg.repa_layer]
    assert tokens.shape[:2] == (2, 16)  # image stream only, and square
    assert out.repa_loss is not None and torch.isfinite(out.repa_loss)
    assert -1.0 <= out.repa_loss.item() <= 1.0


# --------------------------------------------------------------------------
# 7. HF teacher source (DINOv3)
# --------------------------------------------------------------------------


def test_hf_teacher_source_loads_and_strips_cls_and_register_tokens(monkeypatch):
    calls = stub_hf(monkeypatch, 16, patch=16)
    loss = REPALoss(
        tiny_cfg(
            teacher_source="hf",
            teacher="facebook/dinov3-vitb16-pretrain-lvd1689m",
            teacher_image_size=256,
            teacher_patch_size=16,
        ),
        student_dim=32,
    )
    assert calls == ["facebook/dinov3-vitb16-pretrain-lvd1689m"]
    teacher = loss._teacher_holder[0]
    assert all(not p.requires_grad for p in teacher.parameters())

    images = torch.randn(2, 3, 64, 64).clamp(-1.0, 1.0)
    with torch.no_grad():
        tokens = loss._teacher_tokens(images, (16, 16))
    # 256/16 -> 16x16 patch grid; CLS + 4 registers sliced off
    assert tokens.shape == (2, 256, 16)
    assert torch.isfinite(loss(images, torch.randn(2, 256, 32), (16, 16)))


def test_unknown_teacher_source_raises_before_any_load(monkeypatch):
    calls = stub_hub(monkeypatch, 16)
    with pytest.raises(ValueError, match="unknown repa teacher_source 'huggingface'"):
        REPALoss(tiny_cfg(teacher_source="huggingface"), student_dim=32)
    assert calls == []


def test_hf_teacher_failure_disables_the_loss(monkeypatch):
    import transformers

    def boom(repo, **kwargs):
        raise RuntimeError("gated")

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", staticmethod(boom))
    loss = REPALoss(tiny_cfg(teacher_source="hf"), student_dim=32)
    assert loss.available is False and loss._teacher_holder == ()
