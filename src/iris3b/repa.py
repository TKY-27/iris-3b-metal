"""Representation-alignment auxiliary loss against a frozen vision teacher."""

import logging

import torch
import torch.nn.functional as F
from torch import nn

from iris3b.config import RepaConfig

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _resize_token_grid(tokens: torch.Tensor, src: tuple[int, int], dst: tuple[int, int]) -> torch.Tensor:
    """Bilinearly resample a [B, src_h*src_w, C] token map onto the ``dst`` grid."""
    b, n, c = tokens.shape
    if src[0] * src[1] != n:
        raise ValueError(f"token count {n} does not match grid {src[0]}x{src[1]}")
    x = tokens.permute(0, 2, 1).reshape(b, c, src[0], src[1])
    x = F.interpolate(x, size=dst, mode="bilinear", align_corners=False)
    return x.flatten(2).permute(0, 2, 1)


def _spatial_normalize(tokens: torch.Tensor, gamma: float, eps: float = 1e-6) -> torch.Tensor:
    """Suppress the global component of a [B, N, D] token map (iREPA).

    ``y = (x - gamma * E_N[x]) / sqrt(Var_N[x] + eps)`` — mean/variance over
    the token axis, per channel; amplifies patch-to-patch contrast.
    """
    mean = tokens.mean(dim=1, keepdim=True)
    var = tokens.var(dim=1, keepdim=True, unbiased=False)
    return (tokens - gamma * mean) / torch.sqrt(var + eps)


class REPALoss(nn.Module):
    """Negative-cosine alignment of projected student tokens to teacher patch tokens.

    The teacher is loaded from torch hub, frozen, and held outside the module
    tree so ``projector`` contributes the only trainable parameters. If the
    teacher cannot be loaded, ``available`` is False and callers skip the loss.
    """

    def __init__(self, cfg: RepaConfig, student_dim: int):
        super().__init__()
        self.cfg = cfg
        if cfg.variant == "irepa":
            self.projector = nn.Conv2d(student_dim, cfg.teacher_dim, kernel_size=3, padding=1)
        elif cfg.variant == "repa":
            self.projector = nn.Sequential(
                nn.Linear(student_dim, cfg.proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.proj_hidden_dim, cfg.proj_hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.proj_hidden_dim, cfg.teacher_dim),
            )
        else:
            raise ValueError(f"unknown repa variant '{cfg.variant}'")
        if cfg.teacher_source not in {"hub", "hf"}:
            raise ValueError(f"unknown repa teacher_source '{cfg.teacher_source}'")
        self.available = True
        self._teacher_holder: tuple[nn.Module, ...] = ()
        try:
            if cfg.teacher_source == "hf":
                from transformers import AutoModel

                teacher = AutoModel.from_pretrained(cfg.teacher)
            else:
                kwargs = {"weights": cfg.teacher_weights} if cfg.teacher_weights else {}
                teacher = torch.hub.load(cfg.teacher_hub, cfg.teacher, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "teacher '%s/%s' unavailable, alignment loss disabled: %s",
                cfg.teacher_hub,
                cfg.teacher,
                exc,
            )
            self.available = False
        else:
            teacher.eval().requires_grad_(False)
            self._teacher_holder = (teacher,)

    def _teacher_grid(self, student_grid: tuple[int, int]) -> tuple[int, int]:
        """Teacher input size in pixels, and the patch grid it produces.

        ``teacher_image_size`` is a square side tuned for the 256px student
        (224/14 = 16 = 256/16). ``teacher_match_student`` instead sizes the
        teacher so its grid equals the student's, which is what keeps REPA a
        patch-level target at 512/1024 and under non-square shapes; without it
        the student is pooled down to the teacher's fixed square grid and the
        alignment signal is diluted by the area ratio.
        """
        if not self.cfg.teacher_match_student:
            side = self.cfg.teacher_image_size // self.cfg.teacher_patch_size
            return side, side
        return student_grid

    def _teacher_tokens(self, images: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        """Featurize clean [-1, 1] images into fp32 patch tokens on ``grid``."""
        teacher = self._teacher_holder[0]
        if next(teacher.parameters()).device != images.device:
            teacher.to(images.device)
        patch = self.cfg.teacher_patch_size
        x = (images.float() + 1.0) / 2.0
        x = F.interpolate(x, size=(grid[0] * patch, grid[1] * patch), mode="bicubic", align_corners=False)
        mean = x.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = x.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        x = (x - mean) / std
        if self.cfg.teacher_source == "hf":
            out = teacher(pixel_values=x).last_hidden_state
            registers = int(getattr(teacher.config, "num_register_tokens", 0))
            return out[:, 1 + registers :]
        feats = teacher.forward_features(x)
        if isinstance(feats, dict):
            if "x_norm_patchtokens" in feats:
                return feats["x_norm_patchtokens"]
            return feats["tokens"][:, 1:]
        return feats[:, 1:]

    def _project(self, tokens: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        """Map [B, N, C] student tokens to teacher dim (conv variant on the grid)."""
        if self.cfg.variant == "irepa":
            b, n, c = tokens.shape
            if grid[0] * grid[1] != n:
                raise ValueError(f"token count {n} does not match student grid {grid[0]}x{grid[1]}")
            x = tokens.permute(0, 2, 1).reshape(b, c, grid[0], grid[1])
            return self.projector(x).flatten(2).permute(0, 2, 1)
        return self.projector(tokens)

    def forward(
        self,
        clean_images: torch.Tensor,
        student_tokens: torch.Tensor,
        student_grid: tuple[int, int],
    ) -> torch.Tensor:
        """``student_grid`` is the (rows, cols) patch grid of ``student_tokens``.

        It is passed explicitly rather than inferred from the token count: a
        sqrt() inference is not merely square-only, it is silently WRONG for the
        rectangles whose token count happens to be a perfect square (a 16x64
        grid has 1024 tokens and would be reshaped as 32x32, scrambling rows).
        """
        assert self.available, "alignment teacher failed to load"
        if student_grid[0] * student_grid[1] != student_tokens.shape[1]:
            raise ValueError(
                f"student grid {student_grid[0]}x{student_grid[1]} does not match "
                f"{student_tokens.shape[1]} tokens"
            )
        teacher_grid = self._teacher_grid(student_grid)
        with torch.no_grad():
            teacher_tokens = self._teacher_tokens(clean_images, teacher_grid)
            if teacher_tokens.shape[1] != teacher_grid[0] * teacher_grid[1]:
                raise ValueError(
                    f"teacher returned {teacher_tokens.shape[1]} tokens but "
                    f"repa.teacher_patch_size={self.cfg.teacher_patch_size} implies "
                    f"{teacher_grid[0]}x{teacher_grid[1]}"
                )
            if self.cfg.variant == "irepa":
                teacher_tokens = _spatial_normalize(teacher_tokens, self.cfg.spatial_norm_gamma)

        # The projector runs in its own parameter dtype and the cosine is taken in
        # fp32. Forcing fp32 activations here instead would crash under fsdp2,
        # whose MixedPrecisionPolicy casts these weights to bf16.
        weight_dtype = next(self.projector.parameters()).dtype
        student = F.normalize(self._project(student_tokens.to(weight_dtype), student_grid).float(), dim=-1)
        if teacher_grid != student_grid:
            # upsampling the smaller grid would fabricate detail, so meet on
            # whichever grid has fewer tokens
            if teacher_tokens.shape[1] > student.shape[1]:
                teacher_tokens = _resize_token_grid(teacher_tokens, teacher_grid, student_grid)
            else:
                student = _resize_token_grid(student, student_grid, teacher_grid)
        teacher = F.normalize(teacher_tokens, dim=-1)
        student = F.normalize(student, dim=-1)
        return -(student * teacher).sum(dim=-1).mean()
