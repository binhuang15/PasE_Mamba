import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import digamma, gammaln

def dirichlet_from_evidence(evidence: torch.Tensor, eps: float = 1e-8):
    return evidence + 1.0

def dirichlet_prob_and_uncertainty(alpha: torch.Tensor, eps: float = 1e-8):
    S = alpha.sum(dim=1, keepdim=True).clamp_min(eps)
    prob = alpha / S
    C = alpha.size(1)
    u = (C / S)
    return prob, u


def dpe_easf_routing_uncertainty(
    alpha: torch.Tensor,
    prior_logits: torch.Tensor,
    num_classes: int,
    conflict_weight: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    **DPE (Decoupled Prediction-Evidence) routing field** u ∈ [B,1,H,W]: per-sample min–max maps for **EASF**
    (Evidence-driven Anisotropic Scan Fusion) gating.

    - **Vacuity**: Dirichlet epistemic uncertainty K/S (matches Phase-1 u).
    - **Conflict**: half the total-variation distance between prior softmax(prior) and EDL expectation p = α/S,
      in [0,1], emphasizing prior–evidence disagreement (PasE-specific; vanilla EDL segmentation omits this branch).

    Fused as ``(1-w) * norm(vacuity) + w * norm(conflict)``, then clamped; no extra trainable parameters.
    """
    S = alpha.sum(dim=1, keepdim=True).clamp_min(eps)
    u_v = float(num_classes) / S
    p_edl = alpha / S
    p_pr = torch.softmax(prior_logits.detach(), dim=1)
    conflict = (p_pr - p_edl).abs().sum(dim=1, keepdim=True) * 0.5

    def _minmax(t: torch.Tensor) -> torch.Tensor:
        tmin = t.amin(dim=(-2, -1), keepdim=True)
        tmax = t.amax(dim=(-2, -1), keepdim=True)
        return (t - tmin) / (tmax - tmin + eps)

    u_n = _minmax(u_v)
    c_n = _minmax(conflict)
    w = float(conflict_weight)
    w = max(0.0, min(1.0, w))
    fused = (1.0 - w) * u_n + w * c_n
    return fused.clamp(0.0, 1.0)

def edl_nll_kl_loss(alpha: torch.Tensor, target: torch.Tensor, kl_weight: float = 1e-3, eps: float = 1e-8):
    N, C, H, W = alpha.shape
    S = alpha.sum(dim=1, keepdim=True).clamp_min(eps)
    prob = alpha / S

    if target.dim() == 3:
        y = F.one_hot(target, num_classes=C).permute(0,3,1,2).float()
    else:
        y = target.float()

    nll_map = - (y * (digamma(alpha.clamp_min(eps)) - digamma(S))).sum(dim=1, keepdim=True)
    logB_alpha = gammaln(alpha.clamp_min(eps)).sum(dim=1, keepdim=True) - gammaln(S)
    kl_map = - logB_alpha + ((alpha - 1.0) * (digamma(alpha.clamp_min(eps)) - digamma(S))).sum(dim=1, keepdim=True)

    loss = nll_map.mean() + kl_weight * kl_map.mean()
    return loss, prob

def dice_ce_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0):
    N, C, H, W = logits.shape
    ce = F.cross_entropy(logits, target, reduction='mean')
    prob = torch.softmax(logits, dim=1)
    target_oh = F.one_hot(target, num_classes=C).permute(0,3,1,2).float()

    inter = (prob * target_oh).sum(dim=(0,2,3))
    den = (prob + target_oh).sum(dim=(0,2,3))
    dice = (2*inter + smooth) / (den + smooth)
    dice_loss = 1.0 - dice.mean()
    return ce + dice_loss


def multiclass_dice_ce_weighted(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    ignore_index: int = -100,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    ``target``: [B, H, W] class indices; ``logits``: [B, C, H, W].
    Cross-entropy with optional per-class weights; Dice term is a class-weighted mean of soft Dice (valid pixels only).
    """
    N, C, H, W = logits.shape
    w_ce = class_weights.to(logits.device) if class_weights is not None else None
    t = target.long()
    ce = F.cross_entropy(
        logits,
        t,
        weight=w_ce,
        ignore_index=ignore_index,
        reduction="mean",
    )
    prob = torch.softmax(logits, dim=1)
    valid = t != ignore_index
    if not valid.any():
        return ce
    t_safe = t.clamp(0, C - 1)
    target_oh = F.one_hot(t_safe, num_classes=C).permute(0, 3, 1, 2).float()
    vf = valid.unsqueeze(1).float()
    target_oh = target_oh * vf
    prob = prob * vf
    inter = (prob * target_oh).sum(dim=(0, 2, 3))
    den = (prob + target_oh).sum(dim=(0, 2, 3))
    dice = (2 * inter + smooth) / (den + smooth)
    if class_weights is None:
        dice_loss = 1.0 - dice.mean()
    else:
        w = class_weights.to(logits.device).clamp_min(0.0)
        w = w / w.sum().clamp_min(1e-8)
        dice_loss = 1.0 - (dice * w).sum()
    return ce + dice_loss

def kl_consistency(p_edl: torch.Tensor, p_seg: torch.Tensor, reduction: str = "mean", eps: float = 1e-8, symmetric: bool = True):
    p_edl = p_edl.clamp_min(eps)
    p_seg = p_seg.clamp_min(eps)
    kl1 = (p_edl * (p_edl.log() - p_seg.log())).sum(dim=1, keepdim=True)
    if symmetric:
        kl2 = (p_seg * (p_seg.log() - p_edl.log())).sum(dim=1, keepdim=True)
        kl = kl1 + kl2
    else:
        kl = kl1
    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    else:
        return kl


def _zero_shot_easf_legacy(
    Y_dirs: torch.Tensor,
    uncertainty: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Legacy EASF (Evidence-driven Anisotropic Scan Fusion): gradient-sharpness softmax; blend with uniform 0.25 via DPE u."""
    dx = torch.abs(Y_dirs[:, :, :, :, 1:] - Y_dirs[:, :, :, :, :-1])
    dx = F.pad(dx, (1, 0, 0, 0))
    dy = torch.abs(Y_dirs[:, :, :, 1:, :] - Y_dirs[:, :, :, :-1, :])
    dy = F.pad(dy, (0, 0, 1, 0))
    sharpness = torch.mean(dx + dy, dim=2, keepdim=True)
    heuristic_weights = F.softmax(sharpness / max(temperature, 1e-6), dim=1)
    u_expanded = uncertainty.unsqueeze(1)
    uniform_weights = torch.full_like(heuristic_weights, 0.25)
    final_weights = (1.0 - u_expanded) * uniform_weights + u_expanded * heuristic_weights
    return torch.sum(Y_dirs * final_weights, dim=1)


def _zero_shot_easf_v2(
    Y_dirs: torch.Tensor,
    uncertainty: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Training-free EASF v2 (Evidence-driven Anisotropic Scan Fusion): modulate sharpness by **four-way agreement** (lower variance → stronger anisotropic trust);
    add a **uniform floor** after softmax to avoid collapse; apply **gamma compression** to u plus optional
    **global high-uncertainty dampening** (reduce heuristic reliance when the whole field is uncertain / OOD).

    Optional environment knobs: ``EASF_AGREE_SCALE``, ``EASF_WEIGHT_FLOOR``, ``EASF_U_BLEND_GAMMA``,
    ``EASF_U_GLOBAL_DAMPEN``, ``EASF_GLOBAL_DAMPEN_K``, ``EASF_GLOBAL_DAMPEN_THR``.
    """
    dx = torch.abs(Y_dirs[:, :, :, :, 1:] - Y_dirs[:, :, :, :, :-1])
    dx = F.pad(dx, (1, 0, 0, 0))
    dy = torch.abs(Y_dirs[:, :, :, 1:, :] - Y_dirs[:, :, :, :-1, :])
    dy = F.pad(dy, (0, 0, 1, 0))
    sharpness = torch.mean(dx + dy, dim=2, keepdim=True)

    var_map = Y_dirs.var(dim=1, correction=False).mean(dim=1, keepdim=True).clamp_min(1e-8)
    try:
        vscale = float(os.environ.get("EASF_AGREE_SCALE", "1.0"))
    except ValueError:
        vscale = 1.0
    agree = 1.0 / (1.0 + vscale * var_map)
    score = sharpness * agree.unsqueeze(1)

    temp = max(float(temperature), 1e-6)
    h = F.softmax(score / temp, dim=1)
    try:
        floor = float(os.environ.get("EASF_WEIGHT_FLOOR", "0.07"))
    except ValueError:
        floor = 0.07
    floor = max(0.0, min(0.22, floor))
    h = (1.0 - floor) * h + floor * 0.25

    u = uncertainty.unsqueeze(1).clamp(0.0, 1.0)
    try:
        gam = float(os.environ.get("EASF_U_BLEND_GAMMA", "0.9"))
    except ValueError:
        gam = 0.9
    u_eff = u.pow(max(gam, 1e-3))

    if os.environ.get("EASF_U_GLOBAL_DAMPEN", "1").lower() in ("1", "true", "yes"):
        g = u.mean(dim=(-2, -1), keepdim=True)
        try:
            k = float(os.environ.get("EASF_GLOBAL_DAMPEN_K", "1.5"))
            thr = float(os.environ.get("EASF_GLOBAL_DAMPEN_THR", "0.38"))
        except ValueError:
            k, thr = 1.5, 0.38
        factor = 1.0 / (1.0 + k * (g - thr).clamp(min=0.0))
        u_eff = (u_eff * factor).clamp(0.0, 1.0)

    u_eff = u_eff.to(dtype=h.dtype)
    final_weights = (1.0 - u_eff) * 0.25 + u_eff * h
    return torch.sum(Y_dirs * final_weights, dim=1)


def zero_shot_easf(
    Y_dirs: torch.Tensor,
    uncertainty: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Four-way SS2D feature fusion (**EASF**: Evidence-driven Anisotropic Scan Fusion; no learned parameters). Default ``EASF_FUSION_VARIANT=v2``; ``legacy`` matches the original variant.
    """
    variant = os.environ.get("EASF_FUSION_VARIANT", "v2").strip().lower()
    if variant in ("legacy", "v0", "v1", "old", "0"):
        return _zero_shot_easf_legacy(Y_dirs, uncertainty, temperature)
    return _zero_shot_easf_v2(Y_dirs, uncertainty, temperature)
