# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_sys import VSSM as VSSM_with_runtime_attn

try:
    from edl_utils import dirichlet_from_evidence, dirichlet_prob_and_uncertainty
except Exception:
    def dirichlet_from_evidence(evidence, eps: float = 1e-8):
        return evidence + 1.0

    def dirichlet_prob_and_uncertainty(alpha, eps: float = 1e-8):
        S = alpha.sum(dim=1, keepdim=True).clamp_min(eps)
        prob = alpha / S
        C = alpha.size(1)
        u = C / S
        return prob, u


class UnifiedSegEDLHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, mid_ch: int = None):
        super().__init__()
        if mid_ch is None:
            mid_ch = max(32, in_ch // 2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.seg_out = nn.Conv2d(mid_ch, num_classes, kernel_size=1, bias=True)
        self.edl_ev = nn.Sequential(
            nn.Conv2d(mid_ch, num_classes, kernel_size=1, bias=True),
            nn.Softplus(),
        )

    def forward(self, feat: torch.Tensor):
        f = self.stem(feat)
        logits = self.seg_out(f)
        evidence = self.edl_ev(f)
        alpha = dirichlet_from_evidence(evidence)
        return logits, alpha


class PasEMamba(nn.Module):
    """PasE-Mamba: U-Mamba-style encoder–decoder, dual heads (segmentation logits + EDL), with DPE (Decoupled Prediction-Evidence) and EASF (Evidence-driven Anisotropic Scan Fusion) in the SS2D path."""

    def __init__(self, num_classes: int, config):
        super().__init__()
        self.num_classes = num_classes
        _vssm_embed = int(
            config.MODEL["VSSM"]["EMBED_DIM"]
            if "EMBED_DIM" in config.MODEL["VSSM"]
            else config.MODEL["VSSM"].get("DIMS", [64])[0]
        )
        self.backbone = VSSM_with_runtime_attn(
            patch_size=config.MODEL["VSSM"]["PATCH_SIZE"],
            in_chans=config.MODEL["VSSM"]["IN_CHANS"],
            num_classes=_vssm_embed,
            dims=config.MODEL["VSSM"].get(
                "DIMS",
                [_vssm_embed * (2**i) for i in range(len(config.MODEL["VSSM"]["DEPTHS"]))],
            ),
            embed_dim=_vssm_embed,
            depths=config.MODEL["VSSM"]["DEPTHS"],
            d_state=config.MODEL["VSSM"].get("D_STATE", 16),
            drop_rate=config.MODEL.get("DROP_RATE", 0.0),
            drop_path_rate=config.MODEL.get("DROP_PATH_RATE", 0.1),
            patch_norm=config.MODEL["SWIN"].get("PATCH_NORM", True),
            use_checkpoint=config.TRAIN.get("USE_CHECKPOINT", False),
        )

        # Final conv after ``up_x4`` has out_channels == backbone ``num_classes`` (here ``_vssm_embed``)
        in_ch = self.backbone.num_classes
        self.head = UnifiedSegEDLHead(in_ch=in_ch, num_classes=num_classes)
        dec_ch = getattr(self.backbone, "embed_dim", _vssm_embed)
        self.dec_prob_proj = nn.Conv2d(num_classes, dec_ch, kernel_size=1, bias=True)
        nn.init.zeros_(self.dec_prob_proj.weight)
        nn.init.zeros_(self.dec_prob_proj.bias)

    def _inject_prob_decoder(self, x_dec: torch.Tensor, prob_prior: torch.Tensor) -> torch.Tensor:
        """x_dec: [B,H,W,C]; inject first-pass softmax class probabilities (residual) before the segmentation head."""
        B, H, W, C = x_dec.shape
        if prob_prior.shape[-2:] != (H, W):
            prob_prior = F.interpolate(prob_prior, size=(H, W), mode="bilinear", align_corners=False)
        x_bchw = x_dec.permute(0, 3, 1, 2).contiguous()
        x_bchw = x_bchw + self.dec_prob_proj(prob_prior)
        return x_bchw.permute(0, 2, 3, 1).contiguous()

    def _forward_once(self, x: torch.Tensor, prob_prior: torch.Tensor = None):
        """One encoder–decoder pass + EDL head; ``prob_prior`` is fed only on the refined (second) pass."""
        x_enc, x_skips = self.backbone.forward_features(x)
        x_dec = self.backbone.forward_up_features(x_enc, x_skips)
        if prob_prior is not None:
            x_dec = self._inject_prob_decoder(x_dec, prob_prior)
        feat_full = self.backbone.up_x4(x_dec)

        logits, alpha = self.head(feat_full)
        prob_seg = torch.softmax(logits, dim=1)
        prob_edl, u_map = dirichlet_prob_and_uncertainty(alpha)
        return dict(logits=logits, prob_seg=prob_seg, alpha=alpha, prob_edl=prob_edl, u_map=u_map)

    def forward(self, x: torch.Tensor, refine_with_prob: bool = False):
        if hasattr(self.backbone, "_runtime_attn"):
            self.backbone._runtime_attn = None
        if hasattr(self.backbone, "_fusion_u"):
            self.backbone._fusion_u = None

        out1 = self._forward_once(x)

        if refine_with_prob:
            if hasattr(self.backbone, "set_runtime_attn"):
                self.backbone.set_runtime_attn(out1["u_map"])
            prob_prior = out1["prob_seg"].detach()
            out2 = self._forward_once(x, prob_prior=prob_prior)
            if hasattr(self.backbone, "set_runtime_attn"):
                self.backbone.set_runtime_attn(out2["u_map"])
            out2["logits_first"] = out1["logits"]
            out2["prob_seg_first"] = out1["prob_seg"]
            out2["u_map_first"] = out1["u_map"]
            out2["alpha_first"] = out1["alpha"]
            return out2

        if hasattr(self.backbone, "set_runtime_attn"):
            self.backbone.set_runtime_attn(out1["u_map"])
        return out1
