import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from vision_mamba import EGAMamba
from train import EGA_Mamba_Config, set_anisotropic_fusion
from datagenerator import MyDatasetLoader
from augmentation_strategies import get_train_transform_2D
from data_paths import resolve_best_model_pt, resolve_validation_npy_path
from PIL import Image

# -------------------------- 1. Metric helpers --------------------------
def dsc_calc(labels, outputs):
    """Match ``train.dsc_calc``: softmax, ``torch.round`` on class probabilities, then soft Dice."""
    outputs = F.softmax(outputs, dim=1)
    outputs = torch.round(outputs)
    labels_one_hot = F.one_hot(labels[:, 0, :, :].long(), num_classes=3).float()
    labels_one_hot = labels_one_hot.permute([0, 3, 1, 2])
    dsc_list = []
    for i in range(3):
        dsc_i = 2 * (torch.sum(outputs[:, i, :, :] * labels_one_hot[:, i, :, :])) / (
                    torch.sum(outputs[:, i, :, :]) + torch.sum(labels_one_hot[:, i, :, :]) + 1e-8)
        dsc_list.append(dsc_i.item())

    return dsc_list[1], dsc_list[2]

def iou_calc(labels, outputs):
    """Per-case IoU with a differentiable-safe union (no bitwise OR on floats)."""
    outputs = F.softmax(outputs, dim=1)
    preds_oh = torch.round(outputs)
    labels = labels.squeeze(1)  # [B,H,W]

    labels_oh = torch.nn.functional.one_hot(labels.long(), num_classes=3).permute(0, 3, 1, 2).float()

    def iou(c):
        inter = torch.sum(preds_oh[:, c] * labels_oh[:, c])
        # Union via min(sum, 1) per pixel (float-safe); tensor lives on preds_oh.device
        union = torch.sum(torch.min(preds_oh[:, c] + labels_oh[:, c], torch.tensor(1.0, device=preds_oh.device))) + 1e-8
        return (inter / union).item()

    return iou(1), iou(2)  # Placenta IoU, Myometrium IoU


def hd95_calc(labels, outputs, spacing=(1.0, 1.0)):
    """95% HD"""
    from scipy.ndimage import distance_transform_edt

    outputs = F.softmax(outputs, dim=1)
    preds = torch.round(outputs).cpu().numpy()
    labels = labels.squeeze(1)  # [B,H,W]
    labels = torch.nn.functional.one_hot(labels.long(), num_classes=3).permute(0, 3, 1, 2).float().cpu().numpy()

    def single_hd95(pred, gt):
        if np.sum(pred) == 0 or np.sum(gt) == 0:
            return np.nan
        gt_dist = distance_transform_edt(1 - gt)
        pred_dist = distance_transform_edt(1 - pred)
        dists = np.concatenate([gt_dist[pred > 0], pred_dist[gt > 0]])
        return np.percentile(dists, 95) * spacing[0]

    return single_hd95(preds[:,1], labels[:,1]), single_hd95(preds[:,2], labels[:,2])


def nsd_calc(labels, outputs, spacing=(1.0, 1.0), tau=2.0):
    """Normalized surface distance (NSD)."""
    from scipy.signal import convolve2d
    from scipy.spatial import cKDTree

    outputs = F.softmax(outputs, dim=1)
    preds = torch.round(outputs).cpu().numpy()
    labels = labels.squeeze(1)  # [B,H,W]
    labels = torch.nn.functional.one_hot(labels.long(), num_classes=3).permute(0, 3, 1, 2).float().cpu().numpy()

    def get_surface(mask):
        kernel = np.ones((3, 3))
        nb = convolve2d(mask, kernel, mode="same", boundary="symm")
        return (mask == 1) & (nb < 9)

    def single_nsd(pred, gt):
        if np.sum(pred) == 0 or np.sum(gt) == 0:
            return np.nan
        gt_surf = get_surface(gt)
        pred_surf = get_surface(pred)
        gt_coords = np.argwhere(gt_surf) * np.array(spacing)
        pred_coords = np.argwhere(pred_surf) * np.array(spacing)
        if len(gt_coords) == 0 or len(pred_coords) == 0:
            return np.nan
        gt_tree = cKDTree(gt_coords)
        pred2gt_dist, _ = gt_tree.query(pred_coords, k=1)
        pred_overlap = np.sum(pred2gt_dist <= tau)
        pred_tree = cKDTree(pred_coords)
        gt2pred_dist, _ = pred_tree.query(gt_coords, k=1)
        gt_overlap = np.sum(gt2pred_dist <= tau)
        return 2 * (pred_overlap + gt_overlap) / (len(pred_coords) + len(gt_coords))

    return single_nsd(preds[0,1], labels[0,1]), single_nsd(preds[0,2], labels[0,2])


METRIC_COLUMNS = [
    "PlacentaDice",
    "PlacentaIoU",
    "PlacentaHD95(mm)",
    "PlacentaNSD",
    "MyometriumDice",
    "MyometriumIoU",
    "MyometriumHD95(mm)",
    "MyometriumNSD",
]
CASE_HEADER = ["Patient", "FileName", "CaseId", "Group"]

# Case-level export subfolders (must match processed layout: 1-Normal / 2-PAS)
CASE_FOLDER_BY_CLINICAL_LABEL = {1: "1-Normal", 2: "2-PAS", 3: "2-PAS"}

# ---------------------------------------------------------------------------
# Frozen EGA-Mamba inference env (EASF hyperparameters read inside ``mamba_sys``).
# Default ``eval.py`` protocol: first forward → EDL u_map; second forward → EASF (train stays single-pass / no EASF).
# ---------------------------------------------------------------------------
FINAL_MYOMETRIUM_LOGIT_BIAS = 0.14

# Canonical paths under repo root (``python eval.py`` with no args uses these).
DEFAULT_DATA_ROOT = os.path.join("Training_Evaluation_Data", "npy_eval_merged")
DEFAULT_PWD_PATH = os.path.join("Training_Evaluation_Data", "processed_eval_merged")
DEFAULT_MODEL_ROOT = "TrainedCheckpoints"
DEFAULT_RESULTS_ROOT = "Results"

# Strings for os.environ — must match edl_utils / mamba_sys readers.
_FROZEN_EVAL_ENV: dict[str, str] = {
    "EASF_SHARPNESS_TEMP": "2.0",
    "EASF_FUSION_VARIANT": "v2",
    "EASF_AGREE_SCALE": "1.0",
    "EASF_WEIGHT_FLOOR": "0.07",
    "EASF_U_BLEND_GAMMA": "0.9",
    "EASF_U_GLOBAL_DAMPEN": "1",
    "EASF_GLOBAL_DAMPEN_K": "1.5",
    "EASF_GLOBAL_DAMPEN_THR": "0.38",
    "EASF_FUSION_ENCODER": "0",
    "EASF_FEATURE_GATE_STRENGTH": "0.22",
}


def freeze_eval_environment() -> None:
    """Overwrite EASF-related env so inference never depends on the caller's shell."""
    for k, v in _FROZEN_EVAL_ENV.items():
        os.environ[k] = v


SUBGROUP_METRIC_KEYS = [
    ("PlacentaDice", "PlacentaDice_mean", "PlacentaDice_std"),
    ("PlacentaHD95(mm)", "PlacentaHD95_mean", "PlacentaHD95_std"),
    ("MyometriumDice", "MyometriumDice_mean", "MyometriumDice_std"),
    ("MyometriumHD95(mm)", "MyometriumHD95_mean", "MyometriumHD95_std"),
]


def _fmt_metric_scalar(x) -> str:
    if isinstance(x, str):
        return x
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if np.isnan(xf):
        return "NaN"
    return f"{xf:.4f}"


def apply_final_eval_logits(logits: torch.Tensor) -> torch.Tensor:
    """Fixed post-hoc adjustment on refined logits: additive bias on myometrium (class 2) only."""
    z = logits.clone()
    z[:, 2:3, :, :] = z[:, 2:3, :, :] + float(FINAL_MYOMETRIUM_LOGIT_BIAS)
    return z


def format_final_inference_summary(
    *,
    use_anisotropic_fusion: bool,
    use_refine_with_prob: bool,
    use_edl_u_second_pass: bool,
) -> str:
    """Log frozen EASF env + runtime path + logits bias."""
    parts = [f"{k}={v}" for k, v in sorted(_FROZEN_EVAL_ENV.items())]
    parts.append(f"myometrium_logit_bias={float(FINAL_MYOMETRIUM_LOGIT_BIAS):.4f}")
    if use_refine_with_prob:
        path = "dec_prob_refine(Egamamba.forward)" + ("+EASF" if use_anisotropic_fusion else "_no_EASF")
    elif use_edl_u_second_pass and use_anisotropic_fusion:
        path = "two_pass_first_u_then_EASF"
    elif use_edl_u_second_pass:
        path = "two_pass_first_u_no_EASF"
    else:
        path = "single_pass_no_EASF"
    parts.append(f"path={path}")
    return "[Eval] Inference | " + " | ".join(parts)


def _mean_std_numeric(series: pd.Series) -> tuple[float, float]:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return m, s


def save_fold_subgroup_metrics_summary_csv(
    df: pd.DataFrame,
    *,
    suite_slug: str,
    suite_display: str,
    out_dir: str,
) -> str | None:
    """Normal vs PAS (+ All): Placenta / Myometrium Dice and HD95 mean±std for papers."""
    path = os.path.join(out_dir, "validation_subgroup_metrics_summary.csv")
    group_col = "Group"
    subsets = [
        ("All cases", pd.Series(True, index=df.index)),
        ("Normal", df[group_col] == "Normal"),
        ("PAS", df[group_col] == "PAS"),
    ]
    rows: list[dict] = []
    for subset_en, mask in subsets:
        sub = df.loc[mask]
        n = int(mask.sum())
        row: dict = {
            "suite_slug": suite_slug or "",
            "suite_display": suite_display or "",
            "subset": subset_en,
            "n_samples": n,
        }
        if n == 0:
            for _, mk_mean, mk_std in SUBGROUP_METRIC_KEYS:
                row[mk_mean] = float("nan")
                row[mk_std] = float("nan")
            rows.append(row)
            continue
        for col_src, mk_mean, mk_std in SUBGROUP_METRIC_KEYS:
            m, s = _mean_std_numeric(sub[col_src])
            row[mk_mean] = m
            row[mk_std] = s
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"[Eval] Subgroup metrics CSV (Dice + HD95): {path}")
    return path


def resolve_mamba_edl_ckpt_path(config: dict) -> str:
    """
    If ``model_ckpt`` is set, use that path (``{run_index}`` → 0).
    Otherwise resolve ``Best_model.pt`` under ``model_save_root`` (flat or legacy ``0/``).
    """
    explicit = (config.get("model_ckpt") or "").strip()
    if explicit:
        if "{run_index}" in explicit:
            return explicit.format(run_index=0)
        return explicit
    return resolve_best_model_pt(config["model_save_root"])


def print_mamba_edl_weight_fingerprint(model: torch.nn.Module, *, tag: str = "") -> None:
    """
    For full-model pickles: verify parameters deserialized intact (not filtered by ``load_state_dict``).
    Prints tensor count, element count, and L1 mass for log reconciliation; if fingerprint matches but
    metrics differ, suspect forward/metric code rather than missing weights.
    """
    with torch.no_grad():
        total_elems = 0
        l1_sum = 0.0
        n_param = 0
        for p in model.parameters():
            total_elems += p.numel()
            l1_sum += float(p.detach().float().abs().sum().cpu())
            n_param += 1
    prefix = f"[Eval] weight_fingerprint{tag}"
    print(f"{prefix}: parameter_tensors={n_param}, total_elements={total_elems:,}, sum(abs)= {l1_sum:.6e}")


def patch_mamba_edl_legacy_modules(model: torch.nn.Module) -> None:
    """
    Older ``torch.save(model)`` blobs may omit fields added after training; patch attributes so
    ``mamba_sys`` / ``mamba_sys_legacy`` forward (DPE / EASF hooks) does not raise ``AttributeError``.
    """
    bb = getattr(model, "backbone", None)
    if bb is not None:
        if not hasattr(bb, "_fusion_u"):
            bb._fusion_u = None
        if not hasattr(bb, "_runtime_attn"):
            bb._runtime_attn = None
        if not hasattr(bb, "_route_weights_acc"):
            bb._route_weights_acc = []
        if not hasattr(bb, "_easf_fuse_in_encoder"):
            bb._easf_fuse_in_encoder = False

    for m in model.modules():
        if m.__class__.__name__ == "SS2D":
            if not hasattr(m, "num_directions"):
                m.num_directions = 4
            if not hasattr(m, "use_anisotropic_fusion"):
                m.use_anisotropic_fusion = True


def load_mamba_edl_for_eval(
    model_path: str,
    config_mamba,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load ``torch.save(model)`` full module, plain ``state_dict``, or ``{\"state_dict\": ...}`` as saved by ``train.py``.
    """
    print(f"[Eval] loading: {model_path}")
    blob = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(blob, torch.nn.Module):
        # Full pickle restores Parameters/Buffers verbatim; do not rebuild from config then strict load (shape mismatch).
        model = blob
        print(
            "[Eval] loaded pickled nn.Module (torch.save(model)); "
            "weights=full in-file tensors, not filtered by load_state_dict"
        )
        patch_mamba_edl_legacy_modules(model)
        print_mamba_edl_weight_fingerprint(model, tag="(pickled_module)")
    elif isinstance(blob, dict):
        if "state_dict" in blob and isinstance(blob["state_dict"], dict):
            sd = blob["state_dict"]
        else:
            sd = blob
        model = EGAMamba(num_classes=3, config=config_mamba)
        msg = model.load_state_dict(sd, strict=False)
        print(
            f"[Eval] load_state_dict(strict=False) missing={len(msg.missing_keys)} "
            f"unexpected={len(msg.unexpected_keys)}"
        )
        if msg.missing_keys:
            print(f"  missing_keys (first 10): {msg.missing_keys[:10]}")
        if msg.unexpected_keys:
            print(f"  unexpected_keys (first 10): {msg.unexpected_keys[:10]}")
        print_mamba_edl_weight_fingerprint(model, tag="(state_dict)")
    else:
        raise TypeError(f"[Eval] unsupported checkpoint type: {type(blob)}")
    if isinstance(blob, dict):
        patch_mamba_edl_legacy_modules(model)
    model.to(device)
    return model


def mamba_edl_forward_for_eval(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    refine_with_prob: bool,
    u_second_pass: bool,
) -> dict:
    """
    - ``refine_with_prob=True``: use native probability refinement in ``EGAMamba.forward`` (needs ``dec_prob_proj``).
    - ``u_second_pass=True`` (no prob refine): first pass estimates EDL uncertainty ``u``; second pass calls
      ``set_runtime_attn(u)`` so SS2D **EASF** (Evidence-driven Anisotropic Scan Fusion) uses **DPE** (Decoupled Prediction-Evidence) uncertainty.
    - Both False: single forward (legacy single-pass; backbone does not use ``u``).
    """
    if refine_with_prob:
        return model(image, refine_with_prob=True)
    if u_second_pass and hasattr(model, "_forward_once"):
        bb = model.backbone
        if hasattr(bb, "_runtime_attn"):
            bb._runtime_attn = None
        if hasattr(bb, "_fusion_u"):
            bb._fusion_u = None
        out1 = model._forward_once(image)
        if hasattr(bb, "set_runtime_attn"):
            bb.set_runtime_attn(out1["u_map"])
        out2 = model._forward_once(image)
        out2["logits_first"] = out1["logits"]
        out2["u_map_first"] = out1["u_map"]
        if hasattr(bb, "_runtime_attn"):
            bb._runtime_attn = None
        if hasattr(bb, "_fusion_u"):
            bb._fusion_u = None
        return out2
    return model(image, refine_with_prob=False)


def print_validation_metric_summary(df: pd.DataFrame, suite_display: str = "") -> None:
    """All / Normal / PAS: prominent Dice + HD95 block, then full metric dump."""
    numeric = df[METRIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    hdr = f"{suite_display.strip()} — " if suite_display.strip() else ""
    group_col = "Group"

    print(f"\n{'=' * 72}")
    print(f"{hdr}Placenta / Myometrium Dice & HD95 (Normal vs PAS)")
    print(f"{'=' * 72}")
    hdr_row = (
        f"{'Subset':<14} {'n':>6} "
        f"{'Pla DSC':>11} {'Pla DSC σ':>11} "
        f"{'Myo DSC':>11} {'Myo DSC σ':>11} "
        f"{'Pla HD95':>11} {'Pla HD σ':>11} "
        f"{'Myo HD95':>11} {'Myo HD σ':>11}"
    )
    print(hdr_row)
    dice_groups = [
        ("All cases", pd.Series(True, index=df.index)),
        ("Normal", df[group_col] == "Normal"),
        ("PAS", df[group_col] == "PAS"),
    ]
    for title, mask in dice_groups:
        sub = numeric.loc[mask]
        n = int(mask.sum())
        if n == 0:
            print(f"{title:<14} {n:>6} (empty)")
            continue
        m_pd, s_pd = _mean_std_numeric(sub["PlacentaDice"])
        m_md, s_md = _mean_std_numeric(sub["MyometriumDice"])
        m_ph, s_ph = _mean_std_numeric(sub["PlacentaHD95(mm)"])
        m_mh, s_mh = _mean_std_numeric(sub["MyometriumHD95(mm)"])
        print(
            f"{title:<14} {n:>6} "
            f"{m_pd:>11.4f} {s_pd:>11.4f} "
            f"{m_md:>11.4f} {s_md:>11.4f} "
            f"{m_ph:>11.4f} {s_ph:>11.4f} "
            f"{m_mh:>11.4f} {s_mh:>11.4f}"
        )

    print(f"\n======== {hdr}All metrics (mean / std) ========")
    groups = [
        ("All cases", pd.Series(True, index=df.index)),
        (f'Normal ({group_col}=="Normal")', df[group_col] == "Normal"),
        (f'PAS ({group_col}=="PAS")', df[group_col] == "PAS"),
    ]
    for title, mask in groups:
        sub = numeric.loc[mask]
        n = int(mask.sum())
        print(f"\n--- {title}  n={n} ---")
        if n == 0:
            print("  (no samples)")
            continue
        for col in METRIC_COLUMNS:
            vals = sub[col].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                print(f"  {col}: no valid values")
                continue
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            print(f"  {col}: mean={mean_v:.4f}, std={std_v:.4f}")


# -------------------------- 2. Validation loop --------------------------
def run_validation_eval(device, config):
    print("===== Start validation / test evaluation =====")

    freeze_eval_environment()

    config_mamba = EGA_Mamba_Config()
    # Default test-time protocol: first forward → EDL u_map; second forward → SS2D EASF driven by that u.
    # Training stays single-pass without EASF (see train.py); override via CLI for ablations.
    use_refine_with_prob = bool(config.get("use_refine_with_prob", False))
    use_edl_u_second_pass = bool(config.get("use_edl_u_second_pass", True))
    use_anisotropic_fusion = bool(config.get("use_anisotropic_fusion", True))
    if use_refine_with_prob:
        use_edl_u_second_pass = False
    # Paths
    model_path = resolve_mamba_edl_ckpt_path(config)
    result_base = config.get("eval_result_base") or config["model_save_root"]
    result_save_path = os.path.join(result_base, config["result_save_dir"], "Prediction")
    test_data_path = resolve_validation_npy_path(config["data_root"])
    csv_save_path = os.path.join(result_base, config["result_save_dir"], "validation_results.csv")

    os.makedirs(result_save_path, exist_ok=True)

    # 1. Load checkpoint
    if not os.path.exists(model_path):
        print(f"[Eval] WARNING: checkpoint missing, skipping evaluation ({model_path})")
        return

    model = load_mamba_edl_for_eval(model_path, config_mamba, device)
    model.eval()
    if use_refine_with_prob and not hasattr(model, "dec_prob_proj"):
        print(
            "[Eval] Legacy checkpoint lacks dec_prob_proj; disabling refine_with_prob (single forward). "
            "If you trained with refinement and have dec_prob_proj, reload the matching checkpoint."
        )
        use_refine_with_prob = False
    set_anisotropic_fusion(model, enabled=use_anisotropic_fusion)
    fusion_note = "EASF (DPE u)" if use_anisotropic_fusion else "disabled"
    print(
        f"[Eval] SS2D fusion: {fusion_note} | "
        f"refine_prob={use_refine_with_prob} | edl_u_second_pass={use_edl_u_second_pass}"
    )
    print(
        format_final_inference_summary(
            use_anisotropic_fusion=use_anisotropic_fusion,
            use_refine_with_prob=use_refine_with_prob,
            use_edl_u_second_pass=use_edl_u_second_pass,
        )
    )
    print(
        "[Eval] Note: full-model .pt loads all pickled Parameters; nothing stripped by strict load_state_dict."
    )

    # 2. Forward sanity check
    try:
        dummy_input = torch.randn(1, config["input_channels"], config["patchsize"][0], config["patchsize"][1]).to(
            device)
        with torch.no_grad():
            dummy_output = mamba_edl_forward_for_eval(
                model,
                dummy_input,
                refine_with_prob=use_refine_with_prob,
                u_second_pass=use_edl_u_second_pass,
            )
            if isinstance(dummy_output, dict) and "logits" in dummy_output:
                output_shape = dummy_output["logits"].shape
            else:
                output_shape = dummy_output.shape if not isinstance(dummy_output, dict) else (0,)

            ph, pw = config["patchsize"][0], config["patchsize"][1]
            assert output_shape == (1, 3, ph, pw), f"Bad logits shape {output_shape}, expected (1,3,{ph},{pw})"
        print("[Eval] Forward sanity check passed")
    except Exception as e:
        print(f"[Eval] Forward sanity check failed: {str(e)}")
        return

    # 3. Dataloader
    try:
        if not os.path.exists(test_data_path):
            print(f"[Eval] WARNING: validation list missing: {test_data_path}")
            return

        imgTrans = get_train_transform_2D(config["patchsize"])
        print(f"[Eval] Loading dataset from {test_data_path}")
        test_dataset = MyDatasetLoader(
            test_data_path,
            pwd=config["pwd_path"],
            mode="test",
            transform=imgTrans["val"],
            device=device,
        )

        print(f"[Eval] Dataset size: {len(test_dataset)} cases")
        if len(test_dataset) == 0:
            print(f"[Eval] WARNING: empty test split; check manifest / processed paths")
            return

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        print(f"[Eval] Batches: {len(test_loader)}")

        first_batch = next(iter(test_loader))
        assert len(first_batch) == 7, f"Loader must return 7 tensors, got {len(first_batch)}"
        image, mask, class_label, patient_name, file_name, ori_shape, ori_imageData = first_batch
        print(f"[Eval] First batch OK: image {image.shape}, mask {mask.shape}, patient {patient_name}")

    except Exception as e:
        print(f"[Eval] Dataset error: {str(e)}")
        return

    # 4. Metrics and records
    records = []
    header = CASE_HEADER + METRIC_COLUMNS
    class_name = CASE_FOLDER_BY_CLINICAL_LABEL

    with torch.no_grad():
        for idx, (image, mask, class_label, patient_name, file_name, ori_shape, ori_imageData) in enumerate(
                test_loader):
            patient_name = patient_name[0]
            file_name = file_name[0]
            print(f"[Eval] Case {idx + 1}/{len(test_loader)}: {patient_name}")

            image = image.to(device).float()
            mask = mask.to(device)

            out = mamba_edl_forward_for_eval(
                model,
                image,
                refine_with_prob=use_refine_with_prob,
                u_second_pass=use_edl_u_second_pass,
            )

            seg_second = out["logits"]
            seg_first = out.get("logits_first", seg_second)
            seg_for_metrics = apply_final_eval_logits(seg_second)
            outputs_softmax = torch.softmax(seg_for_metrics, dim=1)

            # Metrics / _pred.bmp: second-pass logits when two-pass u→EASF (or EGAMamba refine); else single-pass
            outputs_softmax_ori = F.interpolate(outputs_softmax, size=(ori_shape[0].item(), ori_shape[1].item()),
                                                mode='bilinear', align_corners=True)
            pred_mask = torch.argmax(outputs_softmax_ori, dim=1).squeeze().cpu().numpy().astype(np.uint8)

            # Optionally scale mask values if needed (e.g., multiply by 125)
            pred_mask = pred_mask * 125
            # Save as BMP
            mask_img = Image.fromarray(pred_mask)
            os.makedirs(os.path.join(result_save_path, class_name[class_label.item()]), exist_ok=True)
            mask_img.save(os.path.join(result_save_path, class_name[class_label.item()], file_name.replace("_image.bmp", "_pred.bmp")))

            pred_image = ori_imageData.squeeze().cpu().numpy().astype(np.uint8)
            pred_image = Image.fromarray(pred_image)
            pred_image.save(os.path.join(result_save_path, class_name[class_label.item()], file_name))

            outputs_softmax_first = torch.softmax(seg_first, dim=1)

            outputs_softmax_ori_first = F.interpolate(outputs_softmax_first, size=(ori_shape[0].item(), ori_shape[1].item()),
                                                mode='bilinear', align_corners=True)
            pred_mask = torch.argmax(outputs_softmax_ori_first, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            pred_mask = pred_mask * 125
            mask_img = Image.fromarray(pred_mask)
            os.makedirs(os.path.join(result_save_path, class_name[class_label.item()]), exist_ok=True)
            mask_img.save(os.path.join(result_save_path, class_name[class_label.item()], file_name.replace("_image.bmp", "_pred1.bmp")))

            # Uncertainty figure: first-pass u_map when available (matches “first pass estimates u” protocol)
            u_for_vis = out.get("u_map_first", out["u_map"])
            uncertainty_map = F.interpolate(u_for_vis, size=(ori_shape[0].item(), ori_shape[1].item()), mode='bilinear', align_corners=True)
            uncertainty_np = uncertainty_map[0,0,:,:].cpu().numpy()
            uncertainty_np = (uncertainty_np - np.min(uncertainty_np)) / (np.max(uncertainty_np) - np.min(uncertainty_np) + 1e-8)
            uncertainty_img = Image.fromarray((uncertainty_np * 255).astype(np.uint8))
            uncertainty_img.save(os.path.join(result_save_path, class_name[class_label.item()], file_name.replace("_image.bmp", "_uncertainty.bmp")))

            # Pass raw logits (``dsc_calc`` softmaxes internally); do not pass softmaxed tensors
            dsc_t, dsc_j = dsc_calc(mask, seg_for_metrics)
            iou_t, iou_j = iou_calc(mask, seg_for_metrics)
            hd95_t, hd95_j = hd95_calc(mask, seg_for_metrics, config["image_spacing"])
            nsd_t, nsd_j = nsd_calc(mask, seg_for_metrics, config["image_spacing"])

            # NaN-safe rounding for CSV
            def safe_value(val):
                return round(val, 4) if not np.isnan(val) else "NaN"

            # Append row
            temp_case_id = f"case_{idx}"
            case_class = "Normal" if class_label.item() == 1 else "PAS"
            row_metrics = [
                safe_value(dsc_t),
                safe_value(iou_t),
                safe_value(hd95_t),
                safe_value(nsd_t),
                safe_value(dsc_j),
                safe_value(iou_j),
                safe_value(hd95_j),
                safe_value(nsd_j),
            ]
            records.append(
                [patient_name, file_name, temp_case_id, case_class, *row_metrics]
            )
            print(
                f"  [case] {patient_name} | {case_class} | "
                f"Placenta Dice={_fmt_metric_scalar(dsc_t)} IoU={_fmt_metric_scalar(iou_t)} "
                f"HD95={_fmt_metric_scalar(hd95_t)} NSD={_fmt_metric_scalar(nsd_t)} | "
                f"Myometrium Dice={_fmt_metric_scalar(dsc_j)} IoU={_fmt_metric_scalar(iou_j)} "
                f"HD95={_fmt_metric_scalar(hd95_j)} NSD={_fmt_metric_scalar(nsd_j)}"
            )

    # 5. Save CSV
    if len(records) == 0:
        print(f"[Eval] WARNING: no rows written; all cases failed")
        return

    os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
    df = pd.DataFrame(records, columns=header)
    df.to_csv(csv_save_path, index=False, encoding="utf-8-sig", float_format="%.4f")
    suite_disp = (config.get("eval_suite_display") or "").strip()
    print_validation_metric_summary(df, suite_display=suite_disp)
    save_fold_subgroup_metrics_summary_csv(
        df,
        suite_slug=(config.get("eval_suite_slug") or "").strip(),
        suite_display=suite_disp,
        out_dir=os.path.dirname(csv_save_path),
    )
    print(f"[Eval] Finished. Results: {csv_save_path}\n")


def main() -> None:
    """CLI entry: default test protocol is two-pass (first u_map → second pass with EASF); ``--single-pass-eval`` matches train inference."""
    _repo = os.path.dirname(os.path.abspath(__file__))
    if _repo not in sys.path:
        sys.path.insert(0, _repo)

    freeze_eval_environment()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    from torch.backends import cudnn

    cudnn.benchmark = False
    cudnn.deterministic = True

    def _abs_under_repo(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(_repo, p)

    parser = argparse.ArgumentParser(
        description=(
            "EGA-Mamba evaluation: frozen EASF env vars for reproducibility. "
            "Default is two-pass inference — first forward yields EDL u, second enables SS2D EASF. "
            "Use --single-pass-eval for the same single-forward/no-EASF setup as train.py."
        )
    )
    parser.add_argument(
        "--ckpt",
        default="",
        help="Optional checkpoint (.pt); default: Best_model.pt under --model-root",
    )
    parser.add_argument(
        "--model-root",
        default=DEFAULT_MODEL_ROOT,
        help=f"Directory with Best_model.pt (default: {DEFAULT_MODEL_ROOT})",
    )
    parser.add_argument(
        "--results-root",
        default=DEFAULT_RESULTS_ROOT,
        help=f"CSV / eval_* output root (default: {DEFAULT_RESULTS_ROOT})",
    )
    parser.add_argument(
        "--result-tag",
        default="",
        help="Suffix for result subdirectory to avoid overwriting prior CSV runs",
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help=f"Folder with validation.npy (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--pwd-path",
        default=DEFAULT_PWD_PATH,
        help=f"Processed images root (default: {DEFAULT_PWD_PATH})",
    )
    parser.add_argument(
        "--single-pass-eval",
        action="store_true",
        help="One forward, no EASF (matches train.py inference). Ignores --refine-with-prob.",
    )
    parser.add_argument(
        "--no-anisotropic-fusion",
        action="store_true",
        help="Disable SS2D EASF (second forward still runs when using default two-pass u protocol).",
    )
    parser.add_argument(
        "--refine-with-prob",
        action="store_true",
        help="Use EGAMamba.forward decoder refinement instead of u-map→manual second forward+EASF.",
    )
    args = parser.parse_args()

    if args.single_pass_eval:
        use_edl_u_second_pass = False
        use_anisotropic_fusion = False
        use_refine_with_prob = False
        if args.refine_with_prob:
            print("[Eval] Note: --single-pass-eval ignores --refine-with-prob.")
    elif args.refine_with_prob:
        use_edl_u_second_pass = False
        use_anisotropic_fusion = not args.no_anisotropic_fusion
        use_refine_with_prob = True
    else:
        use_edl_u_second_pass = True
        use_anisotropic_fusion = not args.no_anisotropic_fusion
        use_refine_with_prob = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        format_final_inference_summary(
            use_anisotropic_fusion=use_anisotropic_fusion,
            use_refine_with_prob=use_refine_with_prob,
            use_edl_u_second_pass=use_edl_u_second_pass,
        )
    )

    model_save_root = _abs_under_repo(args.model_root)
    results_root = _abs_under_repo(args.results_root)
    data_root = _abs_under_repo(args.data_root)
    pwd_path = _abs_under_repo(args.pwd_path)
    if args.ckpt:
        model_ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(_repo, args.ckpt)
    else:
        model_ckpt = resolve_best_model_pt(model_save_root)

    base_config = {
        "data_root": data_root,
        "model_save_root": model_save_root,
        "pwd_path": pwd_path,
        "eval_result_base": results_root,
        "num_class": 3,
        "input_channels": 1,
        "patchsize": (224, 224),
        "image_spacing": (1.0, 1.0),
        "result_save_dir": "eval_EGA-Mamba_retest",
        "model_ckpt": model_ckpt,
        "use_anisotropic_fusion": use_anisotropic_fusion,
        "use_refine_with_prob": use_refine_with_prob,
        "use_edl_u_second_pass": use_edl_u_second_pass,
    }
    os.makedirs(base_config["model_save_root"], exist_ok=True)
    os.makedirs(results_root, exist_ok=True)

    tag_suffix = f"_{args.result_tag.strip()}" if args.result_tag else ""

    suites = [("eval", data_root, pwd_path, "custom")]

    for suite_label, data_root, pwd_path, slug in suites:
        val_list = resolve_validation_npy_path(data_root)
        if not os.path.isdir(data_root) or not os.path.exists(val_list):
            print(f"[Eval] WARNING: skip '{suite_label}': missing dir or list (expected {val_list})")
            continue
        cfg = dict(base_config)
        cfg["data_root"] = data_root
        cfg["pwd_path"] = pwd_path
        cfg["result_save_dir"] = f"eval_EGA-Mamba_{slug}{tag_suffix}"
        cfg["eval_suite_display"] = suite_label
        cfg["eval_suite_slug"] = slug
        print(f"\n{'=' * 16} ▶ {suite_label} ▶ {'=' * 16}")
        print(
            f"  data_root={data_root}\n  pwd_path={pwd_path}\n"
            f"  results_root={cfg.get('eval_result_base') or cfg['model_save_root']}\n"
            f"  result_subdir={cfg['result_save_dir']}\n"
        )
        run_validation_eval(device, cfg)

    print("[Eval] All requested evaluation runs finished.")


if __name__ == "__main__":
    main()
