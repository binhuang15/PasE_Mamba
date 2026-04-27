# PasE-Mamba

**PasE-Mamba** is an ultrasound placenta segmentation codebase built around **`PasEMamba`** (historically named MambaUnetEDL). It reports Placenta / Myometrium region metrics and supports **evidential deep learning (EDL)** uncertainty with **DPE** (dual-source prior–evidence routing) feeding **EASF** (evidence-aware anisotropic state-space fusion) in the SS2D backbone.

## Environment

- Python 3.10+ (validated on 3.13 in-house)
- [PyTorch](https://pytorch.org/) (CUDA recommended; CPU runs but is slow)
- Example dependency install:

```bash
pip install torch torchvision numpy pandas matplotlib pillow scipy timm
```

The repo vendors `mamba_ssm` and the `causal-conv1d` **sources**; you do not need `pip install mamba_ssm` unless you replace the bundle. **Build the CUDA extension locally** (requires a toolchain matching your PyTorch/CUDA):

```bash
cd causal-conv1d
pip install -e .
cd ..
```

If `pip install -e .` fails, follow `causal-conv1d/README.md` (NVCC, matching CUDA version).

## Clone and working directory

Official repository: **[binhuang15/PasE_Mamba](https://github.com/binhuang15/PasE_Mamba)**.

```bash
git clone https://github.com/binhuang15/PasE_Mamba.git
cd PasE_Mamba
```

Run scripts from the **repository root** (same directory as this `README.md`, e.g. `python run_pipeline.py ...`).

**Path arguments are explicit everywhere:** `train.py`, `eval.py`, and `build_npy_from_processed_data.py` do not assume a fixed dataset root. `run_pipeline.py` requires `--artifacts` and `--model-save`, and requires `--processed-internal` / `--processed-external` whenever the manifest build step runs (see `--help`).

---

## Data layout for a single `run_pipeline.py` end-to-end run

`run_pipeline.py` assumes you start from **two processed roots only** (no pre-built `train.npy` / `validation.npy`). It calls `build_npy_from_processed_data.py`, then `train.py`, then `eval.py`.

### Directory layout (two processed roots; names are yours)

Organize data as **two roots** (example uses in-repo `demo_data/`; any absolute paths are fine):

```text
demo_data/
├── processed_internal/          # --processed-internal
│   ├── 1-Normal/
│   │   ├── case001_image.bmp    # or .png; filename must contain "_image."
│   │   ├── case001_mask.bmp     # paired mask: "_image" -> "_mask"
│   │   └── ...
│   └── 2-PAS/
│       └── ...
└── processed_external/          # --processed-external
    ├── 1-Normal/
    └── 2-PAS/
```

### Naming and content rules

| Item | Rule |
|------|------|
| Subfolders | Must be **`1-Normal`** and **`2-PAS`** (loader contract). |
| Image / mask | For each case: `stem_image.ext` and `stem_mask.ext` in the same class folder. |
| Extensions | Common raster formats (e.g. `.bmp`, `.png`, `.jpg`). |
| Merge for evaluation | Files are copied into `artifacts/processed_eval_merged/`. **Avoid duplicate basenames within the same class** across internal vs. external trees. |
| Counts | Internal split needs enough cases per class to form non-empty train/val manifests (see `build_npy_from_processed_data.py`). External should list all external test cases. |

### Four CLI paths for one full pipeline

| Flag | Role |
|------|------|
| `--processed-internal` | Internal processed root (training image root for `train.py`). |
| `--processed-external` | External processed root (held-out style list for evaluation manifest). |
| `--artifacts` | Output: `npy_internal/`, `npy_eval_merged/`, `processed_eval_merged/`, etc. |
| `--model-save` | Output: `Best_model.pt`, training curves; evaluation writes `eval_*` CSVs and predictions under this root as well. |

Example (from the repo root):

```bash
python run_pipeline.py \
  --processed-internal demo_data/processed_internal \
  --processed-external demo_data/processed_external \
  --artifacts pipeline_artifacts \
  --model-save Result
```

Order of stages: **build manifests under `artifacts` → train on internal lists → evaluate on merged list + `processed_eval_merged`**.

Smoke test (few epochs):

```bash
python run_pipeline.py \
  --processed-internal demo_data/processed_internal \
  --processed-external demo_data/processed_external \
  --artifacts pipeline_artifacts \
  --model-save Result \
  --epochs 2 --min-best-epoch 1 --batch-size 2 --eval-tag smoke
```

More options: `python run_pipeline.py --help`.

---

## Evaluating a trained checkpoint only

You need the same artifacts as after training:

1. **`validation.npy`** under `artifacts/npy_eval_merged/` (e.g. `pipeline_artifacts/npy_eval_merged/validation.npy`).
2. **Processed root** aligned with that list: `artifacts/processed_eval_merged/`.
3. **Weights**: `Best_model.pt` under `--model-save` / `--model-root`, or pass **`--ckpt`**.

### Option A: `run_pipeline.py` (evaluation only)

```bash
python run_pipeline.py --skip-build-npy --skip-train \
  --artifacts pipeline_artifacts \
  --model-save Result
```

Custom checkpoint path (forwarded to `eval.py`):

```bash
python run_pipeline.py --skip-build-npy --skip-train \
  --artifacts pipeline_artifacts \
  --model-save Result \
  --ckpt /path/to/Best_model.pt
```

`--model-save` still sets where CSVs and prediction folders are written; `--ckpt` selects the tensor file to load.

### Option B: `eval.py` directly

```bash
python eval.py \
  --data-root pipeline_artifacts/npy_eval_merged \
  --pwd-path pipeline_artifacts/processed_eval_merged \
  --model-root Result
```

With explicit weights:

```bash
python eval.py \
  --data-root pipeline_artifacts/npy_eval_merged \
  --pwd-path pipeline_artifacts/processed_eval_merged \
  --model-root Result \
  --ckpt /path/to/Best_model.pt
```

DPE/EASF-related inference switches: `--aniso` / `--no-aniso`, `--edl-u-second-pass` / `--no-edl-u-second-pass`; fusion variant via `EASF_FUSION_VARIANT` (`v2` vs `legacy`). See `python eval.py --help`.

---

## Full-scale training with your own `.npy` manifests

If manifests and processed trees already exist elsewhere:

```bash
python train.py \
  --data-path /path/to/npy_dir \
  --pwd-path /path/to/processed_root \
  --model-save /path/to/output_dir
```

For evaluation, ensure `--data-root` contains `validation.npy`, `--pwd-path` is the processed root referenced by filenames in that manifest, and `--model-root` (or `--ckpt`) points to the checkpoint.

---

## Step-by-step equivalents

### Manifest build only

```bash
python build_npy_from_processed_data.py \
  --internal demo_data/processed_internal \
  --external demo_data/processed_external \
  --out pipeline_artifacts
```

### Train only

```bash
python train.py \
  --data-path pipeline_artifacts/npy_internal \
  --pwd-path demo_data/processed_internal \
  --model-save Result
```

### Evaluate only

See **Evaluating a trained checkpoint only** above.

---

## Outputs

- Training writes `Best_model.pt` and loss / DSC curves under **`--model-save`**.
- Evaluation writes subfolders such as **`eval_*`** under **`--model-root`**, including `validation_results.csv`, predicted masks, and uncertainty maps.
- CSV columns: Placenta / Myometrium Dice, IoU, HD95, NSD; case columns include `Patient`, `FileName`, `CaseId`, `Group`.

---

## Core file map

| File | Role |
|------|------|
| `train.py` | Training entry (`--data-path`, `--pwd-path`, `--model-save` required) |
| `eval.py` | Evaluation entry (`--data-root`, `--pwd-path`, `--model-root` required) |
| `run_pipeline.py` | Manifest build → train → eval |
| `build_npy_from_processed_data.py` | Build `*.npy` and merged processed tree from two processed roots |
| `data_paths.py` | Resolve `train.npy` / `validation.npy` / `Best_model.pt` |
| `datagenerator.py` | Dataset I/O |
| `vision_mamba.py` | **`PasEMamba`** architecture |
| `mamba_sys.py` | VSSM backbone (runtime DPE/EASF hooks) |
| `edl_utils.py` | EDL losses, **DPE** routing map, **EASF** fusion helpers |

---

## License and citation

This project is released under the **MIT License**; see [`LICENSE`](LICENSE) in the repository root. When citing, refer to the work as **PasE-Mamba** and link this repository.

---

## FAQ

1. **CUDA OOM**: Lower `--batch-size` (also forwarded by `run_pipeline.py` to `train.py`).
2. **Manifest build errors**: Verify `1-Normal` / `2-PAS`, sufficient internal cases, and no filename collisions on merge.
3. **Evaluation missing files**: Confirm `npy_eval_merged/validation.npy`, consistent `processed_eval_merged`, and `Best_model.pt` or `--ckpt`.
