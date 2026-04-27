# PasE-Mamba

**PasE-Mamba** is an ultrasound placenta segmentation codebase built around **`PasEMamba`** (historically named MambaUnetEDL). It reports Placenta / Myometrium region metrics and supports **evidential deep learning (EDL)** uncertainty with **DPE** (**Decoupled Prediction-Evidence**) feeding **EASF** (**Evidence-driven Anisotropic Scan Fusion**) in the SS2D backbone.

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

Run scripts from the **repository root** (same directory as this `README.md`). **Examples below use repo-relative paths** so bundled `demo_data/` runs without edits.

### Recommended directory names

| Name | Role |
|------|------|
| **`Training_Evaluation_Data`** | Pass to `--artifacts` / `build_npy_from_processed_data.py --out`. Holds `npy_*`, `processed_eval_merged/`, etc. |
| **`TrainedCheckpoints`** | Pass to `--model-save` / `train.py --model-save` / `eval.py --model-root`. Holds `Best_model.pt` and training curves. |
| **`Results`** | Evaluation outputs: `eval_*` subfolders with `validation_results.csv`, predictions, uncertainty maps (`eval.py --results-root`, default `Results`). |

**Path arguments remain explicit:** `train.py`, `eval.py`, and `build_npy_from_processed_data.py` do not assume a fixed dataset root. `run_pipeline.py` requires `--artifacts` and `--model-save`, and requires `--processed-internal` / `--processed-external` whenever the manifest build step runs (see `--help`).

---

## Data layout for a single `run_pipeline.py` end-to-end run

`run_pipeline.py` assumes you start from **two processed roots only** (no pre-built `train.npy` / `validation.npy`). It calls `build_npy_from_processed_data.py`, then `train.py`, then `eval.py`.

### Directory layout (two processed roots)

Organize data as **two roots** (example uses in-repo `demo_data/`):

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
| Merge for evaluation | Files are copied into `Training_Evaluation_Data/processed_eval_merged/` (when `--artifacts Training_Evaluation_Data`). **Avoid duplicate basenames within the same class** across internal vs. external trees. |
| Counts | Internal split needs enough cases per class to form non-empty train/val manifests (see `build_npy_from_processed_data.py`). External should list all external test cases. |

### Four CLI paths for one full pipeline

| Flag | Role |
|------|------|
| `--processed-internal` | Internal processed root (training image root for `train.py`). |
| `--processed-external` | External processed root (held-out style list for evaluation manifest). |
| `--artifacts` | Output root: `npy_internal/`, `npy_eval_merged/`, `processed_eval_merged/`, etc. (conventional: `Training_Evaluation_Data`). |
| `--model-save` | Weights and training curves (conventional: `TrainedCheckpoints`). |
| `--results-root` | Evaluation CSVs and `eval_*` prediction folders (default: `Results`). |

Example (from the repo root):

```bash
python run_pipeline.py \
  --processed-internal demo_data/processed_internal \
  --processed-external demo_data/processed_external \
  --artifacts Training_Evaluation_Data \
  --model-save TrainedCheckpoints
```

Order of stages: **build manifests under `Training_Evaluation_Data` → train → evaluate** (merged list + `processed_eval_merged`).

Smoke test (few epochs):

```bash
python run_pipeline.py \
  --processed-internal demo_data/processed_internal \
  --processed-external demo_data/processed_external \
  --artifacts Training_Evaluation_Data \
  --model-save TrainedCheckpoints \
  --epochs 2 --min-best-epoch 1 --batch-size 2 --eval-tag smoke
```

More options: `python run_pipeline.py --help`.

---

## Evaluating a trained checkpoint only

You need the same layout as after training:

1. **`validation.npy`** under `Training_Evaluation_Data/npy_eval_merged/`.
2. **Processed root** aligned with that list: `Training_Evaluation_Data/processed_eval_merged/`.
3. **Weights**: `TrainedCheckpoints/Best_model.pt`, or pass **`--ckpt`** (repo-relative path allowed).

### Option A: `run_pipeline.py` (evaluation only)

```bash
python run_pipeline.py --skip-build-npy --skip-train \
  --artifacts Training_Evaluation_Data \
  --model-save TrainedCheckpoints
```

Custom checkpoint (still writes predictions under `Results` unless you set `--results-root`):

```bash
python run_pipeline.py --skip-build-npy --skip-train \
  --artifacts Training_Evaluation_Data \
  --model-save TrainedCheckpoints \
  --ckpt TrainedCheckpoints/Best_model.pt
```

`--model-save` / `--model-root` selects where **`Best_model.pt`** is resolved; **`--results-root`** (default `Results`) is where CSVs and `eval_*` trees are written.

### Option B: `eval.py` directly

```bash
python eval.py \
  --data-root Training_Evaluation_Data/npy_eval_merged \
  --pwd-path Training_Evaluation_Data/processed_eval_merged \
  --model-root TrainedCheckpoints \
  --results-root Results
```

With an explicit weight file:

```bash
python eval.py \
  --data-root Training_Evaluation_Data/npy_eval_merged \
  --pwd-path Training_Evaluation_Data/processed_eval_merged \
  --model-root TrainedCheckpoints \
  --results-root Results \
  --ckpt TrainedCheckpoints/Best_model.pt
```

**DPE** / **EASF** inference switches: `--aniso` / `--no-aniso`, `--edl-u-second-pass` / `--no-edl-u-second-pass`; fusion variant via `EASF_FUSION_VARIANT` (`v2` vs `legacy`). See `python eval.py --help`.

---

## Full-scale training with your own `.npy` manifests

If manifests and processed trees already exist under a layout you control, keep using **repo-relative** paths from the clone root:

```bash
python train.py \
  --data-path Training_Evaluation_Data/npy_internal \
  --pwd-path my_data/processed_internal \
  --model-save TrainedCheckpoints
```

(`my_data/processed_internal` is any **repo-relative** processed tree that matches the filenames in `train.npy`.) Use the same `Training_Evaluation_Data` / `TrainedCheckpoints` / `Results` convention for evaluation as in the section above.

---

## Step-by-step equivalents

### Manifest build only

```bash
python build_npy_from_processed_data.py \
  --internal demo_data/processed_internal \
  --external demo_data/processed_external \
  --out Training_Evaluation_Data
```

### Train only

```bash
python train.py \
  --data-path Training_Evaluation_Data/npy_internal \
  --pwd-path demo_data/processed_internal \
  --model-save TrainedCheckpoints
```

### Evaluate only

See **Evaluating a trained checkpoint only** above.

---

## Outputs

- Training writes `Best_model.pt` and loss / DSC curves under **`TrainedCheckpoints`** (`--model-save`).
- Evaluation writes **`eval_*`** subfolders under **`Results`** (or `--results-root`), including `validation_results.csv`, predicted masks, and uncertainty maps.
- CSV columns: Placenta / Myometrium Dice, IoU, HD95, NSD; case columns include `Patient`, `FileName`, `CaseId`, `Group`.

---

## Core file map

| File | Role |
|------|------|
| `train.py` | Training entry (`--data-path`, `--pwd-path`, `--model-save` required) |
| `eval.py` | Evaluation entry (`--data-root`, `--pwd-path`, `--model-root`; `--results-root` optional, default `Results`) |
| `run_pipeline.py` | Manifest build → train → eval |
| `build_npy_from_processed_data.py` | Build `*.npy` and merged processed tree from two processed roots |
| `data_paths.py` | Resolve `train.npy` / `validation.npy` / `Best_model.pt` |
| `datagenerator.py` | Dataset I/O |
| `vision_mamba.py` | **`PasEMamba`** architecture |
| `mamba_sys.py` | VSSM backbone (runtime **DPE** / **EASF** hooks) |
| `edl_utils.py` | EDL losses, **DPE** routing map, **EASF** fusion helpers |

---

## License and citation

This project is released under the **MIT License**; see [`LICENSE`](LICENSE) in the repository root. When citing, refer to the work as **PasE-Mamba** and link this repository.

---

## FAQ

1. **CUDA OOM**: Lower `--batch-size` (also forwarded by `run_pipeline.py` to `train.py`).
2. **Manifest build errors**: Verify `1-Normal` / `2-PAS`, sufficient internal cases, and no filename collisions on merge.
3. **Evaluation missing files**: Confirm `npy_eval_merged/validation.npy`, consistent `processed_eval_merged`, and `Best_model.pt` or `--ckpt`.
