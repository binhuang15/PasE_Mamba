# EGA-Mamba

Ultrasound **placenta / myometrium** segmentation with **`EGAMamba`** (`vision_mamba.py`). Run all commands from the **repository root**. Upstream: [binhuang15/EGA_Mamba](https://github.com/binhuang15/EGA_Mamba).

## Environment

```bash
pip install torch torchvision numpy pandas matplotlib pillow scipy timm
```

Build the vendored causal conv extension (see `causal-conv1d/README.md` if this fails):

```bash
cd causal-conv1d && pip install -e . && cd ..
```

## Data (minimal)

You need two processed folders of images/masks, each with **`1-Normal/`** and **`2-PAS/`**. Pairs look like `case001_image.bmp` and `case001_mask.bmp` (filename must contain `_image.`).

End-to-end tooling writes manifests and merged eval files under an **`--artifacts`** folder (e.g. `Training_Evaluation_Data`). Weights go under **`--model-save`** (e.g. `TrainedCheckpoints`). Evaluation CSVs and masks go under **`Results`** by default.

## Train

**Full pipeline** (build lists → train → eval):

```bash
python run_pipeline.py \
  --processed-internal path/to/processed_internal \
  --processed-external path/to/processed_external \
  --artifacts Training_Evaluation_Data \
  --model-save TrainedCheckpoints
```

**Train only** (after manifests exist under `Training_Evaluation_Data/npy_internal/`):

```bash
python train.py \
  --data-path Training_Evaluation_Data/npy_internal \
  --pwd-path path/to/processed_internal \
  --model-save TrainedCheckpoints
```

Use `python train.py --help` for epochs, batch size, learning rate, etc.

## Test / evaluate

**Eval only** (after training produced `Training_Evaluation_Data/` and `TrainedCheckpoints/Best_model.pt`):

```bash
python run_pipeline.py --skip-build-npy --skip-train \
  --artifacts Training_Evaluation_Data \
  --model-save TrainedCheckpoints
```

Or call **`eval.py`** directly:

```bash
python eval.py \
  --data-root Training_Evaluation_Data/npy_eval_merged \
  --pwd-path Training_Evaluation_Data/processed_eval_merged \
  --model-root TrainedCheckpoints \
  --results-root Results
```

If files already sit at those paths, **`python eval.py`** alone is enough. Optional checkpoint: `--ckpt path/to/model.pt`.

Training uses a **single forward pass** without EASF; **default evaluation** runs **two forwards** (uncertainty from the first pass, then EASF on the second). To match training-style inference: **`eval.py --single-pass-eval`** or **`run_pipeline.py --eval-single-pass`**.

Outputs: **`TrainedCheckpoints/Best_model.pt`** and curves; under **`Results`** (or `--results-root`), an **`eval_*`** folder with **`validation_results.csv`** and prediction images.

## License

Released under the **MIT License**; see [`LICENSE`](LICENSE).
