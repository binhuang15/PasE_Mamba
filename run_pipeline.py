#!/usr/bin/env python3
"""
End-to-end driver: ``build_npy_from_processed_data`` → ``train.py`` → ``eval.py``.

Required: ``--artifacts``, ``--model-save``. When building manifests, also pass
``--processed-internal`` and ``--processed-external``.
If ``--skip-build-npy`` is set and training still runs, a real ``--processed-internal`` is required
(as ``train.py --pwd-path``). For eval-only runs you may omit the processed roots.

Convention (repo-relative paths): ``Training_Evaluation_Data`` for manifests / merged images,
``TrainedCheckpoints`` for weights and training curves, ``Results`` for evaluation CSVs and predictions.

Examples::

  python run_pipeline.py \\
    --processed-internal demo_data/processed_internal \\
    --processed-external demo_data/processed_external \\
    --artifacts Training_Evaluation_Data \\
    --model-save TrainedCheckpoints

  # Manifests exist; evaluation only (writes under Results by default)
  python run_pipeline.py --skip-build-npy --skip-train \\
    --artifacts Training_Evaluation_Data --model-save TrainedCheckpoints

  python build_npy_from_processed_data.py --internal demo_data/processed_internal \\
    --external demo_data/processed_external --out Training_Evaluation_Data
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _repo() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _py() -> str:
    return sys.executable


def main() -> None:
    repo = _repo()

    p = argparse.ArgumentParser(
        description=(
            "Build *.npy under --artifacts from processed roots, train into --model-save, then run eval "
            "(predictions under --results-root, default Results). Paths may be repo-relative."
        ),
    )
    p.add_argument(
        "--skip-build-npy",
        action="store_true",
        help="Skip manifest build (npy_* already present under --artifacts)",
    )
    p.add_argument("--skip-train", action="store_true", help="Skip training (build + eval only)")
    p.add_argument("--skip-eval", action="store_true", help="Skip evaluation (build + train only)")

    p.add_argument(
        "--processed-internal",
        default=None,
        help="Internal processed root (required for manifest build; required for training if not skipped)",
    )
    p.add_argument(
        "--processed-external",
        default=None,
        help="External processed root (required for manifest build unless --skip-build-npy)",
    )
    p.add_argument(
        "--artifacts",
        required=True,
        help="Output root for npy_* and processed_eval_merged (e.g. Training_Evaluation_Data)",
    )
    p.add_argument(
        "--model-save",
        required=True,
        help="Directory for Best_model.pt and training curves (e.g. TrainedCheckpoints)",
    )
    p.add_argument(
        "--results-root",
        default="Results",
        help="Directory for evaluation CSVs and prediction folders (forwarded to eval.py --results-root)",
    )
    p.add_argument("--epochs", type=int, default=None, help="Forwarded to train.py (default in train.py)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--min-best-epoch", type=int, default=None)

    p.add_argument(
        "--eval-tag",
        default="pipeline",
        help="Suffix forwarded to eval.py --result-tag (separate result subfolders)",
    )
    p.add_argument(
        "--ckpt",
        default="",
        help="Optional eval.py --ckpt (overrides Best_model.pt under --model-save; repo-relative ok)",
    )

    args = p.parse_args()

    if not args.skip_build_npy:
        if args.processed_internal is None or args.processed_external is None:
            p.error("Manifest build requires --processed-internal and --processed-external (or pass --skip-build-npy)")
    elif not args.skip_train and args.processed_internal is None:
        p.error("Training with --skip-build-npy still requires --processed-internal (training image root)")

    internal = (
        None
        if args.processed_internal is None
        else (
            args.processed_internal
            if os.path.isabs(args.processed_internal)
            else os.path.join(repo, args.processed_internal)
        )
    )
    external = (
        None
        if args.processed_external is None
        else (
            args.processed_external
            if os.path.isabs(args.processed_external)
            else os.path.join(repo, args.processed_external)
        )
    )
    artifacts = args.artifacts if os.path.isabs(args.artifacts) else os.path.join(repo, args.artifacts)
    model_save = args.model_save if os.path.isabs(args.model_save) else os.path.join(repo, args.model_save)
    results_root = (
        args.results_root if os.path.isabs(args.results_root) else os.path.join(repo, args.results_root)
    )

    train_data = os.path.join(artifacts, "npy_internal")
    train_pwd = internal
    eval_data = os.path.join(artifacts, "npy_eval_merged")
    eval_pwd = os.path.join(artifacts, "processed_eval_merged")

    def run_step(title: str, cmd: list[str]) -> None:
        banner = f"\n{'=' * 60}\n>> {title}\n{'=' * 60}\n  {' '.join(cmd)}\n"
        print(banner, flush=True)
        r = subprocess.run(cmd, cwd=repo)
        if r.returncode != 0:
            sys.exit(r.returncode)

    if not args.skip_build_npy:
        run_step(
            "build_npy_from_processed_data",
            [
                _py(),
                os.path.join(repo, "build_npy_from_processed_data.py"),
                "--internal",
                internal,
                "--external",
                external,
                "--out",
                artifacts,
            ],
        )

    if not args.skip_train:
        train_cmd = [
            _py(),
            os.path.join(repo, "train.py"),
            "--data-path",
            train_data,
            "--pwd-path",
            train_pwd,
            "--model-save",
            model_save,
        ]
        if args.epochs is not None:
            train_cmd.extend(["--epochs", str(args.epochs)])
        if args.batch_size is not None:
            train_cmd.extend(["--batch-size", str(args.batch_size)])
        if args.lr is not None:
            train_cmd.extend(["--lr", str(args.lr)])
        if args.min_best_epoch is not None:
            train_cmd.extend(["--min-best-epoch", str(args.min_best_epoch)])
        run_step("train", train_cmd)

    if not args.skip_eval:
        eval_cmd = [
            _py(),
            os.path.join(repo, "eval.py"),
            "--data-root",
            eval_data,
            "--pwd-path",
            eval_pwd,
            "--model-root",
            model_save,
            "--results-root",
            results_root,
            "--result-tag",
            args.eval_tag,
        ]
        if args.ckpt:
            eval_cmd.extend(["--ckpt", args.ckpt])
        run_step("eval", eval_cmd)

    print("\nPipeline finished.\n")
    print(f"  Trained weights / training curves: {model_save}")
    print(f"  Manifests / merged processed images: {artifacts}")
    if not args.skip_eval:
        print(f"  Evaluation CSV and predictions: under {results_root} in eval_* subfolders (see logs)\n")
    else:
        print()


if __name__ == "__main__":
    main()
