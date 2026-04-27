#!/usr/bin/env python3
"""
Build train/validation manifests and an evaluation merge tree from image-only
``processed_internal`` / ``processed_external`` (``1-Normal``, ``2-PAS`` subfolders).

Does not write back into ``demo_data`` by default; output root is user-chosen (e.g. ``pipeline_artifacts/``).

Each row matches ``MyDatasetLoader``: ``[filename, folder_label]``.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np

CLASS_FOLDERS = ("1-Normal", "2-PAS")


def _repo() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _list_image_basenames(class_dir: str) -> list[str]:
    if not os.path.isdir(class_dir):
        return []
    names = []
    for f in os.listdir(class_dir):
        if "_image." not in f:
            continue
        low = f.lower()
        if low.endswith((".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            names.append(f)
    return sorted(names)


def _row(filename: str, folder: str) -> np.ndarray:
    return np.array([filename, folder], dtype=object)


def _internal_train_val_rows(processed_internal: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per class: all but the last file → train; last file → internal validation if class size ≥ 2."""
    train_rows: list[np.ndarray] = []
    val_rows: list[np.ndarray] = []
    for folder in CLASS_FOLDERS:
        d = os.path.join(processed_internal, folder)
        files = _list_image_basenames(d)
        if not files:
            continue
        if len(files) >= 2:
            for fn in files[:-1]:
                train_rows.append(_row(fn, folder))
            val_rows.append(_row(files[-1], folder))
        else:
            train_rows.append(_row(files[0], folder))
    return train_rows, val_rows


def _all_rows_under(processed_root: str) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    for folder in CLASS_FOLDERS:
        d = os.path.join(processed_root, folder)
        for fn in _list_image_basenames(d):
            rows.append(_row(fn, folder))
    return rows


def _merge_processed(internal_root: str, external_root: str, dst_root: str) -> None:
    if os.path.isdir(dst_root):
        shutil.rmtree(dst_root)
    for label in CLASS_FOLDERS:
        for src_root in (internal_root, external_root):
            sd = os.path.join(src_root, label)
            if not os.path.isdir(sd):
                continue
            dd = os.path.join(dst_root, label)
            os.makedirs(dd, exist_ok=True)
            for fn in os.listdir(sd):
                sp = os.path.join(sd, fn)
                if not os.path.isfile(sp):
                    continue
                dp = os.path.join(dd, fn)
                if os.path.exists(dp):
                    raise FileExistsError(
                        f"Merge conflict for {fn} under {label}. "
                        "Use distinct basenames across internal vs. external trees."
                    )
                shutil.copy2(sp, dp)


def build_pipeline_artifacts(
    *,
    processed_internal: str,
    processed_external: str,
    artifacts_dir: str,
) -> dict[str, str]:
    """
    Writes under ``artifacts_dir``::

      npy_internal/{train,validation}.npy
      npy_external/validation.npy
      npy_eval_merged/validation.npy
      processed_eval_merged/   # image root paired with npy_eval_merged

    Returns absolute paths for ``run_pipeline`` / CLI.
    """
    processed_internal = os.path.abspath(processed_internal)
    processed_external = os.path.abspath(processed_external)
    artifacts_dir = os.path.abspath(artifacts_dir)

    if not os.path.isdir(processed_internal):
        raise FileNotFoundError(f"Missing internal processed directory: {processed_internal}")
    if not os.path.isdir(processed_external):
        raise FileNotFoundError(f"Missing external processed directory: {processed_external}")

    npy_int = os.path.join(artifacts_dir, "npy_internal")
    npy_ext = os.path.join(artifacts_dir, "npy_external")
    npy_merged = os.path.join(artifacts_dir, "npy_eval_merged")
    merged_pwd = os.path.join(artifacts_dir, "processed_eval_merged")

    for d in (npy_int, npy_ext, npy_merged):
        os.makedirs(d, exist_ok=True)

    train_rows, val_int_rows = _internal_train_val_rows(processed_internal)
    if not train_rows:
        raise ValueError(
            f"No *_image.* cases under {processed_internal} (expected subfolders {CLASS_FOLDERS})."
        )
    if not val_int_rows:
        if len(train_rows) < 2:
            raise ValueError(
                "Too few internal cases to form non-empty train and validation splits; add samples per class."
            )
        val_int_rows.append(train_rows.pop())

    ext_val_rows = _all_rows_under(processed_external)
    if not ext_val_rows:
        raise ValueError(
            f"No *_image.* cases under {processed_external} (expected subfolders {CLASS_FOLDERS})."
        )

    merged_rows = _all_rows_under(processed_internal) + _all_rows_under(processed_external)

    np.save(os.path.join(npy_int, "train.npy"), np.array(train_rows, dtype=object))
    np.save(os.path.join(npy_int, "validation.npy"), np.array(val_int_rows, dtype=object))
    np.save(os.path.join(npy_ext, "validation.npy"), np.array(ext_val_rows, dtype=object))
    np.save(os.path.join(npy_merged, "validation.npy"), np.array(merged_rows, dtype=object))

    _merge_processed(processed_internal, processed_external, merged_pwd)

    return {
        "artifacts_dir": artifacts_dir,
        "train_data": npy_int,
        "train_pwd": processed_internal,
        "eval_data": npy_merged,
        "eval_pwd": merged_pwd,
        "npy_external": npy_ext,
        "processed_external": processed_external,
    }


def main() -> None:
    repo = _repo()
    p = argparse.ArgumentParser(
        description="Build npy manifests and merged processed tree from internal/external processed roots"
    )
    p.add_argument(
        "--internal",
        required=True,
        help="Internal processed root (1-Normal, 2-PAS); repo-relative or absolute",
    )
    p.add_argument(
        "--external",
        required=True,
        help="External processed root; repo-relative or absolute",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output root for npy_* and processed_eval_merged",
    )
    args = p.parse_args()

    internal = args.internal if os.path.isabs(args.internal) else os.path.join(repo, args.internal)
    external = args.external if os.path.isabs(args.external) else os.path.join(repo, args.external)
    out = args.out if os.path.isabs(args.out) else os.path.join(repo, args.out)

    paths = build_pipeline_artifacts(
        processed_internal=internal,
        processed_external=external,
        artifacts_dir=out,
    )
    print("Wrote pipeline manifests and merged processed tree:")
    for k in ("train_data", "train_pwd", "eval_data", "eval_pwd"):
        print(f"  {k}: {paths[k]}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
