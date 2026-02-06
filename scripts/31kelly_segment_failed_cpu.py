# scripts/31kelly_segment_failed_cpu.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd
import cv2
from cellpose import models as cpmodels, io as cpio


VALID_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def project_root_from_script() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent


def autodetect_latest_workdir(results_root: Path) -> Optional[Path]:
    cands = []
    for d in results_root.glob("kelly_auranofin_*"):
        if d.is_dir() and (d / "segment_manifest_images.csv").exists():
            cands.append(d)
    if not cands:
        return None
    cands.sort(key=lambda p: (p / "segment_manifest_images.csv").stat().st_mtime, reverse=True)
    return cands[0]


def ensure_dirs(work_dir: Path) -> Dict[str, Path]:
    d = {
        "work_dir": work_dir,
        "masks_npz": work_dir / "masks_npz",
        "masks_png": work_dir / "masks_png",
        "overlays": work_dir / "overlays",
        "logs": work_dir / "logs",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def load_seg_cfg(work_dir: Path) -> dict:
    cfg_path = work_dir / "seg_config.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def latest_status_per_image(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["_row"] = range(len(df2))
    df2 = df2.sort_values("_row").groupby("image_path", as_index=False).tail(1).drop(columns=["_row"])
    return df2


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def save_mask_png_16u(path: Path, masks: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    mx = int(masks.max()) if masks.size else 0
    if mx <= 65535:
        ok = cv2.imwrite(str(path), masks.astype(np.uint16))
        if ok:
            return str(path)
    return ""


def save_overlay(gray: np.ndarray, masks: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    edges = cv2.Canny((masks > 0).astype(np.uint8) * 255, 50, 150)
    rgb[edges > 0] = (0, 0, 255)
    cv2.imwrite(str(out_path), rgb)


def segment_one_cpu(
    image_path: Path,
    work_dir: Path,
    cfg: dict,
) -> Dict[str, Any]:
    dirs = ensure_dirs(work_dir)
    stem = image_path.stem

    out_npz = dirs["masks_npz"] / f"{stem}__masks.npz"
    out_png = dirs["masks_png"] / f"{stem}__masks.png"
    out_ovr = dirs["overlays"] / f"{stem}__overlay.png"
    log_path = dirs["logs"] / f"{stem}__worker_cpu.log"

    t0 = time.time()

    # If already segmented, treat as OK (idempotent)
    if out_npz.exists() and out_ovr.exists():
        return {
            "status": "ok",
            "device_used": "cpu",
            "mask_npz": str(out_npz),
            "mask_png": str(out_png) if out_png.exists() else "",
            "overlay_png": str(out_ovr),
            "n_cells": -1,
            "skipped": True,
            "elapsed_sec": 0.0,
            "worker_log": str(log_path),
        }

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as lf, redirect_stdout(lf), redirect_stderr(lf):
            img = cpio.imread(str(image_path))
            gray = to_gray(img).astype(np.uint8)
            gray = np.ascontiguousarray(gray)

            if bool(cfg.get("use_clahe", True)):
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)

            model_name = cfg.get("cellpose_model", "cyto2")
            model = cpmodels.CellposeModel(gpu=False, pretrained_model=model_name)

            out = model.eval(
                gray,
                diameter=(None if cfg.get("diameter", None) is None else float(cfg["diameter"])),
                channels=[0, 0],
                flow_threshold=float(cfg.get("flow_threshold", 0.4)),
                cellprob_threshold=float(cfg.get("cellprob_threshold", 0.0)),
            )

            masks = out[0].astype(np.int32)
            ids = np.unique(masks)
            n_cells = int(ids.size - 1) if ids.size else 0

            np.savez_compressed(out_npz, masks=masks, image=str(image_path), elapsed_sec=float(time.time() - t0))
            mask_png_path = save_mask_png_16u(out_png, masks)
            save_overlay(gray, masks, out_ovr)

        return {
            "status": "ok",
            "device_used": "cpu",
            "mask_npz": str(out_npz),
            "mask_png": mask_png_path,
            "overlay_png": str(out_ovr),
            "n_cells": int(n_cells),
            "skipped": False,
            "elapsed_sec": float(time.time() - t0),
            "worker_log": str(log_path),
        }

    except Exception as e:
        return {
            "status": "fail",
            "device_used": "cpu",
            "error": str(e)[:4000],
            "elapsed_sec": float(time.time() - t0),
            "worker_log": str(log_path),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default=None, help="results/kelly_auranofin_<run_id>. If omitted, auto-detect latest.")
    args = ap.parse_args()

    root = project_root_from_script()
    results_root = root / "results"

    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        work_dir = autodetect_latest_workdir(results_root)
        if work_dir is None:
            raise FileNotFoundError(f"Could not auto-detect work_dir in {results_root}")

    manifest_path = work_dir / "segment_manifest_images.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}")

    print(f"Using work_dir: {work_dir}")
    cfg = load_seg_cfg(work_dir)

    df = pd.read_csv(manifest_path)
    if "image_path" not in df.columns or "status" not in df.columns:
        raise ValueError("segment_manifest_images.csv must contain columns: image_path, status")

    df_last = latest_status_per_image(df)
    todo = df_last[df_last["status"].astype(str) != "ok"]["image_path"].astype(str).tolist()

    if not todo:
        print("Nothing to do: all images are already OK (latest status).")
        return

    total = len(todo)
    still_failed = []

    for i, ip_str in enumerate(todo, start=1):
        ip = Path(ip_str)
        print(f"[{i:02d}/{total:02d}] CPU segment: {ip.name} ...")
        rec = {"image_path": str(ip), "image_name": ip.name}
        rec.update(segment_one_cpu(ip, work_dir, cfg))

        if rec.get("status") == "ok":
            print(f"          OK  cells={rec.get('n_cells','NA')}  time={rec.get('elapsed_sec',0):.1f}s")
        else:
            print(f"          FAIL time={rec.get('elapsed_sec',0):.1f}s  error={str(rec.get('error',''))[:180]}")
            still_failed.append(ip.name)

        # append attempt
        df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
        df.to_csv(manifest_path, index=False)

    (work_dir / "failed_after_cpu.txt").write_text("\n".join(still_failed) + ("\n" if still_failed else ""), encoding="utf-8")
    print(f"\nDone. CPU processed: {total}; still failed: {len(still_failed)}")
    print(f"Manifest updated: {manifest_path}")


if __name__ == "__main__":
    main()
