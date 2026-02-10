# scripts/30kelly_segment_all_gpu.py
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

VALID_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass
class SegConfig:
    cellpose_model: str = "cyto2"
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    diameter: Optional[float] = None
    timeout_sec: int = 15 * 60   # per image
    use_clahe: bool = True


def project_root_from_script() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent


def list_images(raw_dir: Path) -> List[Path]:
    return sorted([p for p in raw_dir.rglob("*") if p.suffix.lower() in VALID_EXT])


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


def extract_last_json(stdout_text: str) -> Optional[dict]:
    """
    Worker should print a JSON line as the last line, but if anything else leaked,
    we still try to find the last valid JSON object line.
    """
    lines = [ln.strip() for ln in (stdout_text or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("{") and ln.endswith("}"):
            try:
                return json.loads(ln)
            except Exception:
                continue
    return None


def run_worker(script_path: Path, image_path: Path, work_dir: Path, cfg: SegConfig) -> Dict[str, Any]:
    cmd = [
        sys.executable, str(script_path),
        "--worker",
        "--device", "gpu",
        "--image", str(image_path),
        "--work_dir", str(work_dir),
        "--cellpose_model", cfg.cellpose_model,
        "--flow_threshold", str(cfg.flow_threshold),
        "--cellprob_threshold", str(cfg.cellprob_threshold),
        "--timeout_sec", str(cfg.timeout_sec),
        "--clahe", "1" if cfg.use_clahe else "0",
    ]
    if cfg.diameter is not None:
        cmd += ["--diameter", str(cfg.diameter)]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.timeout_sec, env=env)
    except subprocess.TimeoutExpired:
        return {"status": "fail", "error": f"timeout>{cfg.timeout_sec}s", "device_used": "gpu", "elapsed_sec": time.time() - t0}

    stdout = p.stdout or ""
    stderr = p.stderr or ""

    rec = extract_last_json(stdout)
    if rec is None:
        # keep a short tail of stderr for debugging
        tail = (stderr[-4000:] if stderr else stdout[-4000:])
        return {"status": "fail", "error": f"no_json_from_worker; rc={p.returncode}; tail={tail}", "device_used": "gpu", "elapsed_sec": time.time() - t0}

    rec["elapsed_sec"] = time.time() - t0
    rec["returncode"] = int(p.returncode)
    # attach small tails for quick triage; full logs are saved by worker itself
    rec["stdout_tail"] = stdout[-1000:]
    rec["stderr_tail"] = stderr[-1000:]
    return rec


# ---------------- Worker side (same file) ----------------
def worker_segment_one(args) -> None:
    """
    Runs segmentation for one image and prints ONE JSON line to stdout.
    All other output from libraries is redirected to a per-image worker log file.
    """
    import numpy as np
    import cv2
    from contextlib import redirect_stdout, redirect_stderr
    from cellpose import models as cpmodels, io as cpio

    ip = Path(args.image)
    work_dir = Path(args.work_dir)

    dirs = ensure_dirs(work_dir)
    stem = ip.stem

    out_npz = dirs["masks_npz"] / f"{stem}__masks.npz"
    out_png = dirs["masks_png"] / f"{stem}__masks.png"
    out_ovr = dirs["overlays"] / f"{stem}__overlay.png"
    worker_log = dirs["logs"] / f"{stem}__worker_{args.device}.log"

    t0 = time.time()

    # If already done, return quickly
    if out_npz.exists() and out_ovr.exists():
        print(json.dumps({
            "status": "ok",
            "device_used": args.device,
            "mask_npz": str(out_npz),
            "mask_png": str(out_png) if out_png.exists() else "",
            "overlay_png": str(out_ovr),
            "n_cells": -1,
            "skipped": True,
        }), flush=True)
        return

    try:
        worker_log.parent.mkdir(parents=True, exist_ok=True)
        with open(worker_log, "w", encoding="utf-8") as lf, redirect_stdout(lf), redirect_stderr(lf):
            img = cpio.imread(str(ip))

            # to gray
            if img.ndim == 2:
                gray = img
            elif img.ndim == 3 and img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                raise ValueError(f"Unsupported image shape: {img.shape}")

            gray = gray.astype(np.uint8)
            gray = np.ascontiguousarray(gray)

            if int(args.clahe) == 1:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)

            gpu = (args.device == "gpu")
            model = cpmodels.CellposeModel(gpu=gpu, pretrained_model=args.cellpose_model)

            out = model.eval(
                gray,
                diameter=(None if args.diameter is None else float(args.diameter)),
                channels=[0, 0],
                flow_threshold=float(args.flow_threshold),
                cellprob_threshold=float(args.cellprob_threshold),
            )
            masks = out[0].astype(np.int32)
            ids = np.unique(masks)
            n_cells = int(ids.size - 1) if ids.size else 0

            # save NPZ
            np.savez_compressed(out_npz, masks=masks, image=str(ip), elapsed_sec=float(time.time() - t0))

            # save 16-bit PNG if possible
            mask_png_path = ""
            mx = int(masks.max()) if masks.size else 0
            if mx <= 65535:
                ok = cv2.imwrite(str(out_png), masks.astype(np.uint16))
                if ok:
                    mask_png_path = str(out_png)

            # overlay boundaries
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            edges = cv2.Canny((masks > 0).astype(np.uint8) * 255, 50, 150)
            rgb[edges > 0] = (0, 0, 255)
            cv2.imwrite(str(out_ovr), rgb)

        # Only JSON to stdout
        print(json.dumps({
            "status": "ok",
            "device_used": args.device,
            "mask_npz": str(out_npz),
            "mask_png": mask_png_path,
            "overlay_png": str(out_ovr),
            "n_cells": n_cells,
            "skipped": False,
            "worker_log": str(worker_log),
        }), flush=True)

    except Exception as e:
        print(json.dumps({
            "status": "fail",
            "device_used": args.device,
            "error": str(e)[:4000],
            "worker_log": str(worker_log),
        }), flush=True)
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--timeout_sec", type=int, default=15 * 60)
    ap.add_argument("--flow_threshold", type=float, default=0.4)
    ap.add_argument("--cellprob_threshold", type=float, default=0.0)
    ap.add_argument("--diameter", type=float, default=None)
    ap.add_argument("--cellpose_model", default="cyto2")
    ap.add_argument("--clahe", type=int, default=1)
    ap.add_argument("--raw_dir", default=None)
    ap.add_argument("--work_dir", default=None)

    # worker mode
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--image", default=None)

    args = ap.parse_args()

    if args.worker:
        if not args.image or not args.work_dir:
            raise ValueError("--worker requires --image and --work_dir")
        worker_segment_one(args)
        return

    root = project_root_from_script()
    raw_dir = Path(args.raw_dir) if args.raw_dir else (root / "data" / "kelly_auranofin")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = Path(args.work_dir) if args.work_dir else (root / "results" / f"kelly_auranofin_{run_id}")
    ensure_dirs(work_dir)

    cfg = SegConfig(
        cellpose_model=args.cellpose_model,
        flow_threshold=float(args.flow_threshold),
        cellprob_threshold=float(args.cellprob_threshold),
        diameter=(float(args.diameter) if args.diameter is not None else None),
        timeout_sec=int(args.timeout_sec),
        use_clahe=bool(args.clahe),
    )

    (work_dir / "seg_config.json").write_text(json.dumps(cfg.__dict__, indent=2), encoding="utf-8")

    imgs = list_images(raw_dir)
    if not imgs:
        raise FileNotFoundError(f"No images found in {raw_dir}")

    manifest_path = work_dir / "segment_manifest_images.csv"
    done_ok = set()
    if manifest_path.exists():
        old = pd.read_csv(manifest_path)
        if "image_path" in old.columns and "status" in old.columns:
            done_ok = set(old.loc[old["status"].astype(str) == "ok", "image_path"].astype(str).tolist())

    script_path = Path(__file__).resolve()

    rows = []
    failed = []

    total = len(imgs)
    for i, ip in enumerate(imgs, start=1):
        if str(ip) in done_ok:
            print(f"[{i:02d}/{total:02d}] SKIP ok: {ip.name}")
            continue

        print(f"[{i:02d}/{total:02d}] GPU segment: {ip.name} ...")
        rec = {"image_path": str(ip), "image_name": ip.name}
        w = run_worker(script_path, ip, work_dir, cfg)
        rec.update(w)

        status = rec.get("status", "fail")
        n_cells = rec.get("n_cells", None)
        elapsed = rec.get("elapsed_sec", None)

        if status == "ok":
            print(f"          OK  cells={n_cells}  time={elapsed:.1f}s")
        else:
            print(f"          FAIL time={elapsed:.1f}s  error={str(rec.get('error',''))[:140]}")
            failed.append(ip.name)

        rows.append(rec)

        # append to manifest incrementally
        df_new = pd.DataFrame([rec])
        if manifest_path.exists():
            df_old = pd.read_csv(manifest_path)
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new
        df.to_csv(manifest_path, index=False)

    (work_dir / "failed_gpu.txt").write_text("\n".join(failed) + ("\n" if failed else ""), encoding="utf-8")

    print(f"\nWork dir: {work_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Failed GPU: {len(failed)}")


if __name__ == "__main__":
    main()
