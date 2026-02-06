# scripts/20_build_crops_from_masks.py
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import cv2


@dataclass
class CropConfig:
    crop_size: int = 128
    crop_margin: int = 8
    min_cell_area_px: int = 80
    max_cell_area_px: int = 6000
    use_clahe: bool = True


def project_root_from_script() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent


def parse_filename(fname: str) -> Dict[str, object]:
    base = Path(fname).name
    base = re.sub(r"^\s*\d+\)\s*", "", base)  # strip "01) "
    s = base.lower().replace(",", ".")

    genotype = "KELLY"
    m_time = re.search(r"(\d+)\s*h", s)
    time_h = int(m_time.group(1)) if m_time else -1

    if "ctrl" in s or "control" in s:
        treatment = "CTRL"
        concentration = 0.0
    elif "aura" in s or "auranofin" in s:
        treatment = "AURA"
        m_conc = re.search(r"(\d+(?:\.\d+)?)\s*(?:u|µ|μ)?m", s)
        if not m_conc:
            raise ValueError(f"Could not parse concentration from: {base}")
        concentration = float(m_conc.group(1))
    else:
        raise ValueError(f"Could not parse treatment from: {base}")

    return dict(genotype=genotype, time=time_h, concentration=concentration, treatment=treatment)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True, help="results/kelly_auranofin_<run_id>")
    ap.add_argument("--raw_dir", default=None, help="data/kelly_auranofin (optional)")
    ap.add_argument("--crop_size", type=int, default=128)
    ap.add_argument("--crop_margin", type=int, default=8)
    ap.add_argument("--min_cell_area_px", type=int, default=80)
    ap.add_argument("--max_cell_area_px", type=int, default=6000)
    ap.add_argument("--clahe", type=int, default=1)
    args = ap.parse_args()

    root = project_root_from_script()
    raw_dir = Path(args.raw_dir) if args.raw_dir else (root / "data" / "kelly_auranofin")
    work_dir = Path(args.work_dir)

    cfg = CropConfig(
        crop_size=int(args.crop_size),
        crop_margin=int(args.crop_margin),
        min_cell_area_px=int(args.min_cell_area_px),
        max_cell_area_px=int(args.max_cell_area_px),
        use_clahe=bool(args.clahe),
    )
    (work_dir / "crop_config.json").write_text(json.dumps(cfg.__dict__, indent=2), encoding="utf-8")

    manifest_img = work_dir / "segment_manifest_images.csv"
    if not manifest_img.exists():
        raise FileNotFoundError(f"Missing {manifest_img}")

    df_img = pd.read_csv(manifest_img)
    df_ok = df_img[df_img["status"].astype(str) == "ok"].copy()
    if df_ok.empty:
        raise RuntimeError("No ok segmentations found.")

    cache_dir = work_dir / "cache_crops"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, r in df_ok.iterrows():
        image_path = Path(r["image_path"])
        mask_npz = Path(r["mask_npz"])
        if not mask_npz.exists():
            continue

        # load image
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        # cv2 loads BGR; convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        if cfg.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        gray = np.ascontiguousarray(gray)

        # load masks
        dat = np.load(mask_npz)
        masks = dat["masks"].astype(np.int32)
        ids = np.unique(masks)
        ids = ids[ids != 0]
        if ids.size == 0:
            continue

        meta = parse_filename(image_path.name)

        stem = image_path.stem
        for cid in ids:
            m = (masks == cid)
            area = int(m.sum())
            if area < cfg.min_cell_area_px or area > cfg.max_cell_area_px:
                continue

            ys, xs = np.where(m)
            if ys.size == 0:
                continue

            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1

            margin = int(cfg.crop_margin)
            y0 = max(0, y0 - margin)
            x0 = max(0, x0 - margin)
            y1 = min(gray.shape[0], y1 + margin)
            x1 = min(gray.shape[1], x1 + margin)

            crop_img = gray[y0:y1, x0:x1]
            crop_msk = m[y0:y1, x0:x1].astype(np.uint8)

            crop_img_r = cv2.resize(crop_img, (cfg.crop_size, cfg.crop_size), interpolation=cv2.INTER_AREA)
            crop_msk_r = cv2.resize(crop_msk, (cfg.crop_size, cfg.crop_size), interpolation=cv2.INTER_NEAREST)

            out_name = f"{stem}__cell{int(cid):05d}.npz"
            out_path = cache_dir / out_name
            np.savez_compressed(out_path, img=crop_img_r.astype(np.uint8), mask=crop_msk_r.astype(np.uint8))

            rows.append(dict(
                crop_path=str(out_path.relative_to(work_dir)),
                source_image=str(image_path.relative_to(root)),
                filename=image_path.name,
                genotype=meta["genotype"],
                time=int(meta["time"]),
                concentration=float(meta["concentration"]),
                treatment=meta["treatment"],
                cell_id=int(cid),
                area_px=int(area),
                crop_height=int(y1 - y0),
                crop_width=int(x1 - x0),
            ))

    if not rows:
        raise RuntimeError("No crops created; check masks/thresholds.")

    manifest_cells = pd.DataFrame(rows)
    out_csv = work_dir / "manifest_cells.csv"
    manifest_cells.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} ({len(manifest_cells)} cells)")


if __name__ == "__main__":
    main()
