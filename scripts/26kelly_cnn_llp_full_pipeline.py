# -*- coding: utf-8 -*-
"""
CNN (LLP) cell-cycle phase detection for Kelly cells with Auranofin treatment.
Validation vs flow cytometry for all time points (2h, 6h, 24h).

Adaptations for Kelly dataset:
1. New filename parser for "Kelly [treatment] [concentration] [time]h" format
2. All time points (2h, 6h, 24h) included in GT
3. Concentration instead of radiation dose
4. Single cell line (Kelly)

Outputs:
- Cached cell crops (img+mask) .npz
- Model weights: models/cnn_llp_best_state_dict.pt
- Per-cell predictions: cell_predictions.csv
- Per-condition predicted %: predicted_phase_percentages.csv (+ bootstrap std)
- Validation vs cytometry (all GT conditions): validation_vs_cytometry.csv, validation_metrics.json
- Figures: pred_vs_gt_all_times.png, pred_vs_gt_by_time.png
"""

from __future__ import annotations

import re
import math
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from cellpose import models as cpmodels, io as cpio


# -----------------------------
# Constants
# -----------------------------

PHASES_3 = ["SubG1", "G1", "G2M"]
PHASE_COLORS = {"SubG1": "#f28e2b", "G1": "#4e79a7", "G2M": "#59a14f"}
VALID_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# -----------------------------
# Config - адаптировано для Kelly
# -----------------------------

@dataclass
class Config:
    # IO
    project_root: Path
    raw_dir: Path
    results_root: Path
    run_id: str
    results_dir: Path
    cache_dir: Path
    figures_dir: Path
    models_dir: Path
    
    # Experiment type
    experiment_type: str = "kelly_auranofin"  # "hct116_radiation" or "kelly_auranofin"
    
    # Segmentation / crops
    cellpose_pretrained: str = "cyto2"  # Changed to cyto2 for better general performance
    use_gpu_cellpose: bool = True
    min_cell_area_px: int = 80
    max_cell_area_px: int = 6000
    crop_size: int = 128
    crop_margin: int = 8  # Increased margin for rounded cells
    cache_overwrite: bool = False
    max_segmentation_previews: int = 12

    # CNN training (LLP)
    seed: int = 42
    batch_size: int = 256
    epochs: int = 25  # Increased for potentially more complex morphology
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    use_mask_channel: bool = True  # 2-channel input: [img, mask]
    temperature: float = 0.8  # Slightly lower temperature for sharper predictions

    # LLP loss weights - adjusted for smaller dataset
    w_llp: float = 1.0
    w_global: float = 0.1  # Reduced for greater condition specificity
    w_entropy: float = 0.03  # Increased for better individual cell predictions

    # Validation split - adjusted for few images
    val_image_frac: float = 0.25  # 3 out of 12 images for validation

    # Bootstrap
    n_bootstrap: int = 200
    
    # Kelly-specific
    min_conc: float = 0.0  # Control
    max_conc: float = 2.0  # µM


def make_config() -> Config:
    here = Path(__file__).resolve()
    project_root = here.parent.parent if here.parent.name.lower() == "scripts" else Path.cwd()
    
    # Check if we're in Kelly dataset
    raw_dir = project_root / "data" / "kelly_auranofin"
    if not raw_dir.exists():
        # Fallback to default raw directory
        raw_dir = project_root / "data" / "raw"
    
    results_root = project_root / "results"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_type = "kelly_auranofin" if "kelly" in str(raw_dir).lower() else "hct116_radiation"
    results_dir = results_root / f"kelly_auranofin_{run_id}" if experiment_type == "kelly_auranofin" else results_root / f"hct116_radiation_{run_id}"

    return Config(
        project_root=project_root,
        raw_dir=raw_dir,
        results_root=results_root,
        run_id=run_id,
        results_dir=results_dir,
        cache_dir=results_dir / "cache_crops",
        figures_dir=results_dir / "figures",
        models_dir=results_dir / "models",
        experiment_type=experiment_type,
        use_gpu_cellpose=torch.cuda.is_available(),
    )


def cfg_to_jsonable(cfg: Config) -> dict:
    d = cfg.__dict__.copy()
    for k, v in list(d.items()):
        if isinstance(v, Path):
            d[k] = str(v)
    return d


def setup_logging(cfg: Config) -> logging.Logger:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    logfile = cfg.results_dir / "run.log"

    logger = logging.getLogger("kelly_auranofin_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# -----------------------------
# GT (flow cytometry proportions) для Kelly
# -----------------------------

def flow_cytometry_gt_table() -> pd.DataFrame:
    """
    Cytometry GT for Kelly cells with Auranofin treatment.
    All time points (2h, 6h, 24h) included.
    Values are in percent.
    """
    rows = [
        # 2h time point
        dict(genotype="Kelly", time=2, concentration=0.0, treatment="NONE", SubG1=3.56, G1=58.71, G2M=37.10),
        dict(genotype="Kelly", time=2, concentration=0.5, treatment="Aura", SubG1=2.94, G1=59.23, G2M=37.33),
        dict(genotype="Kelly", time=2, concentration=1.0, treatment="Aura", SubG1=5.53, G1=64.88, G2M=29.08),
        dict(genotype="Kelly", time=2, concentration=2.0, treatment="Aura", SubG1=7.53, G1=64.16, G2M=27.70),
        
        # 6h time point
        dict(genotype="Kelly", time=6, concentration=0.0, treatment="NONE", SubG1=3.06, G1=57.67, G2M=38.70),
        dict(genotype="Kelly", time=6, concentration=0.5, treatment="Aura", SubG1=8.48, G1=58.12, G2M=32.36),
        dict(genotype="Kelly", time=6, concentration=1.0, treatment="Aura", SubG1=16.05, G1=51.39, G2M=31.10),
        dict(genotype="Kelly", time=6, concentration=2.0, treatment="Aura", SubG1=21.09, G1=53.52, G2M=24.04),
        
        # 24h time point
        dict(genotype="Kelly", time=24, concentration=0.0, treatment="NONE", SubG1=7.59, G1=62.00, G2M=29.72),
        dict(genotype="Kelly", time=24, concentration=0.5, treatment="Aura", SubG1=21.98, G1=57.08, G2M=20.51),
        dict(genotype="Kelly", time=24, concentration=1.0, treatment="Aura", SubG1=40.71, G1=50.61, G2M=8.59),
        dict(genotype="Kelly", time=24, concentration=2.0, treatment="Aura", SubG1=62.65, G1=28.89, G2M=8.06),
    ]
    
    gt = pd.DataFrame(rows)
    # Normalize to sum to 100% (just in case)
    s = gt[PHASES_3].sum(axis=1).replace(0, np.nan)
    gt[PHASES_3] = gt[PHASES_3].div(s, axis=0).fillna(0.0) * 100.0
    return gt


def gt_map_from_table(gt: pd.DataFrame) -> Dict[Tuple[str, int, float, str], torch.Tensor]:
    """Create mapping from condition to ground truth proportions."""
    m: Dict[Tuple[str, int, float, str], torch.Tensor] = {}
    for _, r in gt.iterrows():
        key = (
            str(r["genotype"]).strip().upper(),
            int(r["time"]),
            float(r["concentration"]),
            str(r["treatment"]).strip().upper()
        )
        t = torch.tensor([float(r["SubG1"]), float(r["G1"]), float(r["G2M"])], dtype=torch.float32)
        t = t / (t.sum() + 1e-8)  # Normalize to sum to 1
        m[key] = t
    return m


def global_target_from_gt(gt: pd.DataFrame) -> torch.Tensor:
    """Compute global target distribution from GT."""
    v = gt[PHASES_3].mean(axis=0).astype(float).values
    t = torch.tensor(v, dtype=torch.float32)
    return t / (t.sum() + 1e-8)


# -----------------------------
# Filename parsing for Kelly dataset
# -----------------------------

def parse_filename(fname: str) -> Dict[str, object]:
    """Parse Kelly dataset filenames.
    
    Expected formats:
    - "Kelly ctrl 2h.jpg"
    - "Kelly Aura 0.5uM 2h.jpg"
    - "Kelly Aura 1uM 6h.jpg"
    - "Kelly Aura 2uM 24h.jpg"
    """
    base = Path(fname).name
    
    # Default values
    genotype = "Kelly"
    time_h = -1
    concentration = 0.0
    treatment = "NONE"
    
    # Convert to lowercase for case-insensitive matching
    base_lower = base.lower()
    
    # Extract time (always has 'h' suffix)
    time_match = re.search(r'(\d+)\s*h', base_lower)
    if time_match:
        time_h = int(time_match.group(1))
    
    # Check for control
    if 'ctrl' in base_lower:
        concentration = 0.0
        treatment = "NONE"
    elif 'aura' in base_lower:
        treatment = "Aura"
        # Extract concentration
        conc_match = re.search(r'(\d+\.?\d*)\s*u?m', base_lower)
        if conc_match:
            concentration = float(conc_match.group(1))
        else:
            # Try pattern like "0.5uM" or "1uM"
            conc_match = re.search(r'(\d+\.?\d*)u', base_lower)
            if conc_match:
                concentration = float(conc_match.group(1))
    
    # Log for debugging
    if time_h == -1:
        print(f"Warning: Could not parse time from filename: {base}")
    
    return {
        "genotype": genotype,
        "time": time_h,
        "concentration": concentration,
        "dose": concentration,  # For compatibility with existing code
        "treatment": treatment,
        "original_filename": base
    }


# -----------------------------
# DataLoader collation (keep conds as list[tuple])
# -----------------------------

def collate_keep_conds(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    conds = [b[1] for b in batch]
    return xs, conds


def normalize_conds(conds_batch: Any) -> List[Tuple[str, int, float, str]]:
    """Normalize conditions for Kelly dataset."""
    # case A: list of sequences length 4 (tuple OR list)
    if isinstance(conds_batch, list) and len(conds_batch) > 0 and isinstance(conds_batch[0], (tuple, list)) and len(conds_batch[0]) == 4:
        out = []
        for g, t, c, tr in conds_batch:
            out.append((str(g).upper(), int(t), float(c), str(tr).upper()))
        return out

    # case B: tuple/list of 4 columns (default_collate style)
    if isinstance(conds_batch, (tuple, list)) and len(conds_batch) == 4:
        genos, times, concs, trts = conds_batch

        def as_list(x):
            return x.tolist() if hasattr(x, "tolist") else list(x)

        genos_l = [str(x).upper() for x in as_list(genos)]
        times_l = [int(x) for x in as_list(times)]
        concs_l = [float(x) for x in as_list(concs)]
        trts_l  = [str(x).upper() for x in as_list(trts)]

        return list(zip(genos_l, times_l, concs_l, trts_l))

    # unknown/unexpected shape
    return []


# -----------------------------
# Segmentation + crop cache (adapted for Kelly morphology)
# -----------------------------

def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def save_segmentation_preview(img_gray: np.ndarray, masks: np.ndarray, out_path: Path) -> None:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bnd = (cv2.Canny((masks > 0).astype(np.uint8) * 255, 50, 150) > 0).astype(np.uint8)
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.imshow(img_gray, cmap="gray")
        ax.imshow(bnd, cmap="autumn", alpha=0.8)
        ax.set_title(out_path.stem)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Could not save segmentation preview: {e}")


def build_crop_cache(cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in cfg.raw_dir.rglob("*") if p.suffix.lower() in VALID_EXT])
    if not imgs:
        raise FileNotFoundError(f"No images found in {cfg.raw_dir}")
    
    logger.info(f"Found {len(imgs)} images in {cfg.raw_dir}")
    for img in imgs:
        logger.info(f"  - {img.name}")

    # Initialize Cellpose model
    gpu_ok = bool(cfg.use_gpu_cellpose and torch.cuda.is_available())
    seg_model = None
    if gpu_ok:
        try:
            seg_model = cpmodels.CellposeModel(gpu=True, pretrained_model=cfg.cellpose_pretrained)
            logger.info("Using Cellpose with GPU")
        except Exception as e:
            logger.warning(f"GPU initialization failed: {e}. Falling back to CPU.")
            seg_model = cpmodels.CellposeModel(gpu=False, pretrained_model=cfg.cellpose_pretrained)
    else:
        seg_model = cpmodels.CellposeModel(gpu=False, pretrained_model=cfg.cellpose_pretrained)
        logger.info("Using Cellpose with CPU")

    rows = []
    previews_left = cfg.max_segmentation_previews

    for ip in imgs:
        meta = parse_filename(ip.name)
        
        # Log parsed metadata
        logger.info(f"Parsed {ip.name}: genotype={meta['genotype']}, time={meta['time']}h, "
                   f"concentration={meta['concentration']}µM, treatment={meta['treatment']}")
        
        if meta["time"] < 0:
            logger.warning(f"Skip (could not parse time): {ip.name}")
            continue

        stem = ip.stem
        existing = list(cfg.cache_dir.glob(f"{stem}__cell*.npz"))
        if existing and not cfg.cache_overwrite:
            logger.info(f"Cache exists, skip segmentation: {ip.name} ({len(existing)} cells)")
            idx_csv = cfg.cache_dir / f"{stem}__index.csv"
            if idx_csv.exists():
                df_idx = pd.read_csv(idx_csv)
                rows.extend(df_idx.to_dict("records"))
            continue

        logger.info(f"Segmenting: {ip.name}")
        img = cpio.imread(str(ip))
        img_gray = to_gray(img).astype(np.uint8)
        img_gray = np.ascontiguousarray(img_gray)
        
        # Apply contrast enhancement for better segmentation
        # This helps with varying morphology (spindle-shaped to rounded)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)

        # --- segmentation ---
        try:
            out = seg_model.eval(img_gray, diameter=None, channels=[0, 0], flow_threshold=0.4, cellprob_threshold=0)
        except Exception as e:
            logger.error(f"Segmentation failed for {ip.name}: {e}")
            continue

        masks = out[0]
        ids = np.unique(masks)
        ids = ids[ids != 0]
        
        if ids.size == 0:
            logger.warning(f"No cells detected: {ip.name}")
            continue

        if previews_left > 0:
            prev_path = cfg.figures_dir / "segmentation_previews" / f"{stem}__preview.png"
            save_segmentation_preview(img_gray, masks, prev_path)
            previews_left -= 1

        per_image_rows = []
        for cid in ids:
            mask = (masks == cid)
            area = int(mask.sum())
            
            # Adjust area thresholds for Kelly cells (might be different sizes)
            min_area = cfg.min_cell_area_px
            max_area = cfg.max_cell_area_px
            
            if area < min_area or area > max_area:
                continue

            ys, xs = np.where(mask)
            if ys.size == 0:
                continue
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1

            # Add margin (adjusted for potentially rounded cells)
            margin = cfg.crop_margin
            y0 = max(0, y0 - margin)
            x0 = max(0, x0 - margin)
            y1 = min(img_gray.shape[0], y1 + margin)
            x1 = min(img_gray.shape[1], x1 + margin)

            crop_img = img_gray[y0:y1, x0:x1]
            crop_msk = mask[y0:y1, x0:x1].astype(np.uint8)

            # Resize to fixed size
            crop_img_r = cv2.resize(crop_img, (cfg.crop_size, cfg.crop_size), interpolation=cv2.INTER_AREA)
            crop_msk_r = cv2.resize(crop_msk, (cfg.crop_size, cfg.crop_size), interpolation=cv2.INTER_NEAREST)

            out_name = f"{stem}__cell{int(cid):05d}.npz"
            out_path = cfg.cache_dir / out_name
            np.savez_compressed(out_path, img=crop_img_r.astype(np.uint8), mask=crop_msk_r.astype(np.uint8))

            rec = dict(
                crop_path=str(out_path.relative_to(cfg.results_dir)),
                source_image=str(ip.relative_to(cfg.project_root)),
                filename=ip.name,
                genotype=str(meta["genotype"]).upper(),
                time=int(meta["time"]),
                concentration=float(meta["concentration"]),
                dose=float(meta["concentration"]),  # For compatibility
                treatment=str(meta["treatment"]).upper(),
                cell_id=int(cid),
                area_px=area,
                crop_height=y1 - y0,
                crop_width=x1 - x0,
            )
            rows.append(rec)
            per_image_rows.append(rec)

        idx_csv = cfg.cache_dir / f"{stem}__index.csv"
        if per_image_rows:
            pd.DataFrame(per_image_rows).to_csv(idx_csv, index=False)
        logger.info(f"Cached {len(per_image_rows)} cells from {ip.name}")

    if not rows:
        raise RuntimeError("No cells were segmented from any image.")

    manifest = pd.DataFrame(rows)
    manifest_path = cfg.results_dir / "manifest_cells.csv"
    manifest.to_csv(manifest_path, index=False)
    
    # Log summary statistics
    logger.info(f"Saved manifest: {manifest_path} ({len(manifest)} total cells)")
    logger.info(f"Cells per time point:")
    for time_val in sorted(manifest["time"].unique()):
        n_cells = len(manifest[manifest["time"] == time_val])
        logger.info(f"  {time_val}h: {n_cells} cells")
    
    logger.info(f"Cells per concentration:")
    for conc_val in sorted(manifest["concentration"].unique()):
        n_cells = len(manifest[manifest["concentration"] == conc_val])
        treatment = manifest[manifest["concentration"] == conc_val]["treatment"].iloc[0] if n_cells > 0 else "N/A"
        logger.info(f"  {conc_val}µM ({treatment}): {n_cells} cells")
    
    return manifest


# -----------------------------
# Dataset + augmentation (adapted for Kelly)
# -----------------------------

class CellCropDataset(Dataset):
    def __init__(self, cfg: Config, manifest: pd.DataFrame, split: str, val_images: set, augment: bool):
        self.cfg = cfg
        self.df = manifest.copy()
        self.split = split
        self.augment_enabled = bool(augment)
        
        # Add morphological features if available
        if "area_px" in self.df.columns:
            self.df["log_area"] = np.log1p(self.df["area_px"])
        
        # Split by source image
        is_val = self.df["source_image"].isin(val_images)
        if split == "val":
            self.df = self.df[is_val].reset_index(drop=True)
        elif split == "train":
            self.df = self.df[~is_val].reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)
        
        self.rng = np.random.default_rng(cfg.seed + (1 if split == "val" else 0))

    def __len__(self) -> int:
        return len(self.df)

    def _augment(self, img: np.ndarray, msk: np.ndarray):
        if not self.augment_enabled:
            return img, msk

        # Random flips
        if self.rng.random() < 0.5:
            img = np.fliplr(img).copy()
            msk = np.fliplr(msk).copy()
        if self.rng.random() < 0.5:
            img = np.flipud(img).copy()
            msk = np.flipud(msk).copy()

        # Random rotation (90, 180, 270 degrees)
        k = int(self.rng.integers(0, 4))
        if k:
            img = np.rot90(img, k).copy()
            msk = np.rot90(msk, k).copy()

        # Random brightness/contrast - more aggressive for varying morphology
        if self.rng.random() < 0.8:
            a = float(self.rng.uniform(0.7, 1.3))  # Wider range for contrast
            b = float(self.rng.uniform(-25, 25))   # Wider range for brightness
            img = np.clip(a * img.astype(np.float32) + b, 0, 255).astype(np.uint8)
            
        # Random Gaussian noise
        if self.rng.random() < 0.3:
            noise = self.rng.normal(0, 5, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img, msk

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        crop_abs = self.cfg.results_dir / Path(r["crop_path"])
        
        try:
            dat = np.load(crop_abs)
            img = dat["img"].astype(np.uint8)
            msk = dat["mask"].astype(np.uint8)
        except Exception as e:
            print(f"Error loading crop {crop_abs}: {e}")
            # Return dummy data
            img = np.zeros((self.cfg.crop_size, self.cfg.crop_size), dtype=np.uint8)
            msk = np.zeros((self.cfg.crop_size, self.cfg.crop_size), dtype=np.uint8)

        img, msk = self._augment(img, msk)

        # Normalize image
        x_img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        
        if self.cfg.use_mask_channel:
            x_msk = torch.from_numpy(msk).float().unsqueeze(0)
            x = torch.cat([x_img, x_msk], dim=0)
        else:
            x = x_img
        
        # For Kelly dataset: (genotype, time, concentration, treatment)
        cond = (
            str(r["genotype"]).upper(),
            int(r["time"]),
            float(r["concentration"]),
            str(r["treatment"]).upper()
        )
        
        return x, cond


def make_val_split(cfg: Config, manifest: pd.DataFrame, logger: logging.Logger) -> set:
    """Create validation split ensuring all treatments and times are represented."""
    rng = np.random.default_rng(cfg.seed)
    
    # Get unique images
    imgs = sorted(manifest["source_image"].unique().tolist())
    
    if len(imgs) <= 1:
        return set()
    
    # For small dataset like Kelly (12 images), we want at least 2-3 validation images
    n_val = max(2, min(4, int(len(imgs) * cfg.val_image_frac)))
    
    # Stratified sampling to ensure all time points are represented in validation
    img_df = manifest[["source_image", "time", "treatment", "concentration"]].drop_duplicates()
    
    # Group by time for stratified sampling
    val_imgs = set()
    for time_val in sorted(img_df["time"].unique()):
        time_imgs = img_df[img_df["time"] == time_val]["source_image"].unique().tolist()
        if len(time_imgs) > 1:
            n_time_val = max(1, int(len(time_imgs) * 0.3))  # 30% of images from each time point
            selected = rng.choice(time_imgs, size=min(n_time_val, len(time_imgs)), replace=False)
            val_imgs.update(selected)
    
    # If we don't have enough validation images, add more randomly
    if len(val_imgs) < n_val:
        remaining = [img for img in imgs if img not in val_imgs]
        needed = n_val - len(val_imgs)
        if remaining and needed > 0:
            additional = rng.choice(remaining, size=min(needed, len(remaining)), replace=False)
            val_imgs.update(additional)
    
    logger.info(f"Split by image: {len(imgs) - len(val_imgs)} train images, {len(val_imgs)} val images")
    logger.info(f"Validation images: {list(val_imgs)}")
    
    return val_imgs


# -----------------------------
# Model (same as before)
# -----------------------------

class SmallCNN(nn.Module):
    def __init__(self, in_ch: int, n_classes: int = 3):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(in_ch, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(48, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        z = self.feat(x).flatten(1)
        return self.head(z)


# -----------------------------
# LLP loss + training (adapted for Kelly)
# -----------------------------

def llp_condition_kl(probs: torch.Tensor,
                     conds: List[Tuple[str, int, float, str]],
                     gt_map: Dict[Tuple[str, int, float, str], torch.Tensor]) -> torch.Tensor:
    device = probs.device
    by: Dict[Tuple[str, int, float, str], List[int]] = {}
    for i, c in enumerate(conds):
        by.setdefault(c, []).append(i)

    losses = []
    for c, idxs in by.items():
        if c not in gt_map:
            continue
        pred = probs[idxs].mean(0).clamp_min(1e-8)
        target = gt_map[c].to(device).clamp_min(1e-8)
        losses.append((pred * (pred.log() - target.log())).sum())

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def entropy_penalty(probs: torch.Tensor) -> torch.Tensor:
    p = probs.clamp_min(1e-8)
    ent = -(p * p.log()).sum(dim=1)
    return -ent.mean()


@torch.no_grad()
def eval_llp_divergence(model: nn.Module, dl: DataLoader, device: str,
                        gt_map: Dict[Tuple[str, int, float, str], torch.Tensor],
                        temperature: float) -> float:
    model.eval()
    total = 0.0
    n = 0
    for x, cond in dl:
        x = x.to(device, non_blocking=True)
        conds = normalize_conds(cond)

        logits = model(x) / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=1)
        loss = llp_condition_kl(probs, conds, gt_map)

        total += float(loss.detach().cpu())
        n += 1
    return total / max(n, 1)


def train_cnn_llp(cfg: Config, logger: logging.Logger, manifest: pd.DataFrame, gt: pd.DataFrame) -> Path:
    gt_map = gt_map_from_table(gt)
    global_t = global_target_from_gt(gt)

    val_imgs = make_val_split(cfg, manifest, logger)

    ds_tr = CellCropDataset(cfg, manifest, split="train", val_images=val_imgs, augment=True)
    ds_va = CellCropDataset(cfg, manifest, split="val",   val_images=val_imgs, augment=False)

    # Filter to GT conditions only (for Kelly, all conditions should be in GT)
    def is_gt_row(df: pd.DataFrame) -> pd.Series:
        keys = list(zip(
            df["genotype"].astype(str).str.upper(),
            df["time"].astype(int),
            df["concentration"].astype(float),
            df["treatment"].astype(str).str.upper()
        ))
        return pd.Series([k in gt_map for k in keys], index=df.index)

    ds_tr.df = ds_tr.df[is_gt_row(ds_tr.df)].reset_index(drop=True)
    ds_va.df = ds_va.df[is_gt_row(ds_va.df)].reset_index(drop=True)

    if len(ds_tr) == 0:
        raise RuntimeError("No training cells matched GT conditions. Check parsing/time/treatment.")
    
    logger.info(f"Training cells (GT conditions only): {len(ds_tr)}")
    logger.info(f"Validation cells (GT conditions only): {len(ds_va)}")
    
    # Log distribution of cells across conditions
    logger.info("Training cells per condition:")
    for (g, t, c, tr), group in ds_tr.df.groupby(["genotype", "time", "concentration", "treatment"]):
        logger.info(f"  {g}, {t}h, {c}µM, {tr}: {len(group)} cells")
    
    dl_tr = DataLoader(
        ds_tr, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_keep_conds
    )
    dl_va = DataLoader(
        ds_va, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate_keep_conds
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_ch = 2 if cfg.use_mask_channel else 1
    model = SmallCNN(in_ch=in_ch, n_classes=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=3, verbose=True
    )

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    best_path = cfg.models_dir / "cnn_llp_best_state_dict.pt"
    meta_path = cfg.models_dir / "cnn_llp_best_meta.json"

    best_val = float("inf")
    patience_counter = 0
    max_patience = 7

    logger.info(f"Device: {device}, in_ch={in_ch}, batch={cfg.batch_size}, epochs={cfg.epochs}")
    logger.info(f"Training on {len(ds_tr)} cells, validating on {len(ds_va)} cells")

    for ep in range(cfg.epochs):
        model.train()
        losses = []
        llp_losses = []
        global_losses = []
        entropy_losses = []
        
        for x, cond in dl_tr:
            x = x.to(device, non_blocking=True)
            conds = normalize_conds(cond)

            logits = model(x) / max(cfg.temperature, 1e-6)
            probs = torch.softmax(logits, dim=1)

            l_llp = llp_condition_kl(probs, conds, gt_map)
            llp_losses.append(float(l_llp.detach().cpu()))

            batch_mean = probs.mean(0).clamp_min(1e-8)
            g = global_t.to(device).clamp_min(1e-8)
            l_global = (batch_mean * (batch_mean.log() - g.log())).sum()
            global_losses.append(float(l_global.detach().cpu()))

            l_ent = entropy_penalty(probs)
            entropy_losses.append(float(l_ent.detach().cpu()))

            loss = cfg.w_llp * l_llp + cfg.w_global * l_global + cfg.w_entropy * l_ent

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()

            losses.append(float(loss.detach().cpu()))

        val = eval_llp_divergence(model, dl_va, device, gt_map, cfg.temperature)
        scheduler.step(val)
        
        tr_mean = float(np.mean(losses)) if losses else float("nan")
        llp_mean = float(np.mean(llp_losses)) if llp_losses else float("nan")
        global_mean = float(np.mean(global_losses)) if global_losses else float("nan")
        entropy_mean = float(np.mean(entropy_losses)) if entropy_losses else float("nan")
        
        logger.info(f"Epoch {ep+1:02d}/{cfg.epochs}: "
                   f"train_loss={tr_mean:.4f} (LLP={llp_mean:.4f}, G={global_mean:.4f}, E={entropy_mean:.4f}), "
                   f"val_KL={val:.4f}")

        if val < best_val:
            best_val = val
            patience_counter = 0
            # SAFE SAVE: state_dict only
            torch.save(model.state_dict(), best_path)
            meta = {
                "best_val_KL": float(best_val),
                "epoch": int(ep + 1),
                "in_ch": int(in_ch),
                "n_classes": 3,
                "temperature": float(cfg.temperature),
                "cfg": cfg_to_jsonable(cfg),
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {ep+1}")
                break

    logger.info(f"Best model saved: {best_path} (val_KL={best_val:.4f})")
    return best_path


# -----------------------------
# Prediction + aggregation (adapted for Kelly)
# -----------------------------

@torch.no_grad()
def predict_all_cells(cfg: Config, logger: logging.Logger, manifest: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_ch = 2 if cfg.use_mask_channel else 1
    model = SmallCNN(in_ch=in_ch, n_classes=3).to(device)

    # Load model state dict
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    ds = CellCropDataset(cfg, manifest, split="all", val_images=set(), augment=False)
    dl = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=collate_keep_conds
    )

    out_rows = []
    for x, cond in dl:
        x = x.to(device, non_blocking=True)
        conds = normalize_conds(cond)

        logits = model(x) / max(cfg.temperature, 1e-6)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        pred = probs.argmax(axis=1)

        n = min(len(pred), len(conds))
        for i in range(n):
            out_rows.append({
                "p_SubG1": float(probs[i, 0]),
                "p_G1": float(probs[i, 1]),
                "p_G2M": float(probs[i, 2]),
                "phase_pred": PHASES_3[int(pred[i])],
            })

    # Combine with manifest
    preds_df = pd.DataFrame(out_rows)
    if len(preds_df) != len(manifest):
        logger.warning(f"Mismatch: {len(preds_df)} predictions vs {len(manifest)} cells")
        # Truncate or pad as needed
        if len(preds_df) < len(manifest):
            preds_df = pd.concat([preds_df, pd.DataFrame([{}] * (len(manifest) - len(preds_df)))], ignore_index=True)
        else:
            preds_df = preds_df.iloc[:len(manifest)]
    
    pred_df = pd.concat([manifest.reset_index(drop=True), preds_df], axis=1)
    
    out_csv = cfg.results_dir / "cell_predictions.csv"
    pred_df.to_csv(out_csv, index=False)
    logger.info(f"Saved per-cell predictions: {out_csv} ({len(pred_df)} rows)")
    
    # Log prediction distribution
    logger.info("Prediction distribution:")
    for phase in PHASES_3:
        n_phase = len(pred_df[pred_df["phase_pred"] == phase])
        logger.info(f"  {phase}: {n_phase} cells ({100*n_phase/len(pred_df):.1f}%)")
    
    return pred_df


def bootstrap_condition_std(pred_df: pd.DataFrame, group_cols: List[str], n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in pred_df.groupby(group_cols):
        n = len(g)
        if n < 20:  # Lower threshold for small dataset
            continue
        probs = g[["p_SubG1", "p_G1", "p_G2M"]].to_numpy(dtype=float)
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boots.append(probs[idx].mean(axis=0) * 100.0)
        boots = np.asarray(boots)
        std = boots.std(axis=0, ddof=1)
        mean = boots.mean(axis=0)

        rec = {c: (key[i] if isinstance(key, tuple) else key) for i, c in enumerate(group_cols)} if isinstance(key, tuple) else {group_cols[0]: key}
        rec.update({
            "SubG1_mean": float(mean[0]), "SubG1_std": float(std[0]),
            "G1_mean": float(mean[1]), "G1_std": float(std[1]),
            "G2M_mean": float(mean[2]), "G2M_std": float(std[2]),
            "n_cells": int(n)
        })
        rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=group_cols + ["SubG1_mean", "SubG1_std", "G1_mean", "G1_std", "G2M_mean", "G2M_std", "n_cells"])
    return pd.DataFrame(rows)


def aggregate_condition_percentages(cfg: Config, logger: logging.Logger, pred_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["genotype", "time", "concentration", "treatment"]
    
    # Check for required columns
    missing_cols = [col for col in group_cols if col not in pred_df.columns]
    if missing_cols:
        logger.error(f"Missing columns for grouping: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Aggregate mean percentages
    agg = pred_df.groupby(group_cols)[["p_SubG1", "p_G1", "p_G2M"]].mean().reset_index()
    agg.rename(columns={"p_SubG1": "SubG1", "p_G1": "G1", "p_G2M": "G2M"}, inplace=True)
    agg[PHASES_3] = agg[PHASES_3] * 100.0
    
    # Add bootstrap standard deviations
    std_df = bootstrap_condition_std(pred_df, group_cols, cfg.n_bootstrap, cfg.seed)
    
    if not std_df.empty:
        out = agg.merge(std_df, on=group_cols, how="left")
    else:
        out = agg
        for phase in PHASES_3:
            out[f"{phase}_std"] = 0.0
        out["n_cells"] = pred_df.groupby(group_cols).size().reset_index(name="n_cells")["n_cells"]
    
    # Sort for readability
    out = out.sort_values(["time", "concentration", "treatment"])
    
    out_csv = cfg.results_dir / "predicted_phase_percentages.csv"
    out.to_csv(out_csv, index=False)
    logger.info(f"Saved condition aggregation: {out_csv} ({len(out)} conditions)")
    
    # Log summary
    logger.info("Predicted phase percentages per condition:")
    for _, row in out.iterrows():
        logger.info(f"  Time={row['time']}h, Conc={row['concentration']}µM, Treat={row['treatment']}: "
                   f"SubG1={row['SubG1']:.1f}%, G1={row['G1']:.1f}%, G2M={row['G2M']:.1f}% "
                   f"(n={row.get('n_cells', 'N/A')})")
    
    return out


# -----------------------------
# Validation vs cytometry (adapted for all time points)
# -----------------------------

def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
    if denom <= 1e-12:
        return float("nan")
    return float((x * y).sum() / denom)


def kl_divergence_pct(p_pct: np.ndarray, q_pct: np.ndarray) -> float:
    p = np.clip(np.asarray(p_pct, dtype=float) / 100.0, 1e-8, 1.0)
    q = np.clip(np.asarray(q_pct, dtype=float) / 100.0, 1e-8, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def validate_with_cytometry(cfg: Config, logger: logging.Logger, pred_cond: pd.DataFrame, gt: pd.DataFrame) -> None:
    gt2 = gt.copy()
    gt2["genotype"] = gt2["genotype"].astype(str).str.upper()
    gt2["treatment"] = gt2["treatment"].astype(str).str.upper()
    
    # For compatibility, ensure concentration column exists
    if "concentration" not in pred_cond.columns and "dose" in pred_cond.columns:
        pred_cond = pred_cond.copy()
        pred_cond["concentration"] = pred_cond["dose"]

    join_cols = ["genotype", "time", "concentration", "treatment"]
    merged = pred_cond.merge(gt2, on=join_cols, suffixes=("_pred", "_gt"), how="inner")

    if merged.empty:
        logger.warning("No overlap between predicted conditions and GT table.")
        # Try alternative join columns
        if "dose" in pred_cond.columns and "concentration" not in pred_cond.columns:
            join_cols_alt = ["genotype", "time", "dose", "treatment"]
            merged = pred_cond.merge(gt2, left_on=join_cols_alt, right_on=join_cols, suffixes=("_pred", "_gt"), how="inner")
        
        if merged.empty:
            logger.error("Still no overlap after trying alternative columns.")
            return

    errs = []
    for _, r in merged.iterrows():
        p = np.array([r["SubG1_pred"], r["G1_pred"], r["G2M_pred"]], dtype=float)
        q = np.array([r["SubG1_gt"], r["G1_gt"], r["G2M_gt"]], dtype=float)
        errs.append({
            "genotype": r["genotype"],
            "time": int(r["time"]),
            "concentration": float(r["concentration"]),
            "treatment": r["treatment"],
            "KL_pred||gt": kl_divergence_pct(p, q),
            "MAE_pct": float(np.mean(np.abs(p - q))),
            "RMSE_pct": float(np.sqrt(np.mean((p - q) ** 2))),
            "SubG1_abs_err": float(abs(p[0] - q[0])),
            "G1_abs_err": float(abs(p[1] - q[1])),
            "G2M_abs_err": float(abs(p[2] - q[2])),
            "SubG1_pred": float(p[0]),
            "SubG1_gt": float(q[0]),
            "G1_pred": float(p[1]),
            "G1_gt": float(q[1]),
            "G2M_pred": float(p[2]),
            "G2M_gt": float(q[2]),
        })

    err_df = pd.DataFrame(errs).sort_values(["genotype", "time", "concentration", "treatment"])
    out_csv = cfg.results_dir / "validation_vs_cytometry.csv"
    err_df.to_csv(out_csv, index=False)
    logger.info(f"Saved validation table: {out_csv} ({len(err_df)} GT conditions)")

    # Compute correlation coefficients for each phase
    r_subg1 = pearsonr(merged["SubG1_pred"].values, merged["SubG1_gt"].values)
    r_g1 = pearsonr(merged["G1_pred"].values, merged["G1_gt"].values)
    r_g2m = pearsonr(merged["G2M_pred"].values, merged["G2M_gt"].values)
    
    # Overall metrics
    overall_mae = err_df["MAE_pct"].mean()
    overall_rmse = np.sqrt((err_df["MAE_pct"] ** 2).mean())
    overall_kl = err_df["KL_pred||gt"].mean()

    metrics = {
        "n_gt_conditions": int(len(merged)),
        "pearson_SubG1": r_subg1,
        "pearson_G1": r_g1,
        "pearson_G2M": r_g2m,
        "mean_MAE_pct": float(overall_mae),
        "mean_RMSE_pct": float(overall_rmse),
        "mean_KL_pred||gt": float(overall_kl),
        "per_time_metrics": {}
    }
    
    # Metrics per time point
    for time_val in sorted(err_df["time"].unique()):
        time_df = err_df[err_df["time"] == time_val]
        metrics["per_time_metrics"][str(time_val)] = {
            "MAE": float(time_df["MAE_pct"].mean()),
            "n_conditions": int(len(time_df))
        }

    metrics_path = cfg.results_dir / "validation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    
    logger.info(f"Validation metrics:")
    logger.info(f"  Pearson correlations: SubG1={r_subg1:.3f}, G1={r_g1:.3f}, G2M={r_g2m:.3f}")
    logger.info(f"  Overall MAE: {overall_mae:.2f}%, RMSE: {overall_rmse:.2f}%, KL: {overall_kl:.4f}")
    
    for time_val in sorted(err_df["time"].unique()):
        time_df = err_df[err_df["time"] == time_val]
        logger.info(f"  Time {time_val}h: MAE={time_df['MAE_pct'].mean():.2f}% (n={len(time_df)} conditions)")


# -----------------------------
# Plotting (adapted for Kelly dataset)
# -----------------------------

def plot_stacked_pred_vs_gt_all_times(cfg: Config, logger: logging.Logger, pred_cond: pd.DataFrame, gt: pd.DataFrame) -> None:
    """Create stacked bar plots comparing predictions vs GT for all time points."""
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    pred = pred_cond.copy()
    pred["genotype"] = pred["genotype"].astype(str).str.upper()
    pred["treatment"] = pred["treatment"].astype(str).str.upper()
    
    # Ensure concentration column exists
    if "concentration" not in pred.columns and "dose" in pred.columns:
        pred["concentration"] = pred["dose"]

    gt2 = gt.copy()
    gt2["genotype"] = gt2["genotype"].astype(str).str.upper()
    gt2["treatment"] = gt2["treatment"].astype(str).str.upper()

    if pred.empty:
        logger.warning("No predicted conditions. Skip plot.")
        return

    # Create subplot for each time point
    times = sorted(pred["time"].unique())
    n_times = len(times)
    
    fig, axes = plt.subplots(n_times, 2, figsize=(14, 4 * n_times), sharey=True)
    if n_times == 1:
        axes = axes.reshape(1, -1)
    
    for idx, time_val in enumerate(times):
        pred_time = pred[pred["time"] == time_val].copy()
        gt_time = gt2[gt2["time"] == time_val].copy()
        
        if pred_time.empty or gt_time.empty:
            continue
        
        # Sort by concentration
        concentrations = sorted(pred_time["concentration"].unique())
        
        # Plot predicted
        ax_pred = axes[idx, 0]
        bottoms = np.zeros(len(concentrations))
        
        for phase_idx, phase in enumerate(PHASES_3):
            values = []
            for conc in concentrations:
                mask = pred_time["concentration"] == conc
                if mask.any():
                    values.append(float(pred_time[mask][phase].iloc[0]))
                else:
                    values.append(0.0)
            
            ax_pred.bar(range(len(concentrations)), values, bottom=bottoms, 
                       color=PHASE_COLORS[phase], alpha=0.8, edgecolor="white", linewidth=1)
            bottoms += values
        
        ax_pred.set_xticks(range(len(concentrations)))
        ax_pred.set_xticklabels([f"{c}µM" for c in concentrations])
        ax_pred.set_ylabel("% cells")
        ax_pred.set_title(f"Predicted - {time_val}h")
        ax_pred.set_ylim(0, 105)
        ax_pred.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Plot GT
        ax_gt = axes[idx, 1]
        bottoms = np.zeros(len(concentrations))
        
        for phase_idx, phase in enumerate(PHASES_3):
            values = []
            for conc in concentrations:
                mask = gt_time["concentration"] == conc
                if mask.any():
                    values.append(float(gt_time[mask][phase].iloc[0]))
                else:
                    values.append(0.0)
            
            ax_gt.bar(range(len(concentrations)), values, bottom=bottoms,
                     color=PHASE_COLORS[phase], alpha=0.8, edgecolor="white", linewidth=1)
            bottoms += values
        
        ax_gt.set_xticks(range(len(concentrations)))
        ax_gt.set_xticklabels([f"{c}µM" for c in concentrations])
        ax_gt.set_title(f"Cytometry GT - {time_val}h")
        ax_gt.set_ylim(0, 105)
        ax_gt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color=PHASE_COLORS[p]) for p in PHASES_3]
    fig.legend(handles, PHASES_3, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    out = cfg.figures_dir / "pred_vs_gt_all_times.png"
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved figure: {out}")


def plot_scatter_predictions_vs_gt(cfg: Config, logger: logging.Logger, pred_cond: pd.DataFrame, gt: pd.DataFrame) -> None:
    """Create scatter plots of predictions vs GT for each phase."""
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge predictions and GT
    gt2 = gt.copy()
    gt2["genotype"] = gt2["genotype"].astype(str).str.upper()
    gt2["treatment"] = gt2["treatment"].astype(str).str.upper()
    
    if "concentration" not in pred_cond.columns and "dose" in pred_cond.columns:
        pred_cond = pred_cond.copy()
        pred_cond["concentration"] = pred_cond["dose"]
    
    join_cols = ["genotype", "time", "concentration", "treatment"]
    merged = pred_cond.merge(gt2, on=join_cols, suffixes=("_pred", "_gt"), how="inner")
    
    if merged.empty:
        logger.warning("No overlapping conditions for scatter plot.")
        return
    
    # Create scatter plot for each phase
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, phase in enumerate(PHASES_3):
        ax = axes[idx]
        x = merged[f"{phase}_gt"].values
        y = merged[f"{phase}_pred"].values
        
        # Scatter plot
        scatter = ax.scatter(x, y, c=merged["time"], cmap="viridis", 
                           s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add identity line
        ax.plot([0, 100], [0, 100], 'r--', alpha=0.5, label='Perfect agreement')
        
        # Add regression line
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), 'b-', alpha=0.5, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
        
        ax.set_xlabel(f"Cytometry GT {phase} (%)")
        ax.set_ylabel(f"Predicted {phase} (%)")
        ax.set_title(f"{phase} phase")
        ax.grid(alpha=0.3)
        ax.legend()
        
        # Add correlation coefficient
        r = pearsonr(x, y)
        ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    out = cfg.figures_dir / "scatter_predictions_vs_gt.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    logger.info(f"Saved scatter plot: {out}")


def plot_concentration_response(cfg: Config, logger: logging.Logger, pred_cond: pd.DataFrame, gt: pd.DataFrame) -> None:
    """Plot phase percentages vs concentration for each time point."""
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    if "concentration" not in pred_cond.columns and "dose" in pred_cond.columns:
        pred_cond = pred_cond.copy()
        pred_cond["concentration"] = pred_cond["dose"]
    
    times = sorted(pred_cond["time"].unique())
    
    fig, axes = plt.subplots(len(times), 3, figsize=(15, 5 * len(times)))
    if len(times) == 1:
        axes = axes.reshape(1, -1)
    
    for time_idx, time_val in enumerate(times):
        pred_time = pred_cond[pred_cond["time"] == time_val].copy()
        gt_time = gt[gt["time"] == time_val].copy()
        
        # Sort by concentration
        pred_time = pred_time.sort_values("concentration")
        gt_time = gt_time.sort_values("concentration")
        
        for phase_idx, phase in enumerate(PHASES_3):
            ax = axes[time_idx, phase_idx]
            
            # Plot predictions with error bars
            if f"{phase}_std" in pred_time.columns:
                ax.errorbar(pred_time["concentration"], pred_time[phase], 
                          yerr=pred_time[f"{phase}_std"], fmt='o-', capsize=5,
                          label='Predicted', color=PHASE_COLORS[phase], alpha=0.8, linewidth=2)
            else:
                ax.plot(pred_time["concentration"], pred_time[phase], 'o-',
                       label='Predicted', color=PHASE_COLORS[phase], alpha=0.8, linewidth=2)
            
            # Plot GT
            ax.plot(gt_time["concentration"], gt_time[phase], 's--',
                   label='Cytometry GT', color='black', alpha=0.6, linewidth=1.5)
            
            ax.set_xlabel("Concentration (µM)")
            ax.set_ylabel(f"{phase} (%)")
            ax.set_title(f"Time {time_val}h - {phase}")
            ax.grid(alpha=0.3)
            ax.legend()
            ax.set_ylim(0, 100)
    
    plt.tight_layout()
    out = cfg.figures_dir / "concentration_response.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    logger.info(f"Saved concentration response plot: {out}")


# -----------------------------
# Additional analysis for Kelly dataset
# -----------------------------

def analyze_morphology_changes(cfg: Config, logger: logging.Logger, pred_df: pd.DataFrame) -> None:
    """Analyze how cell morphology changes with Auranofin treatment."""
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    
    if "area_px" not in pred_df.columns:
        logger.warning("Cell area data not available for morphology analysis.")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # 1. Area distribution by treatment
    ax = axes[0]
    treatments = sorted(pred_df["treatment"].unique())
    for treatment in treatments:
        subset = pred_df[pred_df["treatment"] == treatment]
        ax.hist(subset["area_px"], bins=30, alpha=0.5, label=treatment, density=True)
    ax.set_xlabel("Cell area (px)")
    ax.set_ylabel("Density")
    ax.set_title("Cell area distribution by treatment")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Area vs concentration
    ax = axes[1]
    for time_val in sorted(pred_df["time"].unique()):
        subset = pred_df[pred_df["time"] == time_val]
        if not subset.empty:
            mean_area = subset.groupby("concentration")["area_px"].mean()
            std_area = subset.groupby("concentration")["area_px"].std()
            ax.errorbar(mean_area.index, mean_area.values, yerr=std_area.values,
                       marker='o', label=f"{time_val}h", capsize=5)
    ax.set_xlabel("Concentration (µM)")
    ax.set_ylabel("Mean cell area (px)")
    ax.set_title("Cell area vs concentration")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Phase vs area
    ax = axes[2]
    for phase in PHASES_3:
        subset = pred_df[pred_df["phase_pred"] == phase]
        if not subset.empty:
            ax.hist(subset["area_px"], bins=30, alpha=0.5, label=phase, density=True)
    ax.set_xlabel("Cell area (px)")
    ax.set_ylabel("Density")
    ax.set_title("Cell area distribution by predicted phase")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Area vs time for each treatment
    ax = axes[3]
    for treatment in treatments:
        subset = pred_df[pred_df["treatment"] == treatment]
        if not subset.empty:
            mean_area = subset.groupby("time")["area_px"].mean()
            std_area = subset.groupby("time")["area_px"].std()
            ax.errorbar(mean_area.index, mean_area.values, yerr=std_area.values,
                       marker='s', label=treatment, capsize=5)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Mean cell area (px)")
    ax.set_title("Cell area vs time")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    out = cfg.figures_dir / "morphology_analysis.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    
    # Log statistics
    logger.info("Morphology analysis:")
    for treatment in treatments:
        subset = pred_df[pred_df["treatment"] == treatment]
        logger.info(f"  {treatment}: mean area = {subset['area_px'].mean():.1f} px, "
                   f"std = {subset['area_px'].std():.1f} px, n = {len(subset)}")
    
    logger.info(f"Saved morphology analysis: {out}")


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    cfg = make_config()
    logger = setup_logging(cfg)

    logger.info(f"PROJECT_ROOT: {cfg.project_root}")
    logger.info(f"RAW_DIR:      {cfg.raw_dir}")
    logger.info(f"RESULTS_DIR:  {cfg.results_dir}")
    logger.info(f"EXPERIMENT TYPE: {cfg.experiment_type}")
    
    # Log system information
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    if not cfg.raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {cfg.raw_dir}")
    
    # Create directories
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = cfg.results_dir / "config.json"
    config_path.write_text(json.dumps(cfg_to_jsonable(cfg), indent=2), encoding="utf-8")
    logger.info(f"Saved config: {config_path}")
    
    # Load GT data
    gt = flow_cytometry_gt_table()
    gt.to_csv(cfg.results_dir / "cytometry_gt_table_all_times.csv", index=False)
    logger.info(f"Loaded GT data for {len(gt)} conditions")
    
    # Build crop cache
    logger.info("=" * 60)
    logger.info("STEP 1: Segmentation and crop caching")
    logger.info("=" * 60)
    manifest = build_crop_cache(cfg, logger)
    
    # Train CNN with LLP
    logger.info("=" * 60)
    logger.info("STEP 2: Training CNN with LLP")
    logger.info("=" * 60)
    best_model = train_cnn_llp(cfg, logger, manifest, gt)
    
    # Predict all cells
    logger.info("=" * 60)
    logger.info("STEP 3: Predicting phases for all cells")
    logger.info("=" * 60)
    pred_cells = predict_all_cells(cfg, logger, manifest, best_model)
    
    # Aggregate predictions
    logger.info("=" * 60)
    logger.info("STEP 4: Aggregating predictions by condition")
    logger.info("=" * 60)
    pred_cond = aggregate_condition_percentages(cfg, logger, pred_cells)
    
    # Validate vs cytometry
    logger.info("=" * 60)
    logger.info("STEP 5: Validation vs cytometry")
    logger.info("=" * 60)
    validate_with_cytometry(cfg, logger, pred_cond, gt)
    
    # Create plots
    logger.info("=" * 60)
    logger.info("STEP 6: Creating visualizations")
    logger.info("=" * 60)
    plot_stacked_pred_vs_gt_all_times(cfg, logger, pred_cond, gt)
    plot_scatter_predictions_vs_gt(cfg, logger, pred_cond, gt)
    plot_concentration_response(cfg, logger, pred_cond, gt)
    
    # Analyze morphology changes
    logger.info("=" * 60)
    logger.info("STEP 7: Analyzing morphology changes")
    logger.info("=" * 60)
    analyze_morphology_changes(cfg, logger, pred_cells)
    
    # Create summary report
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total cells processed: {len(manifest)}")
    logger.info(f"Conditions analyzed: {len(pred_cond)}")
    logger.info(f"Results saved in: {cfg.results_dir}")
    logger.info(f"Figures saved in: {cfg.figures_dir}")
    
    # Save final manifest with all predictions
    final_manifest_path = cfg.results_dir / "final_cell_predictions_with_morphology.csv"
    pred_cells.to_csv(final_manifest_path, index=False)
    logger.info(f"Final manifest saved: {final_manifest_path}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()