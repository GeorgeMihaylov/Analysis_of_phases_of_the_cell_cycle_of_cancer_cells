# -*- coding: utf-8 -*-
"""
CNN (LLP) cell-cycle phase detection from microscopy with validation vs flow cytometry (24h only).

Fixes included:
1) parse_filename(): WT underscore-safe ("HCT116_WT_24h_0Gy_...").
2) DataLoader cond collation: custom collate_fn keeps conds as list[tuple] (hashable).
3) PyTorch 2.6+ safe checkpoint: save ONLY model.state_dict() (no Path / cfg objects in torch.save).
4) Inference dataset has augment=False.

Outputs:
- Cached cell crops (img+mask) .npz
- Model weights: models/cnn_llp_best_state_dict.pt
- Per-cell predictions: cell_predictions.csv
- Per-condition predicted %: predicted_phase_percentages.csv (+ bootstrap std)
- Validation vs cytometry (only GT conditions): validation_vs_cytometry.csv, validation_metrics.json
- Figure: figures/pred_vs_gt_time24.png
"""

from __future__ import annotations

import re
import math
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import cv2
import matplotlib.pyplot as plt

from cellpose import models as cpmodels, io as cpio


# -----------------------------
# Constants
# -----------------------------

PHASES_3 = ["SubG1", "G1", "G2M"]
PHASE_COLORS = {"SubG1": "#f28e2b", "G1": "#4e79a7", "G2M": "#59a14f"}
VALID_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# -----------------------------
# Config
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

    # Segmentation / crops
    cellpose_pretrained: str = "cpsam"
    use_gpu_cellpose: bool = True
    min_cell_area_px: int = 80
    max_cell_area_px: int = 6000
    crop_size: int = 128
    crop_margin: int = 6
    cache_overwrite: bool = False
    max_segmentation_previews: int = 12

    # CNN training (LLP)
    seed: int = 42
    batch_size: int = 256
    epochs: int = 18
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    use_mask_channel: bool = True  # 2-channel input: [img, mask]
    temperature: float = 1.0

    # LLP loss weights
    w_llp: float = 1.0
    w_global: float = 0.15
    w_entropy: float = 0.02

    # Validation split
    val_image_frac: float = 0.20

    # Bootstrap
    n_bootstrap: int = 200


def make_config() -> Config:
    here = Path(__file__).resolve()
    project_root = here.parent.parent if here.parent.name.lower() == "scripts" else Path.cwd()

    raw_dir = project_root / "data" / "raw"
    results_root = project_root / "results"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = results_root / f"cnn_llp_{run_id}"

    return Config(
        project_root=project_root,
        raw_dir=raw_dir,
        results_root=results_root,
        run_id=run_id,
        results_dir=results_dir,
        cache_dir=results_dir / "cache_crops",
        figures_dir=results_dir / "figures",
        models_dir=results_dir / "models",
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

    logger = logging.getLogger("cnn_llp_pipeline")
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
# GT (flow cytometry proportions)
# -----------------------------

def flow_cytometry_gt_table() -> pd.DataFrame:
    """
    Cytometry GT for 24h only (SubG1/G1/G2M), doses 0/4/10 for WT and CDK8KO.
    Values are in percent.
    """
    rows = [
        # HCT116 WT
        dict(genotype="WT", time=24, dose=0,  treatment="NONE", SubG1=6.9,  G1=61.4, G2M=20.4),
        dict(genotype="WT", time=24, dose=4,  treatment="NONE", SubG1=22.9, G1=26.4, G2M=45.9),
        dict(genotype="WT", time=24, dose=10, treatment="NONE", SubG1=36.3, G1=20.1, G2M=36.1),
        # HCT116 CDK8KO
        dict(genotype="CDK8KO", time=24, dose=0,  treatment="NONE", SubG1=9.0,  G1=63.4, G2M=16.4),
        dict(genotype="CDK8KO", time=24, dose=4,  treatment="NONE", SubG1=35.4, G1=21.8, G2M=33.2),
        dict(genotype="CDK8KO", time=24, dose=10, treatment="NONE", SubG1=48.1, G1=23.2, G2M=19.4),
    ]
    gt = pd.DataFrame(rows)
    s = gt[PHASES_3].sum(axis=1).replace(0, np.nan)
    gt[PHASES_3] = gt[PHASES_3].div(s, axis=0).fillna(0.0) * 100.0
    return gt


def gt_map_from_table(gt: pd.DataFrame) -> Dict[Tuple[str, int, int, str], torch.Tensor]:
    m: Dict[Tuple[str, int, int, str], torch.Tensor] = {}
    for _, r in gt.iterrows():
        key = (str(r["genotype"]).upper(), int(r["time"]), int(r["dose"]), str(r["treatment"]).upper())
        t = torch.tensor([float(r["SubG1"]), float(r["G1"]), float(r["G2M"])], dtype=torch.float32)
        t = t / (t.sum() + 1e-8)
        m[key] = t
    return m


def global_target_from_gt(gt: pd.DataFrame) -> torch.Tensor:
    v = gt[PHASES_3].mean(axis=0).astype(float).values
    t = torch.tensor(v, dtype=torch.float32)
    return t / (t.sum() + 1e-8)


# -----------------------------
# Filename parsing (WT underscore-safe)
# -----------------------------

def parse_filename(fname: str) -> Dict[str, object]:
    base = Path(fname).name
    treatment = "SNXB" if re.search(r"snxb", base, flags=re.IGNORECASE) else "NONE"

    geno = "UNKNOWN"
    if re.search(r"CDK8KO", base, flags=re.IGNORECASE):
        geno = "CDK8KO"
    elif re.search(r"(^|[^A-Za-z0-9])WT([^A-Za-z0-9]|$)", base, flags=re.IGNORECASE):
        geno = "WT"

    time_h = -1
    m_time = re.search(r"(\d+)\s*h", base, flags=re.IGNORECASE)
    if m_time:
        time_h = int(m_time.group(1))

    dose = -1
    m_dose = re.search(r"(\d+)\s*Gy", base, flags=re.IGNORECASE)
    if m_dose:
        dose = int(m_dose.group(1))

    return {"genotype": geno, "time": time_h, "dose": dose, "treatment": treatment}


# -----------------------------
# DataLoader collation (keep conds as list[tuple])
# -----------------------------

def collate_keep_conds(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    conds = [b[1] for b in batch]
    return xs, conds


def normalize_conds(conds_batch: Any) -> List[Tuple[str, int, int, str]]:
    # case A: list of sequences length 4 (tuple OR list)
    if isinstance(conds_batch, list) and len(conds_batch) > 0 and isinstance(conds_batch[0], (tuple, list)) and len(conds_batch[0]) == 4:
        out = []
        for g, t, d, tr in conds_batch:
            out.append((str(g).upper(), int(t), int(d), str(tr).upper()))
        return out

    # case B: tuple/list of 4 columns (default_collate style)
    if isinstance(conds_batch, (tuple, list)) and len(conds_batch) == 4:
        genos, times, doses, trts = conds_batch

        def as_list(x):
            return x.tolist() if hasattr(x, "tolist") else list(x)

        genos_l = [str(x).upper() for x in as_list(genos)]
        times_l = [int(x) for x in as_list(times)]
        doses_l = [int(x) for x in as_list(doses)]
        trts_l  = [str(x).upper() for x in as_list(trts)]

        return list(zip(genos_l, times_l, doses_l, trts_l))

    # unknown/unexpected shape
    return []



# -----------------------------
# Segmentation + crop cache
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
    except Exception:
        pass


def build_crop_cache(cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in cfg.raw_dir.rglob("*") if p.suffix.lower() in VALID_EXT])
    if not imgs:
        raise FileNotFoundError(f"No images found in {cfg.raw_dir}")

    # two models: GPU preferred, CPU fallback
    gpu_ok = bool(cfg.use_gpu_cellpose and torch.cuda.is_available())
    seg_model_gpu = None
    if gpu_ok:
        seg_model_gpu = cpmodels.CellposeModel(gpu=True, pretrained_model=cfg.cellpose_pretrained)
    seg_model_cpu = cpmodels.CellposeModel(gpu=False, pretrained_model=cfg.cellpose_pretrained)

    rows = []
    previews_left = cfg.max_segmentation_previews
    force_cpu = not gpu_ok  # after a CUDA kernel crash, switch to CPU for rest

    for ip in imgs:
        meta = parse_filename(ip.name)
        if meta["genotype"] == "UNKNOWN" or meta["time"] < 0 or meta["dose"] < 0:
            logger.warning(f"Skip (unparsed): {ip.name}")
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

        logger.info(f"Segment: {ip.name}")
        img = cpio.imread(str(ip))
        img_gray = to_gray(img).astype(np.uint8)
        img_gray = np.ascontiguousarray(img_gray)

        # --- segmentation with fallback ---
        try:
            if (not force_cpu) and (seg_model_gpu is not None):
                out = seg_model_gpu.eval(img_gray, diameter=None, channels=None)
            else:
                out = seg_model_cpu.eval(img_gray, diameter=None, channels=None)
        except RuntimeError as e:
            msg = str(e).lower()
            if ("misaligned address" in msg) or ("cuda error" in msg):
                logger.warning(f"Cellpose GPU failed on {ip.name} ({e}). Retrying on CPU and switching to CPU for remaining images.")
                force_cpu = True
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                out = seg_model_cpu.eval(img_gray, diameter=None, channels=None)
            else:
                raise

        masks = out[0]
        ids = np.unique(masks)
        ids = ids[ids != 0]
        if ids.size == 0:
            logger.warning(f"No cells: {ip.name}")
            continue

        if previews_left > 0:
            prev_path = cfg.figures_dir / "segmentation_previews" / f"{stem}__preview.png"
            save_segmentation_preview(img_gray, masks, prev_path)
            previews_left -= 1

        per_image_rows = []
        for cid in ids:
            mask = (masks == cid)
            area = int(mask.sum())
            if area < cfg.min_cell_area_px or area > cfg.max_cell_area_px:
                continue

            ys, xs = np.where(mask)
            if ys.size == 0:
                continue
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1

            y0 = max(0, y0 - cfg.crop_margin)
            x0 = max(0, x0 - cfg.crop_margin)
            y1 = min(img_gray.shape[0], y1 + cfg.crop_margin)
            x1 = min(img_gray.shape[1], x1 + cfg.crop_margin)

            crop_img = img_gray[y0:y1, x0:x1]
            crop_msk = mask[y0:y1, x0:x1].astype(np.uint8)

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
                dose=int(meta["dose"]),
                treatment=str(meta["treatment"]).upper(),
                cell_id=int(cid),
                area_px=area,
            )
            rows.append(rec)
            per_image_rows.append(rec)

        idx_csv = cfg.cache_dir / f"{stem}__index.csv"
        pd.DataFrame(per_image_rows).to_csv(idx_csv, index=False)
        logger.info(f"Cached cells: {len(per_image_rows)} from {ip.name}")

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("Manifest is empty: no parsed images or no valid cells.")

    manifest_path = cfg.results_dir / "manifest_cells.csv"
    manifest.to_csv(manifest_path, index=False)
    logger.info(f"Saved manifest: {manifest_path} ({len(manifest)} cells)")
    return manifest


# -----------------------------
# Dataset + augmentation
# -----------------------------

class CellCropDataset(Dataset):
    def __init__(self, cfg: Config, manifest: pd.DataFrame, split: str, val_images: set, augment: bool):
        self.cfg = cfg
        self.df = manifest.copy()
        self.split = split
        self.augment_enabled = bool(augment)

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

        if self.rng.random() < 0.5:
            img = np.fliplr(img).copy()
            msk = np.fliplr(msk).copy()
        if self.rng.random() < 0.5:
            img = np.flipud(img).copy()
            msk = np.flipud(msk).copy()

        k = int(self.rng.integers(0, 4))
        if k:
            img = np.rot90(img, k).copy()
            msk = np.rot90(msk, k).copy()

        if self.rng.random() < 0.8:
            a = float(self.rng.uniform(0.85, 1.15))
            b = float(self.rng.uniform(-18, 18))
            img = np.clip(a * img.astype(np.float32) + b, 0, 255).astype(np.uint8)

        return img, msk

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        crop_abs = self.cfg.results_dir / Path(r["crop_path"])
        dat = np.load(crop_abs)
        img = dat["img"].astype(np.uint8)
        msk = dat["mask"].astype(np.uint8)

        img, msk = self._augment(img, msk)

        x_img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        if self.cfg.use_mask_channel:
            x_msk = torch.from_numpy(msk).float().unsqueeze(0)
            x = torch.cat([x_img, x_msk], dim=0)
        else:
            x = x_img

        cond = (str(r["genotype"]).upper(), int(r["time"]), int(r["dose"]), str(r["treatment"]).upper())
        return x, cond


def make_val_split(cfg: Config, manifest: pd.DataFrame, logger: logging.Logger) -> set:
    rng = np.random.default_rng(cfg.seed)
    imgs = sorted(manifest["source_image"].unique().tolist())
    n_val = max(1, int(len(imgs) * cfg.val_image_frac))
    val_imgs = set(rng.choice(imgs, size=n_val, replace=False).tolist())
    logger.info(f"Split by image: {len(imgs) - len(val_imgs)} train images, {len(val_imgs)} val images")
    return val_imgs


# -----------------------------
# Model
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
# LLP loss + training
# -----------------------------

def llp_condition_kl(probs: torch.Tensor,
                     conds: List[Tuple[str, int, int, str]],
                     gt_map: Dict[Tuple[str, int, int, str], torch.Tensor]) -> torch.Tensor:
    device = probs.device
    by: Dict[Tuple[str, int, int, str], List[int]] = {}
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
                        gt_map: Dict[Tuple[str, int, int, str], torch.Tensor],
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

    # filter to GT conditions only
    def is_gt_row(df: pd.DataFrame) -> pd.Series:
        keys = list(zip(
            df["genotype"].astype(str).str.upper(),
            df["time"].astype(int),
            df["dose"].astype(int),
            df["treatment"].astype(str).str.upper()
        ))
        return pd.Series([k in gt_map for k in keys], index=df.index)

    ds_tr.df = ds_tr.df[is_gt_row(ds_tr.df)].reset_index(drop=True)
    ds_va.df = ds_va.df[is_gt_row(ds_va.df)].reset_index(drop=True)

    if len(ds_tr) == 0:
        raise RuntimeError("No training cells matched GT conditions. Check parsing/time/treatment.")
    logger.info(f"Train cells (GT conditions only): {len(ds_tr)}")
    logger.info(f"Val   cells (GT conditions only): {len(ds_va)}")

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

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    best_path = cfg.models_dir / "cnn_llp_best_state_dict.pt"
    meta_path = cfg.models_dir / "cnn_llp_best_meta.json"

    best_val = float("inf")

    logger.info(f"Device: {device}, in_ch={in_ch}, batch={cfg.batch_size}, epochs={cfg.epochs}")

    for ep in range(cfg.epochs):
        model.train()
        losses = []
        for x, cond in dl_tr:
            x = x.to(device, non_blocking=True)
            conds = normalize_conds(cond)

            logits = model(x) / max(cfg.temperature, 1e-6)
            probs = torch.softmax(logits, dim=1)

            l_llp = llp_condition_kl(probs, conds, gt_map)

            batch_mean = probs.mean(0).clamp_min(1e-8)
            g = global_t.to(device).clamp_min(1e-8)
            l_global = (batch_mean * (batch_mean.log() - g.log())).sum()

            l_ent = entropy_penalty(probs)

            loss = cfg.w_llp * l_llp + cfg.w_global * l_global + cfg.w_entropy * l_ent

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()

            losses.append(float(loss.detach().cpu()))

        val = eval_llp_divergence(model, dl_va, device, gt_map, cfg.temperature)
        tr_mean = float(np.mean(losses)) if losses else float("nan")
        logger.info(f"Epoch {ep+1:02d}/{cfg.epochs}: train_loss={tr_mean:.4f}, val_KL={val:.4f}")

        if val < best_val:
            best_val = val
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

    logger.info(f"Best model saved: {best_path} (val_KL={best_val:.4f})")
    return best_path


# -----------------------------
# Prediction + aggregation
# -----------------------------

@torch.no_grad()
def predict_all_cells(cfg: Config, logger: logging.Logger, manifest: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_ch = 2 if cfg.use_mask_channel else 1
    model = SmallCNN(in_ch=in_ch, n_classes=3).to(device)

    # SAFE LOAD: state_dict only (works with PyTorch 2.6 default weights_only=True)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    ds = CellCropDataset(cfg, manifest, split="all", val_images=set(), augment=False)
    dl = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=collate_keep_conds
    )

    out_rows = []
    first = True
    for x, cond in dl:
        x = x.to(device, non_blocking=True)

        if first:
            logger.info(f"[DEBUG] type(cond)={type(cond)}, example={str(cond)[:200]}")
            first = False

        conds = normalize_conds(cond)

        logits = model(x) / max(cfg.temperature, 1e-6)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        pred = probs.argmax(axis=1)

        n = min(len(pred), len(conds))
        if n != len(pred):
            logger.warning(
                f"[WARN] conds shorter than batch: len(pred)={len(pred)} len(conds)={len(conds)}; truncating to {n}")

        for i in range(n):
            g, t, d, tr = conds[i]
            out_rows.append({
                "genotype": g,
                "time": int(t),
                "dose": int(d),
                "treatment": tr,
                "p_SubG1": float(probs[i, 0]),
                "p_G1": float(probs[i, 1]),
                "p_G2M": float(probs[i, 2]),
                "phase_pred": PHASES_3[int(pred[i])],
            })

    pred_df = pd.concat([manifest.reset_index(drop=True), pd.DataFrame(out_rows)], axis=1)
    out_csv = cfg.results_dir / "cell_predictions.csv"
    pred_df.to_csv(out_csv, index=False)
    logger.info(f"Saved per-cell predictions: {out_csv} ({len(pred_df)} rows)")
    return pred_df


def bootstrap_condition_std(pred_df: pd.DataFrame, group_cols: List[str], n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in pred_df.groupby(group_cols):
        n = len(g)
        if n < 40:
            continue
        probs = g[["p_SubG1", "p_G1", "p_G2M"]].to_numpy(dtype=float)
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boots.append(probs[idx].mean(axis=0) * 100.0)
        boots = np.asarray(boots)
        std = boots.std(axis=0, ddof=1)

        rec = {c: (key[i] if isinstance(key, tuple) else key) for i, c in enumerate(group_cols)} if isinstance(key, tuple) else {group_cols[0]: key}
        rec.update({"SubG1_std": float(std[0]), "G1_std": float(std[1]), "G2M_std": float(std[2]), "n_cells": int(n)})
        rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=group_cols + ["SubG1_std", "G1_std", "G2M_std", "n_cells"])
    return pd.DataFrame(rows)


def aggregate_condition_percentages(cfg: Config, logger: logging.Logger, pred_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["genotype", "time", "dose", "treatment"]
    agg = pred_df.groupby(group_cols)[["p_SubG1", "p_G1", "p_G2M"]].mean().reset_index()
    agg.rename(columns={"p_SubG1": "SubG1", "p_G1": "G1", "p_G2M": "G2M"}, inplace=True)
    agg[PHASES_3] = agg[PHASES_3] * 100.0

    std = bootstrap_condition_std(pred_df, group_cols, cfg.n_bootstrap, cfg.seed)
    out = agg.merge(std, on=group_cols, how="left")

    out_csv = cfg.results_dir / "predicted_phase_percentages.csv"
    out.to_csv(out_csv, index=False)
    logger.info(f"Saved condition aggregation: {out_csv} ({len(out)} conditions)")
    return out


# -----------------------------
# Validation vs cytometry
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

    join_cols = ["genotype", "time", "dose", "treatment"]
    merged = pred_cond.merge(gt2, on=join_cols, suffixes=("_pred", "_gt"), how="inner")

    if merged.empty:
        logger.warning("No overlap between predicted conditions and GT table.")
        return

    errs = []
    for _, r in merged.iterrows():
        p = np.array([r["SubG1_pred"], r["G1_pred"], r["G2M_pred"]], dtype=float)
        q = np.array([r["SubG1_gt"], r["G1_gt"], r["G2M_gt"]], dtype=float)
        errs.append({
            "genotype": r["genotype"],
            "time": int(r["time"]),
            "dose": int(r["dose"]),
            "treatment": r["treatment"],
            "KL_pred||gt": kl_divergence_pct(p, q),
            "MAE_pct": float(np.mean(np.abs(p - q))),
            "SubG1_abs_err": float(abs(p[0] - q[0])),
            "G1_abs_err": float(abs(p[1] - q[1])),
            "G2M_abs_err": float(abs(p[2] - q[2])),
        })

    err_df = pd.DataFrame(errs).sort_values(["genotype", "time", "dose", "treatment"])
    out_csv = cfg.results_dir / "validation_vs_cytometry.csv"
    err_df.to_csv(out_csv, index=False)
    logger.info(f"Saved validation table: {out_csv} ({len(err_df)} GT conditions)")

    r_subg1 = pearsonr(merged["SubG1_pred"].values, merged["SubG1_gt"].values)
    r_g1 = pearsonr(merged["G1_pred"].values, merged["G1_gt"].values)
    r_g2m = pearsonr(merged["G2M_pred"].values, merged["G2M_gt"].values)

    metrics = {
        "n_gt_conditions": int(len(merged)),
        "pearson_SubG1": r_subg1,
        "pearson_G1": r_g1,
        "pearson_G2M": r_g2m,
        "mean_MAE_pct": float(err_df["MAE_pct"].mean()),
        "mean_KL_pred||gt": float(err_df["KL_pred||gt"].mean()),
    }
    metrics_path = cfg.results_dir / "validation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info(f"Saved metrics: {metrics_path}")
    logger.info(f"Pearson: SubG1={r_subg1:.3f}, G1={r_g1:.3f}, G2M={r_g2m:.3f}")


# -----------------------------
# Plotting
# -----------------------------

def plot_stacked_pred_vs_gt_time24(cfg: Config, logger: logging.Logger, pred_cond: pd.DataFrame, gt: pd.DataFrame) -> None:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    pred = pred_cond.copy()
    pred["genotype"] = pred["genotype"].astype(str).str.upper()
    pred["treatment"] = pred["treatment"].astype(str).str.upper()

    gt2 = gt.copy()
    gt2["genotype"] = gt2["genotype"].astype(str).str.upper()
    gt2["treatment"] = gt2["treatment"].astype(str).str.upper()

    pred = pred[(pred["time"] == 24) & (pred["treatment"] == "NONE")].copy()
    gt2 = gt2[(gt2["time"] == 24) & (gt2["treatment"] == "NONE")].copy()

    if pred.empty:
        logger.warning("No predicted conditions for time=24, treatment=NONE. Skip plot.")
        return

    genotypes = sorted(pred["genotype"].unique().tolist())
    doses = sorted(pred["dose"].unique().tolist())

    fig, axes = plt.subplots(2, 1, figsize=(max(10, 0.9 * len(doses) * max(1, len(genotypes))), 7), sharex=True)
    ax1, ax2 = axes

    def draw_stacked(ax, df, title, alpha=0.95):
        pos_map = {}
        xpos = []
        xlab = []
        pos = 0.0
        for g in genotypes:
            for d in doses:
                pos_map[(g, d)] = pos
                xpos.append(pos)
                xlab.append(f"{d}Gy")
                pos += 1.0
            pos += 0.75

        bottoms = {k: 0.0 for k in pos_map.keys()}
        for ph in PHASES_3:
            for (g, d), x in pos_map.items():
                sub = df[(df["genotype"] == g) & (df["dose"] == d)]
                h = float(sub.iloc[0][ph]) if len(sub) else 0.0
                ax.bar(x, h, bottom=bottoms[(g, d)], width=0.85, color=PHASE_COLORS[ph],
                       alpha=alpha, edgecolor="white", linewidth=0.6)
                bottoms[(g, d)] += h

        ax.set_ylim(0, 105)
        ax.set_ylabel("% cells")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for g in genotypes:
            xs = [pos_map[(g, d)] for d in doses if (g, d) in pos_map]
            if xs:
                ax.text(np.mean(xs), -10, g, ha="center", va="top")

        return xpos, xlab, pos_map

    draw_stacked(ax1, pred, "Predicted (CNN LLP), time=24h, NONE", alpha=0.95)
    xpos, xlab, pos_map = draw_stacked(ax2, pred, "Predicted + Cytometry GT overlay (0/4/10 only)", alpha=0.35)

    gt_overlay = gt2[["genotype", "dose", "SubG1", "G1", "G2M"]].copy()
    bottoms = {(r["genotype"], int(r["dose"])): 0.0 for _, r in gt_overlay.iterrows()}
    for ph in PHASES_3:
        for _, r in gt_overlay.iterrows():
            g = r["genotype"]
            d = int(r["dose"])
            x = pos_map.get((g, d), None)
            if x is None:
                continue
            h = float(r[ph])
            b = bottoms[(g, d)]
            ax2.bar(x, h, bottom=b, width=0.85, facecolor="none", edgecolor="black",
                   linewidth=1.2, hatch="///")
            bottoms[(g, d)] = b + h

    ax2.set_xticks(xpos)
    ax2.set_xticklabels(xlab)

    handles = [plt.Line2D([0], [0], color=PHASE_COLORS[p], lw=8) for p in PHASES_3]
    ax1.legend(handles, PHASES_3, loc="upper right", frameon=False)

    fig.tight_layout()
    out = cfg.figures_dir / "pred_vs_gt_time24.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    logger.info(f"Saved figure: {out}")


# -----------------------------
# Main
# -----------------------------

def main():
    cfg = make_config()
    logger = setup_logging(cfg)

    logger.info(f"PROJECT_ROOT: {cfg.project_root}")
    logger.info(f"RAW_DIR:      {cfg.raw_dir}")
    logger.info(f"RESULTS_DIR:  {cfg.results_dir}")

    if not cfg.raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {cfg.raw_dir}")

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "config.json").write_text(json.dumps(cfg_to_jsonable(cfg), indent=2), encoding="utf-8")

    gt = flow_cytometry_gt_table()
    gt.to_csv(cfg.results_dir / "cytometry_gt_table_24h.csv", index=False)

    manifest = build_crop_cache(cfg, logger)

    best_model = train_cnn_llp(cfg, logger, manifest, gt)

    pred_cells = predict_all_cells(cfg, logger, manifest, best_model)
    pred_cond = aggregate_condition_percentages(cfg, logger, pred_cells)

    validate_with_cytometry(cfg, logger, pred_cond, gt)
    plot_stacked_pred_vs_gt_time24(cfg, logger, pred_cond, gt)

    logger.info("Done.")


if __name__ == "__main__":
    main()
