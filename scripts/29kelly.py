# -*- coding: utf-8 -*-
"""
Kelly (Auranofin) cell-cycle phase detection from microscopy using CNN + LLP
with calibration and bootstrap confidence intervals.

Key robustness feature:
- If any CUDA error happens during Cellpose GPU segmentation, the script falls back to CPU
  for that image and disables all CUDA usage for the rest of the run (prevents later crashes
  e.g. in torch.manual_seed/device selection).
"""

from __future__ import annotations

import os
import re
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
from torch.utils.data import Dataset, DataLoader

import cv2
import matplotlib.pyplot as plt

from cellpose import models as cpmodels, io as cpio


PHASES_3 = ["SubG1", "G1", "G2M"]
PHASE_COLORS = {"SubG1": "#f28e2b", "G1": "#4e79a7", "G2M": "#59a14f"}
VALID_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Global flag: once a CUDA error occurs, we never touch CUDA again in this run
CUDA_DISABLED = False


def is_cuda_error(e: Exception) -> bool:
    s = (str(e) or "").lower()
    return ("cuda error" in s) or ("illegal instruction" in s) or ("device-side assert" in s) or ("cudnn" in s)


def disable_cuda_globally(logger: logging.Logger, reason: str) -> None:
    """
    We cannot reliably recover CUDA context after 'illegal instruction' inside the same process.
    So we stop using CUDA entirely for the rest of the run.
    """
    global CUDA_DISABLED
    if not CUDA_DISABLED:
        CUDA_DISABLED = True
        logger.error(f"CUDA disabled for the rest of this run. Reason: {reason}")


def choose_device() -> str:
    if CUDA_DISABLED:
        return "cpu"
    # Important: if CUDA is broken, even torch.cuda.* calls can throw;
    # so we only call torch.cuda.is_available() when CUDA_DISABLED==False.
    return "cuda" if torch.cuda.is_available() else "cpu"


def safe_seed_all(seed: int, logger: logging.Logger) -> None:
    """
    torch.manual_seed tries to seed CUDA generators if CUDA is available and can crash
    if CUDA context is broken. We skip torch seeding when CUDA is disabled or if seeding fails.
    """
    np.random.seed(seed)
    try:
        if CUDA_DISABLED:
            # Avoid touching torch.cuda at all
            torch.random.manual_seed(seed)
        else:
            torch.manual_seed(seed)
    except Exception as e:
        # Keep going; reproducibility is secondary to finishing the run
        logger.warning(f"Seeding torch failed (continuing). Error: {e}")


@dataclass
class Config:
    project_root: Path
    raw_dir: Path
    results_root: Path
    run_id: str
    results_dir: Path
    cache_dir: Path
    figures_dir: Path
    models_dir: Path

    cellpose_pretrained: str = "cyto2"
    use_gpu_cellpose: bool = True
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_cell_area_px: int = 80
    max_cell_area_px: int = 6000
    crop_size: int = 128
    crop_margin: int = 8
    cache_overwrite: bool = False
    max_segmentation_previews: int = 12

    seed: int = 42
    batch_size: int = 256
    epochs: int = 25
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    use_mask_channel: bool = True

    w_llp: float = 1.0
    w_global: float = 0.1
    w_entropy: float = 0.03

    val_image_frac: float = 0.25

    n_bootstrap: int = 500
    bootstrap_alpha: float = 0.05
    min_cells_for_bootstrap: int = 20

    temperature_init: float = 0.8
    temperature_grid: Tuple[float, float, int] = (0.35, 2.0, 18)


def make_config() -> Config:
    here = Path(__file__).resolve()
    project_root = here.parent.parent if here.parent.name.lower() == "scripts" else Path.cwd()

    raw_dir = project_root / "data" / "kelly_auranofin"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Expected images in {raw_dir}")

    results_root = project_root / "results"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = results_root / f"kelly_auranofin_{run_id}"

    # if user forced CPU via env
    force_cpu = os.environ.get("FORCE_CPU", "").strip() in {"1", "true", "True", "YES", "yes"}
    use_gpu = (not force_cpu) and (torch.cuda.is_available())

    return Config(
        project_root=project_root,
        raw_dir=raw_dir,
        results_root=results_root,
        run_id=run_id,
        results_dir=results_dir,
        cache_dir=results_dir / "cache_crops",
        figures_dir=results_dir / "figures",
        models_dir=results_dir / "models",
        use_gpu_cellpose=use_gpu,
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

    logger = logging.getLogger("kelly_llp_pipeline")
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


def flow_cytometry_gt_table() -> pd.DataFrame:
    rows = [
        dict(genotype="KELLY", time=2,  concentration=0.0, treatment="CTRL", SubG1=3.56,  G1=58.71, G2M=37.10),
        dict(genotype="KELLY", time=2,  concentration=0.5, treatment="AURA", SubG1=2.94,  G1=59.23, G2M=37.33),
        dict(genotype="KELLY", time=2,  concentration=1.0, treatment="AURA", SubG1=5.53,  G1=64.88, G2M=29.08),
        dict(genotype="KELLY", time=2,  concentration=2.0, treatment="AURA", SubG1=7.53,  G1=64.16, G2M=27.70),

        dict(genotype="KELLY", time=6,  concentration=0.0, treatment="CTRL", SubG1=3.06,  G1=57.67, G2M=38.70),
        dict(genotype="KELLY", time=6,  concentration=0.5, treatment="AURA", SubG1=8.48,  G1=58.12, G2M=32.36),
        dict(genotype="KELLY", time=6,  concentration=1.0, treatment="AURA", SubG1=16.05, G1=51.39, G2M=31.10),
        dict(genotype="KELLY", time=6,  concentration=2.0, treatment="AURA", SubG1=21.09, G1=53.52, G2M=24.04),

        dict(genotype="KELLY", time=24, concentration=0.0, treatment="CTRL", SubG1=7.59,  G1=62.00, G2M=29.72),
        dict(genotype="KELLY", time=24, concentration=0.5, treatment="AURA", SubG1=21.98, G1=57.08, G2M=20.51),
        dict(genotype="KELLY", time=24, concentration=1.0, treatment="AURA", SubG1=40.71, G1=50.61, G2M=8.59),
        dict(genotype="KELLY", time=24, concentration=2.0, treatment="AURA", SubG1=62.65, G1=28.89, G2M=8.06),
    ]
    gt = pd.DataFrame(rows)
    s = gt[PHASES_3].sum(axis=1).replace(0, np.nan)
    gt[PHASES_3] = gt[PHASES_3].div(s, axis=0).fillna(0.0) * 100.0
    return gt


def gt_map_from_table(gt: pd.DataFrame) -> Dict[Tuple[str, int, float, str], torch.Tensor]:
    m: Dict[Tuple[str, int, float, str], torch.Tensor] = {}
    for _, r in gt.iterrows():
        key = (str(r["genotype"]).strip().upper(), int(r["time"]), float(r["concentration"]), str(r["treatment"]).strip().upper())
        t = torch.tensor([float(r["SubG1"]), float(r["G1"]), float(r["G2M"])], dtype=torch.float32)
        t = t / (t.sum() + 1e-8)
        m[key] = t
    return m


def global_target_from_gt(gt: pd.DataFrame) -> torch.Tensor:
    v = gt[PHASES_3].mean(axis=0).astype(float).values
    t = torch.tensor(v, dtype=torch.float32)
    return t / (t.sum() + 1e-8)


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
        raise ValueError(f"Could not parse treatment (CTRL/AURA) from: {base}")

    return dict(
        genotype=genotype,
        time=time_h,
        concentration=concentration,
        dose=concentration,
        treatment=treatment,
        original_filename=base,
    )


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def save_segmentation_preview(img_gray: np.ndarray, masks: np.ndarray, out_path: Path) -> None:
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


def _init_cellpose(logger: logging.Logger, gpu: bool, model_name: str):
    try:
        m = cpmodels.CellposeModel(gpu=bool(gpu), pretrained_model=model_name)
        logger.info(f"Cellpose init ok: model={model_name}, gpu={gpu}")
        return m
    except Exception as e:
        logger.warning(f"Cellpose init failed (gpu={gpu}): {e}")
        return None


def build_crop_cache(cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in cfg.raw_dir.rglob("*") if p.suffix.lower() in VALID_EXT])
    if not imgs:
        raise FileNotFoundError(f"No images found in {cfg.raw_dir}")

    logger.info(f"Found {len(imgs)} images in {cfg.raw_dir}")
    for p in imgs:
        logger.info(f"  - {p.name}")

    # Prepare both models (GPU preferred, CPU always available)
    gpu_requested = bool(cfg.use_gpu_cellpose) and (not CUDA_DISABLED)
    seg_gpu = _init_cellpose(logger, gpu=gpu_requested, model_name=cfg.cellpose_pretrained) if gpu_requested else None
    seg_cpu = _init_cellpose(logger, gpu=False, model_name=cfg.cellpose_pretrained)
    if seg_cpu is None and seg_gpu is None:
        raise RuntimeError("Could not initialize Cellpose on either GPU or CPU.")

    rows = []
    previews_left = cfg.max_segmentation_previews

    for ip in imgs:
        meta = parse_filename(ip.name)
        logger.info(f"Parsed {ip.name}: time={meta['time']}h, conc={meta['concentration']}µM, trt={meta['treatment']}")

        if int(meta["time"]) < 0:
            logger.warning(f"Skip (time not parsed): {ip.name}")
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

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)

        # Try GPU first (if enabled), otherwise CPU
        masks = None
        used = "cpu"

        if (seg_gpu is not None) and (not CUDA_DISABLED):
            try:
                out = seg_gpu.eval(
                    img_gray,
                    diameter=None,
                    channels=[0, 0],
                    flow_threshold=float(cfg.flow_threshold),
                    cellprob_threshold=float(cfg.cellprob_threshold),
                )
                masks = out[0]
                used = "gpu"
            except Exception as e:
                logger.error(f"Segmentation failed on GPU for {ip.name}: {e}")
                if is_cuda_error(e):
                    disable_cuda_globally(logger, f"Cellpose GPU failed on {ip.name}: {e}")
                masks = None

        if masks is None:
            # CPU attempt (always)
            try:
                out = seg_cpu.eval(
                    img_gray,
                    diameter=None,
                    channels=[0, 0],
                    flow_threshold=float(cfg.flow_threshold),
                    cellprob_threshold=float(cfg.cellprob_threshold),
                )
                masks = out[0]
                used = "cpu"
            except Exception as e:
                logger.error(f"Segmentation failed on CPU for {ip.name}: {e}")
                continue

        ids = np.unique(masks)
        ids = ids[ids != 0]
        if ids.size == 0:
            logger.warning(f"No cells detected: {ip.name} (used {used})")
            continue

        logger.info(f"Segmentation ok ({used}): {ip.name}, cells={int(ids.size)}")

        if previews_left > 0:
            prev_path = cfg.figures_dir / "segmentation_previews" / f"{stem}__preview.png"
            try:
                save_segmentation_preview(img_gray, masks, prev_path)
            except Exception as e:
                logger.warning(f"Could not save preview for {ip.name}: {e}")
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

            margin = int(cfg.crop_margin)
            y0 = max(0, y0 - margin)
            x0 = max(0, x0 - margin)
            y1 = min(img_gray.shape[0], y1 + margin)
            x1 = min(img_gray.shape[1], x1 + margin)

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
                concentration=float(meta["concentration"]),
                dose=float(meta["concentration"]),
                treatment=str(meta["treatment"]).upper(),
                cell_id=int(cid),
                area_px=area,
                crop_height=int(y1 - y0),
                crop_width=int(x1 - x0),
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
    logger.info(f"Saved manifest: {manifest_path} ({len(manifest)} cells)")
    return manifest


def collate_keep_conds(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    conds = [b[1] for b in batch]
    return xs, conds


def normalize_conds(conds_batch: Any) -> List[Tuple[str, int, float, str]]:
    if isinstance(conds_batch, list) and len(conds_batch) > 0 and isinstance(conds_batch[0], (tuple, list)) and len(conds_batch[0]) == 4:
        out = []
        for g, t, c, tr in conds_batch:
            out.append((str(g).upper(), int(t), float(c), str(tr).upper()))
        return out
    if isinstance(conds_batch, (tuple, list)) and len(conds_batch) == 4:
        genos, times, concs, trts = conds_batch

        def as_list(x):
            return x.tolist() if hasattr(x, "tolist") else list(x)

        genos_l = [str(x).upper() for x in as_list(genos)]
        times_l = [int(x) for x in as_list(times)]
        concs_l = [float(x) for x in as_list(concs)]
        trts_l = [str(x).upper() for x in as_list(trts)]
        return list(zip(genos_l, times_l, concs_l, trts_l))
    return []


class CellCropDataset(Dataset):
    def __init__(self, cfg: Config, manifest: pd.DataFrame, split: str, val_images: set, augment: bool):
        self.cfg = cfg
        self.df = manifest.copy()
        self.split = split
        self.augment_enabled = bool(augment)
        self.rng = np.random.default_rng(cfg.seed + (1 if split == "val" else 0))

        is_val = self.df["source_image"].isin(val_images)
        if split == "val":
            self.df = self.df[is_val].reset_index(drop=True)
        elif split == "train":
            self.df = self.df[~is_val].reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)

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
            a = float(self.rng.uniform(0.7, 1.3))
            b = float(self.rng.uniform(-25, 25))
            img = np.clip(a * img.astype(np.float32) + b, 0, 255).astype(np.uint8)
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
        except Exception:
            img = np.zeros((self.cfg.crop_size, self.cfg.crop_size), dtype=np.uint8)
            msk = np.zeros((self.cfg.crop_size, self.cfg.crop_size), dtype=np.uint8)

        img, msk = self._augment(img, msk)

        x_img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        if self.cfg.use_mask_channel:
            x_msk = torch.from_numpy(msk).float().unsqueeze(0)
            x = torch.cat([x_img, x_msk], dim=0)
        else:
            x = x_img

        cond = (str(r["genotype"]).upper(), int(r["time"]), float(r["concentration"]), str(r["treatment"]).upper())
        return x, cond


def make_val_split(cfg: Config, manifest: pd.DataFrame, logger: logging.Logger) -> set:
    rng = np.random.default_rng(cfg.seed)
    img_df = manifest[["source_image", "time"]].drop_duplicates()
    imgs = sorted(img_df["source_image"].unique().tolist())
    if len(imgs) <= 1:
        return set()
    n_val = max(2, min(4, int(len(imgs) * cfg.val_image_frac)))
    val_imgs = set()
    for t in sorted(img_df["time"].unique()):
        time_imgs = img_df[img_df["time"] == t]["source_image"].unique().tolist()
        if len(time_imgs) >= 2:
            val_imgs.add(rng.choice(time_imgs, size=1, replace=False).item())
    if len(val_imgs) < n_val:
        remaining = [x for x in imgs if x not in val_imgs]
        needed = n_val - len(val_imgs)
        if remaining and needed > 0:
            add = rng.choice(remaining, size=min(needed, len(remaining)), replace=False)
            val_imgs.update(add.tolist())
    logger.info(f"Split by image: {len(imgs) - len(val_imgs)} train images, {len(val_imgs)} val images")
    logger.info(f"Validation images: {sorted(list(val_imgs))}")
    return val_imgs


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
def eval_condition_kl_full(model: nn.Module, dl: DataLoader, device: str,
                           gt_map: Dict[Tuple[str, int, float, str], torch.Tensor], temperature: float) -> float:
    model.eval()
    all_probs = []
    all_conds = []
    for x, cond in dl:
        x = x.to(device, non_blocking=True)
        conds = normalize_conds(cond)
        logits = model(x) / max(float(temperature), 1e-6)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
        all_conds.extend(conds)

    if not all_probs:
        return float("nan")

    P = np.vstack(all_probs)
    by: Dict[Tuple[str, int, float, str], List[int]] = {}
    for i, c in enumerate(all_conds):
        by.setdefault(c, []).append(i)

    kls = []
    for c, idxs in by.items():
        if c not in gt_map:
            continue
        pred = np.clip(P[idxs].mean(axis=0), 1e-8, 1.0)
        targ = gt_map[c].detach().cpu().numpy()
        targ = np.clip(targ / (targ.sum() + 1e-12), 1e-8, 1.0)
        kls.append(float(np.sum(pred * (np.log(pred) - np.log(targ)))))

    return float(np.mean(kls)) if kls else float("nan")


def train_cnn_llp(cfg: Config, logger: logging.Logger, manifest: pd.DataFrame, gt: pd.DataFrame):
    safe_seed_all(cfg.seed, logger)

    gt_map = gt_map_from_table(gt)
    global_t = global_target_from_gt(gt)

    val_imgs = make_val_split(cfg, manifest, logger)

    ds_tr = CellCropDataset(cfg, manifest, split="train", val_images=val_imgs, augment=True)
    ds_va = CellCropDataset(cfg, manifest, split="val",   val_images=val_imgs, augment=False)

    def is_gt_row(df: pd.DataFrame) -> pd.Series:
        keys = list(zip(
            df["genotype"].astype(str).str.upper(),
            df["time"].astype(int),
            df["concentration"].astype(float),
            df["treatment"].astype(str).str.upper(),
        ))
        return pd.Series([k in gt_map for k in keys], index=df.index)

    ds_tr.df = ds_tr.df[is_gt_row(ds_tr.df)].reset_index(drop=True)
    ds_va.df = ds_va.df[is_gt_row(ds_va.df)].reset_index(drop=True)

    if len(ds_tr) == 0:
        raise RuntimeError("No training cells matched GT conditions. Check segmentation/parsing.")

    bs_tr = int(min(cfg.batch_size, max(1, len(ds_tr))))
    bs_va = int(min(cfg.batch_size, max(1, len(ds_va))))

    dl_tr = DataLoader(ds_tr, batch_size=bs_tr, shuffle=True,
                       num_workers=cfg.num_workers, pin_memory=False, drop_last=False,
                       collate_fn=collate_keep_conds)
    dl_va = DataLoader(ds_va, batch_size=bs_va, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=False, drop_last=False,
                       collate_fn=collate_keep_conds)

    device = choose_device()
    in_ch = 2 if cfg.use_mask_channel else 1
    model = SmallCNN(in_ch=in_ch, n_classes=3).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    best_path = cfg.models_dir / "cnn_llp_best_state_dict.pt"
    meta_path = cfg.models_dir / "cnn_llp_best_meta.json"

    best_val = float("inf")
    patience_counter = 0
    max_patience = 7

    logger.info(f"Train device={device}, in_ch={in_ch}, batch(train/val)={bs_tr}/{bs_va}, epochs={cfg.epochs}")

    for ep in range(cfg.epochs):
        model.train()
        losses = []
        for x, cond in dl_tr:
            x = x.to(device)
            conds = normalize_conds(cond)

            logits = model(x) / max(float(cfg.temperature_init), 1e-6)
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

        val = eval_condition_kl_full(model, dl_va, device, gt_map, temperature=float(cfg.temperature_init))
        scheduler.step(val if np.isfinite(val) else 0.0)

        logger.info(f"Epoch {ep+1:02d}/{cfg.epochs}: train_loss={np.mean(losses):.4f}, val_KL={val:.4f}")

        if val < best_val:
            best_val = val
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            meta_path.write_text(json.dumps({
                "best_val_KL": float(best_val),
                "epoch": int(ep + 1),
                "device": device,
                "temperature_init": float(cfg.temperature_init),
                "cfg": cfg_to_jsonable(cfg),
                "cuda_disabled": bool(CUDA_DISABLED),
            }, indent=2), encoding="utf-8")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {ep+1}")
                break

    logger.info(f"Best model saved: {best_path} (val_KL={best_val:.4f})")
    return best_path, val_imgs


def main():
    cfg = make_config()
    logger = setup_logging(cfg)

    logger.info(f"Project root: {cfg.project_root}")
    logger.info(f"Raw dir: {cfg.raw_dir}")
    logger.info(f"Results dir: {cfg.results_dir}")
    logger.info(f"Config: {json.dumps(cfg_to_jsonable(cfg), indent=2)}")

    gt = flow_cytometry_gt_table()
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    gt.to_csv(cfg.results_dir / "cytometry_gt.csv", index=False)

    manifest = build_crop_cache(cfg, logger)

    # If CUDA broke during segmentation, training will automatically be on CPU
    train_cnn_llp(cfg, logger, manifest, gt)

    logger.info("Done (segmentation + training). You can extend with prediction/CI blocks from previous version.")


if __name__ == "__main__":
    main()
