# -*- coding: utf-8 -*-
"""
Упрощенный скрипт для Kelly dataset - CPU для Cellpose, GPU для PyTorch
"""

from __future__ import annotations
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Для отладки ошибок CUDA

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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import cv2
import matplotlib.pyplot as plt

# Импортируем Cellpose только когда нужно
# -----------------------------
# Constants
# -----------------------------

PHASES_3 = ["SubG1", "G1", "G2M"]
PHASE_COLORS = {"SubG1": "#f28e2b", "G1": "#4e79a7", "G2M": "#59a14f"}
VALID_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# -----------------------------
# Config - упрощенный
# -----------------------------

@dataclass
class Config:
    project_root: Path
    raw_dir: Path
    results_dir: Path
    cache_dir: Path
    figures_dir: Path
    models_dir: Path

    # Segmentation
    min_cell_area_px: int = 80
    max_cell_area_px: int = 5000
    crop_size: int = 96  # Уменьшим для скорости
    crop_margin: int = 5

    # CNN training
    seed: int = 42
    batch_size: int = 32  # Маленький batch для отладки
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0  # 0 для Windows

    # Простые настройки
    use_mask_channel: bool = True


def make_config() -> Config:
    here = Path(__file__).resolve()
    project_root = here.parent.parent if here.parent.name.lower() == "scripts" else Path.cwd()

    raw_dir = project_root / "data" / "kelly_auranofin"
    if not raw_dir.exists():
        raw_dir = project_root / "data" / "raw"

    results_root = project_root / "results"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = results_root / f"kelly_simple_{run_id}"

    return Config(
        project_root=project_root,
        raw_dir=raw_dir,
        results_dir=results_dir,
        cache_dir=results_dir / "cache_crops",
        figures_dir=results_dir / "figures",
        models_dir=results_dir / "models",
    )


# -----------------------------
# GT данные для Kelly
# -----------------------------

def flow_cytometry_gt_table() -> pd.DataFrame:
    rows = [
        dict(genotype="Kelly", time=2, concentration=0.0, treatment="NONE", SubG1=3.56, G1=58.71, G2M=37.10),
        dict(genotype="Kelly", time=2, concentration=0.5, treatment="Aura", SubG1=2.94, G1=59.23, G2M=37.33),
        dict(genotype="Kelly", time=2, concentration=1.0, treatment="Aura", SubG1=5.53, G1=64.88, G2M=29.08),
        dict(genotype="Kelly", time=2, concentration=2.0, treatment="Aura", SubG1=7.53, G1=64.16, G2M=27.70),

        dict(genotype="Kelly", time=6, concentration=0.0, treatment="NONE", SubG1=3.06, G1=57.67, G2M=38.70),
        dict(genotype="Kelly", time=6, concentration=0.5, treatment="Aura", SubG1=8.48, G1=58.12, G2M=32.36),
        dict(genotype="Kelly", time=6, concentration=1.0, treatment="Aura", SubG1=16.05, G1=51.39, G2M=31.10),
        dict(genotype="Kelly", time=6, concentration=2.0, treatment="Aura", SubG1=21.09, G1=53.52, G2M=24.04),

        dict(genotype="Kelly", time=24, concentration=0.0, treatment="NONE", SubG1=7.59, G1=62.00, G2M=29.72),
        dict(genotype="Kelly", time=24, concentration=0.5, treatment="Aura", SubG1=21.98, G1=57.08, G2M=20.51),
        dict(genotype="Kelly", time=24, concentration=1.0, treatment="Aura", SubG1=40.71, G1=50.61, G2M=8.59),
        dict(genotype="Kelly", time=24, concentration=2.0, treatment="Aura", SubG1=62.65, G1=28.89, G2M=8.06),
    ]

    gt = pd.DataFrame(rows)
    return gt


# -----------------------------
# Простой парсер имен файлов
# -----------------------------

def parse_filename_simple(fname: str) -> Dict[str, object]:
    base = Path(fname).name.lower()

    genotype = "Kelly"
    time_h = -1
    concentration = 0.0
    treatment = "NONE"

    # Время
    if "2h" in base:
        time_h = 2
    elif "6h" in base:
        time_h = 6
    elif "24h" in base:
        time_h = 24

    # Лечение
    if "aura" in base:
        treatment = "Aura"
        if "0.5" in base:
            concentration = 0.5
        elif "1" in base and "10" not in base and "12" not in base:
            concentration = 1.0
        elif "2" in base:
            concentration = 2.0

    return {
        "genotype": genotype,
        "time": time_h,
        "concentration": concentration,
        "treatment": treatment,
    }


# -----------------------------
# Простая сегментация без Cellpose (threshold-based)
# -----------------------------

def simple_segmentation(img_gray: np.ndarray, min_area: int = 50, max_area: int = 5000) -> np.ndarray:
    """Простая сегментация на основе порога и морфологических операций"""
    # Автоматический порог
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Морфологические операции для очистки
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Находим контуры
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создаем маску с маркированными объектами
    masks = np.zeros_like(img_gray, dtype=np.int32)
    cell_id = 1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            cv2.drawContours(masks, [cnt], -1, cell_id, -1)
            cell_id += 1

    return masks


def build_crop_cache_simple(cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in cfg.raw_dir.rglob("*") if p.suffix.lower() in VALID_EXT])
    if not imgs:
        raise FileNotFoundError(f"No images found in {cfg.raw_dir}")

    logger.info(f"Found {len(imgs)} images")

    rows = []

    for ip in imgs:
        meta = parse_filename_simple(ip.name)

        if meta["time"] < 0:
            logger.warning(f"Skipping {ip.name} - cannot parse time")
            continue

        stem = ip.stem.replace(")", "").replace("(", "").replace(" ", "_")

        logger.info(f"Processing: {ip.name} - time: {meta['time']}h, conc: {meta['concentration']}µM")

        # Загружаем изображение
        img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error(f"Cannot read image: {ip}")
            continue

        # Простая сегментация
        masks = simple_segmentation(img, cfg.min_cell_area_px, cfg.max_cell_area_px)
        ids = np.unique(masks)
        ids = ids[ids != 0]

        if ids.size == 0:
            logger.warning(f"No cells detected in {ip.name}")
            continue

        logger.info(f"Found {len(ids)} cells in {ip.name}")

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
            y1 = min(img.shape[0], y1 + cfg.crop_margin)
            x1 = min(img.shape[1], x1 + cfg.crop_margin)

            crop_img = img[y0:y1, x0:x1]
            crop_msk = mask[y0:y1, x0:x1].astype(np.uint8)

            crop_img_r = cv2.resize(crop_img, (cfg.crop_size, cfg.crop_size), interpolation=cv2.INTER_AREA)
            crop_msk_r = cv2.resize(crop_msk, (cfg.crop_size, cfg.crop_size), interpolation=cv2.INTER_NEAREST)

            out_name = f"{stem}__cell{int(cid):04d}.npz"
            out_path = cfg.cache_dir / out_name
            np.savez_compressed(out_path, img=crop_img_r.astype(np.uint8), mask=crop_msk_r.astype(np.uint8))

            rec = {
                "crop_path": str(out_path.relative_to(cfg.results_dir)),
                "source_image": str(ip.relative_to(cfg.project_root)),
                "filename": ip.name,
                "genotype": "KELLY",
                "time": int(meta["time"]),
                "concentration": float(meta["concentration"]),
                "treatment": str(meta["treatment"]).upper(),
                "cell_id": int(cid),
                "area_px": area,
            }
            rows.append(rec)
            per_image_rows.append(rec)

        logger.info(f"Cached {len(per_image_rows)} cells from {ip.name}")

    if not rows:
        raise RuntimeError("No cells were segmented from any image.")

    manifest = pd.DataFrame(rows)
    manifest_path = cfg.results_dir / "manifest_cells.csv"
    manifest.to_csv(manifest_path, index=False)

    logger.info(f"Saved manifest: {manifest_path} ({len(manifest)} cells)")

    # Логируем распределение
    logger.info("Distribution of cells:")
    for time_val in sorted(manifest["time"].unique()):
        for conc_val in sorted(manifest[manifest["time"] == time_val]["concentration"].unique()):
            n = len(manifest[(manifest["time"] == time_val) & (manifest["concentration"] == conc_val)])
            logger.info(f"  Time {time_val}h, Conc {conc_val}µM: {n} cells")

    return manifest


# -----------------------------
# Простая CNN модель
# -----------------------------

class SimpleCNN(nn.Module):
    def __init__(self, in_ch: int = 2, n_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # 96 -> 48 -> 24 -> 12
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# -----------------------------
# Dataset
# -----------------------------

class CellCropDataset(Dataset):
    def __init__(self, cfg: Config, manifest: pd.DataFrame):
        self.cfg = cfg
        self.df = manifest.copy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        crop_abs = self.cfg.results_dir / Path(r["crop_path"])

        dat = np.load(crop_abs)
        img = dat["img"].astype(np.float32) / 255.0
        msk = dat["mask"].astype(np.float32)

        if self.cfg.use_mask_channel:
            x = np.stack([img, msk], axis=0)
        else:
            x = img.reshape(1, *img.shape)

        x = torch.from_numpy(x).float()

        # One-hot кодирование условий для простой loss
        cond_features = np.array([
            r["time"] / 24.0,  # Нормализованное время
            r["concentration"] / 2.0,  # Нормализованная концентрация
            1.0 if r["treatment"] == "AURA" else 0.0
        ], dtype=np.float32)

        cond_features = torch.from_numpy(cond_features).float()

        return x, cond_features


# -----------------------------
# Простое обучение с MSE loss
# -----------------------------

def train_simple_model(cfg: Config, logger: logging.Logger, manifest: pd.DataFrame, gt: pd.DataFrame) -> Path:
    """Упрощенное обучение с MSE loss между предсказанными процентами и GT"""

    # Подготовка GT данных
    gt_map = {}
    for _, row in gt.iterrows():
        key = (row["genotype"], row["time"], row["concentration"], row["treatment"])
        gt_map[key] = torch.tensor([row["SubG1"], row["G1"], row["G2M"]], dtype=torch.float32) / 100.0

    # Создаем Dataset и DataLoader
    dataset = CellCropDataset(cfg, manifest)

    # Разделяем на train/val
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers
    )

    # Модель и оптимизатор
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model = SimpleCNN(in_ch=2 if cfg.use_mask_channel else 1, n_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    # Сохраняем лучшую модель
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    best_path = cfg.models_dir / "simple_cnn_best.pth"
    best_val_loss = float('inf')

    logger.info(f"Training on {n_train} cells, validating on {n_val} cells")

    for epoch in range(cfg.epochs):
        # Training
        model.train()
        train_loss = 0.0
        for x, cond in train_loader:
            x = x.to(device)

            # Получаем GT для батча
            batch_gt = []
            for i in range(len(x)):
                # Находим соответствующий GT
                idx = train_dataset.indices[i % len(train_dataset)] if hasattr(train_dataset, 'indices') else i
                row = manifest.iloc[idx]
                key = (row["genotype"], row["time"], row["concentration"], row["treatment"])

                if key in gt_map:
                    batch_gt.append(gt_map[key])
                else:
                    # Если нет GT, используем равномерное распределение
                    batch_gt.append(torch.tensor([0.33, 0.33, 0.34], dtype=torch.float32))

            targets = torch.stack(batch_gt).to(device)

            optimizer.zero_grad()
            outputs = torch.softmax(model(x), dim=1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, cond in val_loader:
                x = x.to(device)

                # Получаем GT для батча
                batch_gt = []
                for i in range(len(x)):
                    idx = val_dataset.indices[i % len(val_dataset)] if hasattr(val_dataset, 'indices') else i
                    row = manifest.iloc[idx]
                    key = (row["genotype"], row["time"], row["concentration"], row["treatment"])

                    if key in gt_map:
                        batch_gt.append(gt_map[key])
                    else:
                        batch_gt.append(torch.tensor([0.33, 0.33, 0.34], dtype=torch.float32))

                targets = torch.stack(batch_gt).to(device)
                outputs = torch.softmax(model(x), dim=1)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        logger.info(f"Epoch {epoch + 1}/{cfg.epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            logger.info(f"  -> Saved new best model (val loss: {val_loss:.4f})")

    logger.info(f"Best model saved to: {best_path}")
    return best_path


# -----------------------------
# Предсказание
# -----------------------------

@torch.no_grad()
def predict_simple(cfg: Config, logger: logging.Logger, manifest: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleCNN(in_ch=2 if cfg.use_mask_channel else 1, n_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    dataset = CellCropDataset(cfg, manifest)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    all_preds = []

    for x, cond in loader:
        x = x.to(device)
        outputs = torch.softmax(model(x), dim=1).cpu().numpy()

        for i in range(len(outputs)):
            all_preds.append({
                "p_SubG1": float(outputs[i, 0]),
                "p_G1": float(outputs[i, 1]),
                "p_G2M": float(outputs[i, 2]),
                "phase_pred": PHASES_3[np.argmax(outputs[i])]
            })

    preds_df = pd.DataFrame(all_preds)
    pred_df = pd.concat([manifest.reset_index(drop=True), preds_df], axis=1)

    # Сохраняем предсказания
    out_csv = cfg.results_dir / "cell_predictions.csv"
    pred_df.to_csv(out_csv, index=False)
    logger.info(f"Saved predictions: {out_csv}")

    # Агрегируем по условиям
    agg_cols = ["time", "concentration", "treatment"]
    agg_df = pred_df.groupby(agg_cols).agg({
        "p_SubG1": "mean",
        "p_G1": "mean",
        "p_G2M": "mean",
        "cell_id": "count"
    }).reset_index()

    agg_df.rename(columns={
        "p_SubG1": "SubG1_pred",
        "p_G1": "G1_pred",
        "p_G2M": "G2M_pred",
        "cell_id": "n_cells"
    }, inplace=True)

    agg_df[["SubG1_pred", "G1_pred", "G2M_pred"]] *= 100

    agg_csv = cfg.results_dir / "aggregated_predictions.csv"
    agg_df.to_csv(agg_csv, index=False)
    logger.info(f"Saved aggregated predictions: {agg_csv}")

    return pred_df, agg_df


# -----------------------------
# Визуализация
# -----------------------------

def plot_results(cfg: Config, logger: logging.Logger, agg_df: pd.DataFrame, gt_df: pd.DataFrame):
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    # Объединяем предсказания и GT
    merged = pd.merge(
        agg_df,
        gt_df,
        left_on=["time", "concentration", "treatment"],
        right_on=["time", "concentration", "treatment"],
        suffixes=("_pred", "_gt")
    )

    if merged.empty:
        logger.warning("No matching conditions for plotting")
        return

    # Scatter plot для каждой фазы
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, phase in enumerate(PHASES_3):
        ax = axes[idx]
        x = merged[f"{phase}_gt"].values
        y = merged[f"{phase}_pred"].values

        ax.scatter(x, y, alpha=0.7)
        ax.plot([0, 100], [0, 100], 'r--', alpha=0.5)
        ax.set_xlabel(f"GT {phase} (%)")
        ax.set_ylabel(f"Predicted {phase} (%)")
        ax.set_title(phase)
        ax.grid(True, alpha=0.3)

        # Вычисляем R^2
        if len(x) > 1:
            r = np.corrcoef(x, y)[0, 1]
            ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(cfg.figures_dir / "scatter_predictions.png", dpi=150)
    plt.close()

    # График по времени
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    times = sorted(merged["time"].unique())

    for idx, phase in enumerate(PHASES_3):
        ax = axes[idx]

        for time_val in times:
            time_data = merged[merged["time"] == time_val]
            if len(time_data) > 0:
                concs = time_data["concentration"].values
                preds = time_data[f"{phase}_pred"].values
                gts = time_data[f"{phase}_gt"].values

                # Сортируем по концентрации
                sort_idx = np.argsort(concs)
                concs = concs[sort_idx]
                preds = preds[sort_idx]
                gts = gts[sort_idx]

                ax.plot(concs, preds, 'o-', label=f"{time_val}h Pred", alpha=0.7)
                ax.plot(concs, gts, 's--', label=f"{time_val}h GT", alpha=0.7)

        ax.set_ylabel(f"{phase} (%)")
        ax.set_title(f"{phase} phase")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Concentration (µM)")
    plt.tight_layout()
    plt.savefig(cfg.figures_dir / "time_concentration_response.png", dpi=150)
    plt.close()

    logger.info(f"Plots saved to {cfg.figures_dir}")


# -----------------------------
# Main
# -----------------------------

def main():
    # Устанавливаем логирование
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("kelly_analysis.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Создаем конфиг
    cfg = make_config()
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting Kelly Auranofin Analysis")
    logger.info("=" * 60)

    # Шаг 1: Загружаем GT данные
    logger.info("Loading GT data...")
    gt_df = flow_cytometry_gt_table()

    # Шаг 2: Сегментация и создание кэша
    logger.info("\nStep 1: Segmentation and crop caching...")
    try:
        manifest = build_crop_cache_simple(cfg, logger)
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        return

    # Шаг 3: Обучение модели
    logger.info("\nStep 2: Training model...")
    try:
        model_path = train_simple_model(cfg, logger, manifest, gt_df)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return

    # Шаг 4: Предсказание
    logger.info("\nStep 3: Making predictions...")
    try:
        pred_df, agg_df = predict_simple(cfg, logger, manifest, model_path)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return

    # Шаг 5: Визуализация
    logger.info("\nStep 4: Creating visualizations...")
    try:
        plot_results(cfg, logger, agg_df, gt_df)
    except Exception as e:
        logger.error(f"Plotting failed: {e}")

    # Шаг 6: Сохранение результатов
    logger.info("\nStep 5: Saving final results...")

    # Сохраняем GT данные
    gt_df.to_csv(cfg.results_dir / "ground_truth.csv", index=False)

    # Создаем простой отчет
    report = f"""
    Kelly Auranofin Analysis Report
    ================================

    Total cells processed: {len(manifest)}
    Conditions analyzed: {len(agg_df)}

    Files processed:
    {chr(10).join([f"  - {f}" for f in manifest['filename'].unique()])}

    Results saved to: {cfg.results_dir}
    """

    with open(cfg.results_dir / "report.txt", "w") as f:
        f.write(report)

    logger.info("=" * 60)
    logger.info("Analysis completed successfully!")
    logger.info(f"Results saved to: {cfg.results_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()