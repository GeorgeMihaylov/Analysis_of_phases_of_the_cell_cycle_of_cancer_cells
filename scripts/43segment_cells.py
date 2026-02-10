#!/usr/bin/env python3
"""
43segment_cells_fast.py - Watershed исправленная версия
Работает 1-2 сек/изображение, 12 изображений = 30 сек
"""

import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def preprocess_phase_contrast(img: np.ndarray) -> np.ndarray:
    """CLAHE + Laplacian для phase contrast"""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    gray = gray.astype(np.float32)
    if gray.max() > gray.min():
        gray = (gray - gray.min()) / (gray.max() - gray.min()) * 255
    gray = gray.astype(np.uint8)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Laplacian contours
    laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    return cv2.addWeighted(enhanced, 0.7, laplacian, 0.3, 0)


def threshold_segmentation(gray: np.ndarray, min_size: int = 300) -> np.ndarray:
    """Исправленный Watershed для OpenCV"""
    # Бинаризация Otsu (инвертируем - темные клетки)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Морфология
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Distance transform
    dist_transform = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(closed, sure_fg)

    # MARKERS - ИСПРАВЛЕНО: cv2.connectedComponents вместо skimage
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers.astype(np.int32)

    # Unknown regions → 0
    markers[unknown == 255] = 0

    # Watershed ожидает uint8 для входа, но markers int32
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    markers_watershed = cv2.watershed(gray_3ch, markers)

    # Преобразуем в маски (отрицательные значения → фон)
    masks = np.int32(markers_watershed) - 1
    masks[masks < 0] = 0

    # Фильтр по размеру
    for label in range(1, num_labels):
        component = (masks == label)
        if np.sum(component) < min_size:
            masks[component] = 0

    return masks


def segment_and_save(image_path: Path, output_dir: Path):
    masks_dir = output_dir / "masks"
    visuals_dir = output_dir / "visuals"
    data_dir = output_dir / "data"

    for d in [masks_dir, visuals_dir, data_dir]:
        d.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Processing: {image_path.name} | {img_rgb.shape}")

    gray = preprocess_phase_contrast(img_rgb)
    masks = threshold_segmentation(gray)
    n_cells = len(np.unique(masks)) - 1
    print(f"  → Found {n_cells} cells")

    # Безопасное имя
    stem = "".join(c for c in image_path.stem if c.isalnum() or c in "._-")
    npz_path = data_dir / f"{stem}_masks.npz"

    # Сохраняем NPZ
    np.savez_compressed(npz_path, masks=masks, n_cells=n_cells, img_shape=img_rgb.shape)

    # Overlay
    overlay = img_rgb.copy()
    if n_cells > 0:
        boundaries = np.zeros(masks.shape, dtype=np.uint8)
        for label in np.unique(masks):
            if label <= 0: continue
            mask_label = (masks == label).astype(np.uint8)
            contours, _ = cv2.findContours(mask_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(boundaries, contours, -1, 255, 2)
        overlay[boundaries > 0] = [0, 255, 0]

    overlay_path = visuals_dir / f"{stem}_overlay.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Preview
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb);
    axes[0].set_title('Original');
    axes[0].axis('off')
    axes[1].imshow(masks, cmap='tab20c');
    axes[1].set_title(f'Masks ({n_cells})');
    axes[1].axis('off')
    axes[2].imshow(overlay);
    axes[2].set_title('Overlay');
    axes[2].axis('off')
    plt.tight_layout()
    preview_path = visuals_dir / f"{stem}_preview.png"
    plt.savefig(str(preview_path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return {"image": image_path.name, "n_cells": n_cells, "npz": npz_path.name}


def main():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "results/01_manifest/kelly_auranofin_manifest.csv"
    output_dir = root / "results/02_segmentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    image_paths = [Path(r["abs_path"]) for _, r in df.iterrows()]

    print(f"Fast segmentation of {len(image_paths)} images → {output_dir}")

    results = []
    for img_path in tqdm(image_paths, desc="Segmenting"):
        if not img_path.exists():
            results.append({"image": img_path.name, "n_cells": 0, "error": "missing"})
            continue
        result = segment_and_save(img_path, output_dir)
        results.append(result)

    summary_df = pd.DataFrame(results)
    summary_path = output_dir / "segmentation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    total_cells = summary_df["n_cells"].sum()
    print(f"\n=== SUMMARY ===")
    print(summary_df[["image", "n_cells"]].to_string(index=False))
    print(f"TOTAL CELLS: {total_cells:,}")
    print(f"Summary: {summary_path}")
    print(f"Previews: {output_dir / 'visuals'}")


if __name__ == "__main__":
    main()
