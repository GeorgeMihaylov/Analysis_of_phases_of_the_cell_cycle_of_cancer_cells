#!/usr/bin/env python3
"""
44improved_segmentation.py - Агрессивная сегментация phase contrast
Находит 10x больше клеток за счет multi-scale + adaptive threshold
"""

import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def aggressive_preprocess(img_rgb: np.ndarray) -> np.ndarray:
    """Многоступенчатая предобработка для phase contrast"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Background normalization
    gray = cv2.GaussianBlur(gray, (101, 101), 0)
    gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # 2. Multi-scale DoG (Difference of Gaussians)
    blurred1 = cv2.GaussianBlur(gray_norm, (9, 9), 2)
    blurred2 = cv2.GaussianBlur(gray_norm, (15, 15), 5)
    dog = blurred1.astype(float) - blurred2.astype(float)
    dog = np.clip(dog, 0, 255).astype(np.uint8)

    # 3. CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(dog)

    # 4. Morphological gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel)

    return gradient


def multi_scale_segmentation(enhanced: np.ndarray, min_size: int = 100) -> np.ndarray:
    """Multi-scale adaptive thresholding + Watershed"""

    # Multi-scale adaptive threshold
    binary1 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 51, 10)

    binary2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 8)

    # Combine scales
    combined = cv2.bitwise_or(binary1, binary2)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Distance transform + markers
    dist_transform = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(closed, sure_fg)

    # Connected components для markers
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers.astype(np.int32)
    markers[unknown == 255] = 0

    # Watershed
    gray_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    markers_watershed = cv2.watershed(gray_bgr, markers)
    masks = np.int32(markers_watershed) - 1
    masks[masks < 0] = 0

    # Size filter (более мягкий)
    for label in range(1, num_labels):
        component = (masks == label)
        area = np.sum(component)
        if area < min_size or area > 5000:  # max_size добавлен
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

    # Агрессивная предобработка
    enhanced = aggressive_preprocess(img_rgb)
    masks = multi_scale_segmentation(enhanced, min_size=80)
    n_cells = len(np.unique(masks)) - 1
    print(f"  → Found {n_cells} cells")

    # Безопасное имя
    stem = "".join(c for c in image_path.stem if c.isalnum() or c in "._-")
    npz_path = data_dir / f"{stem}_masks.npz"

    np.savez_compressed(npz_path, masks=masks, n_cells=n_cells, img_shape=img_rgb.shape)

    # Overlay с толстой линией
    overlay = img_rgb.copy()
    if n_cells > 0:
        boundaries = np.zeros(masks.shape, dtype=np.uint8)
        for label in np.unique(masks):
            if label <= 0: continue
            mask_label = (masks == label).astype(np.uint8)
            contours, _ = cv2.findContours(mask_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(boundaries, contours, -1, 255, 3)  # толще!
        overlay[boundaries > 0] = [0, 255, 0]

    overlay_path = visuals_dir / f"{stem}_overlay.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Preview
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_rgb);
    axes[0].set_title('Original');
    axes[0].axis('off')
    axes[1].imshow(enhanced, cmap='gray');
    axes[1].set_title('Enhanced');
    axes[1].axis('off')
    axes[2].imshow(masks, cmap='tab20c');
    axes[2].set_title(f'Masks ({n_cells})');
    axes[2].axis('off')
    axes[3].imshow(overlay);
    axes[3].set_title('Overlay');
    axes[3].axis('off')
    plt.tight_layout()
    preview_path = visuals_dir / f"{stem}_preview.png"
    plt.savefig(str(preview_path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return {"image": image_path.name, "n_cells": n_cells, "npz": npz_path.name}


def main():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "results/01_manifest/kelly_auranofin_manifest.csv"
    output_dir = root / "results/02_segmentation_improved"  # новая папка!
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    image_paths = [Path(r["abs_path"]) for _, r in df.iterrows()]

    print(f"🚀 AGGRESSIVE SEGMENTATION of {len(image_paths)} images → {output_dir}")

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
    print(f"\n🎯 === RESULTS ===")
    print(summary_df[["image", "n_cells"]].to_string(index=False))
    print(f"TOTAL CELLS: {total_cells:,} ✅")
    print(f"Summary: {summary_path}")
    print(f"Previews: {output_dir / 'visuals'} ← ПРОВЕРИ КАЧЕСТВО!")


if __name__ == "__main__":
    main()
