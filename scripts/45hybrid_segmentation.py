#!/usr/bin/env python3
"""
45hybrid_segmentation.py - Гибрид: берет лучшее из предыдущих версий
"""

import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def hybrid_preprocess(img_rgb):
    """Комбинированная предобработка"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # CLAHE (из первой версии - работала)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Легкий Laplacian (контуры)
    laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian) * 2)
    return cv2.addWeighted(enhanced, 0.8, laplacian, 0.2, 0)


def hybrid_segmentation(gray):
    """3 метода параллельно → выбираем лучший"""
    methods = []

    # Метод 1: Otsu (из первой версии)
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=1)
    methods.append(opened1)

    # Метод 2: Adaptive threshold
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 41, 15)
    methods.append(thresh2)

    # Метод 3: Local variance (phase contrast специфика)
    kernel_size = 15
    local_var = cv2.Laplacian(gray, cv2.CV_64F)
    local_var = np.uint8(np.absolute(local_var))
    _, thresh3 = cv2.threshold(local_var, 20, 255, cv2.THRESH_BINARY)
    methods.append(thresh3)

    # Объединяем все методы
    combined = np.zeros_like(gray)
    for method in methods:
        combined = cv2.bitwise_or(combined, method)

    # Мягкая морфология
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    # Connected components (без Watershed - стабильнее)
    num_labels, labels = cv2.connectedComponents(cleaned)

    # Фильтр размера (очень мягкий!)
    masks = np.zeros_like(labels, dtype=np.int32)
    for i in range(1, num_labels):
        component = (labels == i)
        area = np.sum(component)
        if 50 <= area <= 8000:  # Широкий диапазон!
            masks[component] = i

    n_cells = len(np.unique(masks)) - 1
    return masks, n_cells


def segment_and_save(image_path: Path, output_dir: Path):
    masks_dir = output_dir / "masks"
    visuals_dir = output_dir / "visuals"
    data_dir = output_dir / "data"

    for d in [masks_dir, visuals_dir, data_dir]:
        d.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Processing: {image_path.name} | {img_rgb.shape}")

    enhanced = hybrid_preprocess(img_rgb)
    masks, n_cells = hybrid_segmentation(enhanced)
    print(f"  → Found {n_cells} cells")

    # Имя файла
    stem = "".join(c for c in image_path.stem if c.isalnum() or c in "._-")
    npz_path = data_dir / f"{stem}_masks.npz"
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

    # 4 панели для отладки
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

    return {"image": image_path.name, "n_cells": n_cells}


def main():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "results/01_manifest/kelly_auranofin_manifest.csv"
    output_dir = root / "results/02_segmentation_hybrid"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    image_paths = [Path(r["abs_path"]) for _, r in df.iterrows()]

    print(f"🔥 HYBRID SEGMENTATION (3 метода → лучший) → {output_dir}")

    results = []
    for img_path in tqdm(image_paths, desc="Segmenting"):
        result = segment_and_save(img_path, output_dir)
        results.append(result)

    summary_df = pd.DataFrame(results)
    summary_path = output_dir / "segmentation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    total_cells = summary_df["n_cells"].sum()
    print(f"\n🎯 === HYBRID RESULTS ===")
    print(summary_df[["image", "n_cells"]].to_string(index=False))
    print(f"TOTAL CELLS: {total_cells:,}")
    print(f"📊 Summary: {summary_path}")
    print(f"🖼️  Previews: {output_dir / 'visuals'} ← КРИТИЧНО ПРОВЕРИ!")


if __name__ == "__main__":
    main()
