#!/usr/bin/env python3
"""
46build_cell_crops.py - Извлекает кропы клеток из масок
Input: results/02_segmentation_hybrid/data/*.npz
Output: results/03_crops/crops/*.npz + manifest_cells.csv
"""

import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re
from typing import Dict, List, Tuple


def parse_filename(filename: str) -> Dict:
    """Парсит Kelly имена как в 38buildcropsfrommasksfixed.py"""
    name = Path(filename).stem.lower()

    # Time: 2h/6h/24h
    time_match = re.search(r'(\d+)h', name)
    time_h = int(time_match.group(1)) if time_match else 2

    # Concentration + treatment
    concentration = 0.0
    treatment = "CTRL"
    if 'aura' in name:
        treatment = "AURA"
        conc_match = re.search(r'(\d+(?:\.\d+)?)u?m?', name)
        if conc_match:
            concentration = float(conc_match.group(1))
        else:
            concentration = 0.5  # default

    genotype = "KELLY"
    return {
        "genotype": genotype,
        "time": time_h,
        "concentration": concentration,
        "treatment": treatment,
        "source_image": filename
    }


def extract_cell_crops(mask_path: Path, image_path: Path, output_dir: Path,
                       min_size: int = 50, max_size: int = 300, padding: int = 15,
                       target_size: int = 128) -> List[Dict]:
    """Извлекает кропы точно как в 38buildcropsfrommasksfixed.py"""
    crops_info = []

    try:
        data = np.load(mask_path)
        masks = data['masks']
    except:
        print(f"  → Cannot load masks: {mask_path}")
        return crops_info

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  → Cannot load image: {image_path}")
        return crops_info

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    conditions = parse_filename(image_path.name)

    labels = np.unique(masks)
    labels = labels[labels > 0]  # exclude background

    for label in tqdm(labels, desc=f"Extracting from {mask_path.stem}", leave=False):
        cell_mask = (masks == label).astype(np.uint8)

        # Bounding box
        rows, cols = np.where(cell_mask)
        if len(rows) == 0:
            continue

        ymin, ymax = rows.min(), rows.max()
        xmin, xmax = cols.min(), cols.max()

        h, w = ymax - ymin, xmax - xmin
        if h < min_size or w < min_size or h > max_size or w > max_size:
            continue

        # Padding
        pad_ymin = max(0, ymin - padding)
        pad_ymax = min(masks.shape[0], ymax + padding)
        pad_xmin = max(0, xmin - padding)
        pad_xmax = min(masks.shape[1], xmax + padding)

        # Crop
        img_crop = img[pad_ymin:pad_ymax, pad_xmin:pad_xmax]
        mask_crop = cell_mask[pad_ymin:pad_ymax, pad_xmin:pad_xmax]

        # Resize to 128x128
        img_resized = cv2.resize(img_crop, (target_size, target_size),
                                 interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask_crop, (target_size, target_size),
                                  interpolation=cv2.INTER_NEAREST)

        # Normalize image
        img_resized = img_resized.astype(np.float32)
        img_resized = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8) * 255
        img_resized = img_resized.astype(np.uint8)

        # Save NPZ
        crop_id = f"{Path(image_path).stem}_cell{label:04d}"
        crop_path = output_dir / "crops" / f"{crop_id}.npz"
        crop_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(crop_path,
                            img=img_resized,
                            mask=mask_resized,
                            label=label,
                            bbox=(pad_xmin, pad_ymin, pad_xmax - pad_xmin, pad_ymax - pad_ymin),
                            original_shape=img.shape,
                            conditions=conditions)

        crop_info = {
            "cell_id": crop_id,
            "crop_path": str(crop_path.relative_to(output_dir.parent)),
            "source_image": image_path.name,
            "conditions": str(conditions),
            "bbox_x": int(pad_xmin),
            "bbox_y": int(pad_ymin),
            "bbox_w": int(pad_xmax - pad_xmin),
            "bbox_h": int(pad_ymax - pad_ymin),
            "area_pixels": int(np.sum(cell_mask))
        }
        crops_info.append(crop_info)

    return crops_info


def main():
    root = Path(__file__).resolve().parents[1]

    # Input: маски из hybrid сегментации
    seg_dir = root / "results/02_segmentation_hybrid/data"
    manifest_path = root / "results/01_manifest/kelly_auranofin_manifest.csv"

    # Output
    work_dir = root / "results/03_crops"
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting crops from {seg_dir}")
    print(f"Data dir: {root / 'data/kelly_auranofin'}")

    # Загружаем манифест изображений
    df_images = pd.read_csv(manifest_path)
    image_dict = {row['filename']: row['abs_path'] for _, row in df_images.iterrows()}

    all_crops_info = []

    # Все маски
    mask_files = sorted(seg_dir.glob("*.npz"))
    print(f"Found {len(mask_files)} mask files")

    for mask_path in tqdm(mask_files, desc="Processing masks"):
        # Находим соответствующее изображение
        img_name = mask_path.stem.replace("_masks", "")
        for pattern in [".jpg", ".jpeg", ".png", ".tif"]:
            test_name = img_name + pattern
            if test_name in image_dict:
                img_path = Path(image_dict[test_name])
                break
        else:
            print(f"  → No image for {mask_path.name}")
            continue

        crops_info = extract_cell_crops(mask_path, img_path, work_dir)
        all_crops_info.extend(crops_info)

    # Сохраняем манифест клеток
    df_crops = pd.DataFrame(all_crops_info)
    manifest_path = work_dir / "manifest_cells.csv"
    df_crops.to_csv(manifest_path, index=False)

    print(f"\n🎉 CROPS EXTRACTED!")
    print(f"Total crops: {len(df_crops):,}")
    print(f"Manifest: {manifest_path}")
    print(f"Crops: {work_dir / 'crops'}")

    # Статистика по условиям
    summary = df_crops.groupby(['conditions']).size().reset_index(name='n_cells')
    print("\nCells per condition:")
    print(summary)


if __name__ == "__main__":
    main()
