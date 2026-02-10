# scripts/38_build_crops_from_masks_fixed.py
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import json
from tqdm import tqdm
import re


def parse_filename(filename: str) -> dict:
    """Парсинг имени файла для извлечения условий - ИСПРАВЛЕННАЯ ВЕРСИЯ"""
    name = Path(filename).stem

    # Ищем время
    time_match = re.search(r'(\d+)\s*h', name.lower())
    time_h = int(time_match.group(1)) if time_match else 2

    # По умолчанию
    concentration = 0.0
    treatment = "CTRL"

    # Определяем, есть ли "aura" в названии
    if "aura" in name.lower():
        treatment = "AURA"
        # Ищем концентрацию в формате "0.5uM", "1uM", "2uM"
        conc_match = re.search(r'(\d+\.?\d*)\s*u?m', name.lower())
        if conc_match:
            concentration = float(conc_match.group(1))
    else:
        # Если нет "aura", то это контроль
        treatment = "CTRL"
        concentration = 0.0

    # Генотип
    genotype = "KELLY"

    return {
        "genotype": genotype,
        "time": time_h,
        "concentration": concentration,
        "treatment": treatment,
        "source_image": filename
    }


def extract_cell_crops(mask_path: Path, image_path: Path, output_dir: Path,
                       min_size: int = 50, max_size: int = 300,
                       padding: int = 15, target_size: int = 128):
    """Извлечение crops клеток из масок"""

    # Загружаем маску
    try:
        data = np.load(mask_path)
        masks = data['masks']
    except:
        print(f"Не удалось загрузить маску: {mask_path}")
        return []

    # Загружаем изображение
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return []

    # Конвертируем в RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Получаем информацию об условиях
    conditions = parse_filename(image_path.name)

    # Находим уникальные метки клеток
    labels = np.unique(masks)
    labels = labels[labels > 0]  # Исключаем фон

    crops_info = []

    for label in tqdm(labels, desc=f"Клетки", leave=False):
        # Создаем бинарную маску для текущей клетки
        cell_mask = (masks == label).astype(np.uint8)

        # Находим bounding box
        rows, cols = np.where(cell_mask)
        if len(rows) == 0:
            continue

        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()

        # Проверяем размер
        h, w = y_max - y_min, x_max - x_min
        if h < min_size or w < min_size or h > max_size or w > max_size:
            continue

        # Добавляем padding
        y_min = max(0, y_min - padding)
        y_max = min(masks.shape[0], y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(masks.shape[1], x_max + padding)

        # Вырезаем crop из изображения
        img_crop = img[y_min:y_max, x_min:x_max]

        # Вырезаем соответствующую часть маски
        mask_crop = cell_mask[y_min:y_max, x_min:x_max]

        # Преобразуем изображение в grayscale
        if len(img_crop.shape) == 3:
            img_gray = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_crop

        # Ресайз до фиксированного размера
        try:
            img_resized = cv2.resize(img_gray, (target_size, target_size),
                                     interpolation=cv2.INTER_AREA)
            mask_resized = cv2.resize(mask_crop, (target_size, target_size),
                                      interpolation=cv2.INTER_NEAREST)
        except:
            continue

        # Нормализуем изображение
        if img_resized.max() > img_resized.min():
            img_resized = ((img_resized - img_resized.min()) /
                           (img_resized.max() - img_resized.min()) * 255).astype(np.uint8)

        # Сохраняем crop
        crop_id = f"{image_path.stem}_cell_{label:04d}"
        crop_path = output_dir / "crops" / f"{crop_id}.npz"
        crop_path.parent.mkdir(parents=True, exist_ok=True)

        # Сохраняем в npz
        np.savez_compressed(
            crop_path,
            img=img_resized,
            mask=mask_resized,
            label=label,
            bbox=[x_min, y_min, x_max, y_max],
            original_shape=img_gray.shape
        )

        # Добавляем информацию в manifest
        crop_info = {
            "cell_id": crop_id,
            "crop_path": str(crop_path.relative_to(output_dir.parent)),  # Изменено!
            "source_image": image_path.name,
            **conditions,
            "bbox_x": int(x_min),
            "bbox_y": int(y_min),
            "bbox_w": int(x_max - x_min),
            "bbox_h": int(y_max - y_min),
            "area_pixels": int(np.sum(cell_mask))
        }

        crops_info.append(crop_info)

    return crops_info


def process_all_images(work_dir: Path):
    """Обработка всех изображений и создание manifest"""

    root = Path(__file__).parent.parent
    data_dir = root / "data" / "kelly_auranofin"
    seg_dir = root / "results" / "segmentation_fixed" / "data"

    work_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = work_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Находим все маски
    mask_files = sorted(seg_dir.glob("*_masks.npz"))
    print(f"Найдено {len(mask_files)} файлов масок")

    all_crops_info = []

    for mask_path in tqdm(mask_files, desc="Обработка изображений"):
        # Находим соответствующее изображение
        img_name = mask_path.stem.replace("_masks", "")
        img_path = None

        # Ищем изображение
        for ext in [".jpg", ".jpeg", ".png"]:
            potential_path = data_dir / f"{img_name}{ext}"
            if potential_path.exists():
                img_path = potential_path
                break

        if img_path is None:
            print(f"Не найдено изображение для маски: {mask_path.stem}")
            continue

        # Извлекаем crops
        crops_info = extract_cell_crops(mask_path, img_path, crops_dir)
        all_crops_info.extend(crops_info)

    # Сохраняем manifest
    df = pd.DataFrame(all_crops_info)
    manifest_path = work_dir / "manifest_cells.csv"
    df.to_csv(manifest_path, index=False)

    # Статистика
    print(f"\nСтатистика:")
    print(f"Всего клеток: {len(df)}")
    print(f"Условия:")

    # Группируем по условиям
    grouped = df.groupby(["treatment", "concentration", "time"])
    for (treatment, conc, time_h), group in grouped:
        cond_name = "Ctrl" if treatment == "CTRL" else f"Aura {conc}uM"
        print(f"  {cond_name}, {time_h}h: {len(group)} клеток")

    # Сохраняем summary
    summary = df.groupby(["treatment", "concentration", "time"]).agg({
        "cell_id": "count",
        "area_pixels": ["mean", "std"]
    }).round(2)

    summary_path = work_dir / "crops_summary.csv"
    summary.to_csv(summary_path)

    print(f"\nManifest сохранен: {manifest_path}")
    print(f"Summary сохранен: {summary_path}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, required=True,
                        help="Рабочая директория для сохранения crops и manifest")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    df = process_all_images(work_dir)