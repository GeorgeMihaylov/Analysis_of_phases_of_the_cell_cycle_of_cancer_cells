# scripts/03extract_features.py
import numpy as np
import pandas as pd
import cv2
import logging
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import shannon_entropy, regionprops_table
from cellpose import io

# Настройка логгера (как в вашем основном скрипте)
logger = logging.getLogger("FeatureExtraction")


def extract_advanced_features(img_gray, mask, filename):
    """
    Извлекает расширенные признаки для классификации клеточного цикла.
    """
    props = regionprops_table(mask, intensity_image=img_gray,
                              properties=['label', 'area', 'perimeter',
                                          'eccentricity', 'solidity',
                                          'mean_intensity', 'max_intensity'])
    df = pd.DataFrame(props)

    # -- 1. Геометрические признаки --
    # Circularity: 1.0 для идеального круга. Апоптотические клетки (SubG1) часто неправильной формы.
    df['circularity'] = 4 * np.pi * df['area'] / (df['perimeter'] ** 2 + 1e-7)

    # Compactness: Похоже на circularity, но иногда используется иначе
    df['compactness'] = df['perimeter'] ** 2 / (df['area'] + 1e-7)

    # -- 2. Текстурные признаки (LBP & Haralick) --
    # G2/M клетки часто имеют другую текстуру хроматина по сравнению с G1
    lbp_entropies = []
    haralick_contrasts = []

    for region_id in df['label']:
        # Вырезаем bounding box для конкретной клетки
        region_mask = (mask == region_id)
        coords = np.where(region_mask)
        if len(coords[0]) == 0:
            lbp_entropies.append(0)
            haralick_contrasts.append(0)
            continue

        y_min, y_max = np.min(coords[0]), np.max(coords[0])
        x_min, x_max = np.min(coords[1]), np.max(coords[1])

        cell_crop = img_gray[y_min:y_max + 1, x_min:x_max + 1]
        mask_crop = region_mask[y_min:y_max + 1, x_min:x_max + 1]

        # Обнуляем фон
        cell_masked = cell_crop.copy()
        cell_masked[~mask_crop] = 0

        # LBP Entropy
        try:
            lbp = local_binary_pattern(cell_masked, P=8, R=1, method="uniform")
            # Считаем энтропию только внутри маски
            vals = lbp[mask_crop]
            lbp_entropies.append(shannon_entropy(vals))
        except:
            lbp_entropies.append(0)

        # Haralick Contrast (GLCM)
        try:
            glcm = graycomatrix(cell_masked, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            haralick_contrasts.append(contrast)
        except:
            haralick_contrasts.append(0)

    df['lbp_entropy'] = lbp_entropies
    df['haralick_contrast'] = haralick_contrasts
    df['filename'] = filename

    return df

# Пример использования в цикле обработки (интегрируйте в свой pipeline)
# for f in files:
#     img = io.imread(f)
#     masks = ... # ваш вывод Cellpose
#     df_feat = extract_advanced_features(img, masks, f.name)
