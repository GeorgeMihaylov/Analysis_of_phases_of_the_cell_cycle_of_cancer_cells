"""
Скрипт для извлечения морфологических признаков из сегментированных клеток
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, feature
import cv2
import tqdm

# Добавляем корневую директорию в путь для импорта
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from cellpose import io
except ImportError:
    print("Ошибка: Cellpose не установлен. Установите его командой: pip install cellpose")
    sys.exit(1)


def extract_morphological_features(image, mask, cell_id):
    """
    Извлечение морфологических признаков для отдельной клетки

    Parameters:
    -----------
    image : ndarray
        Исходное изображение
    mask : ndarray
        Маска сегментации
    cell_id : int
        ID клетки в маске

    Returns:
    --------
    dict: Словарь с признаками клетки
    """

    # Создаем бинарную маску для текущей клетки
    cell_mask = (mask == cell_id)

    # Если клетка слишком маленькая, пропускаем
    if np.sum(cell_mask) < 10:
        return None

    # Извлекаем область изображения с клеткой
    cell_region = image.copy()
    cell_region[~cell_mask] = 0

    # Получаем координаты bounding box
    rows, cols = np.where(cell_mask)
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    # Вырезаем область клетки
    cell_patch = cell_region[min_row:max_row + 1, min_col:max_col + 1]
    mask_patch = cell_mask[min_row:max_row + 1, min_col:max_col + 1]

    # Базовые интенсивностные признаки
    cell_pixels = image[cell_mask]
    intensity_features = {
        'intensity_mean': np.mean(cell_pixels),
        'intensity_median': np.median(cell_pixels),
        'intensity_std': np.std(cell_pixels),
        'intensity_sum': np.sum(cell_pixels),
        'intensity_min': np.min(cell_pixels),
        'intensity_max': np.max(cell_pixels),
        'intensity_range': np.ptp(cell_pixels),
    }

    # Геометрические признаки через regionprops
    props = measure.regionprops(cell_mask.astype(np.uint8), intensity_image=image)[0]

    geometric_features = {
        'area': props.area,
        'perimeter': props.perimeter,
        'centroid_x': props.centroid[1],
        'centroid_y': props.centroid[0],
        'bbox_width': props.bbox[3] - props.bbox[1],
        'bbox_height': props.bbox[2] - props.bbox[0],
        'major_axis_length': props.major_axis_length,
        'minor_axis_length': props.minor_axis_length,
        'eccentricity': props.eccentricity,
        'solidity': props.solidity,
        'extent': props.extent,
        'orientation': props.orientation,
        'equivalent_diameter': props.equivalent_diameter,
    }

    # Рассчитываем производные геометрические признаки
    geometric_features['circularity'] = (4 * np.pi * props.area) / (props.perimeter ** 2) if props.perimeter > 0 else 0
    geometric_features[
        'aspect_ratio'] = props.major_axis_length / props.minor_axis_length if props.minor_axis_length > 0 else 0
    geometric_features['compactness'] = (props.perimeter ** 2) / (4 * np.pi * props.area) if props.area > 0 else 0

    # Моменты Ху (инвариантные к масштабу и вращению)
    moments = cv2.moments(cell_mask.astype(np.uint8))
    if moments['m00'] != 0:
        hu_moments = cv2.HuMoments(moments).flatten()
        # Логарифмическое преобразование для лучшего масштабирования
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
    else:
        hu_moments = np.zeros(7)

    hu_features = {f'hu_moment_{i + 1}': hu_moments[i] for i in range(7)}

    # Текстура (GLCM) - только если клетка достаточно большая
    if props.area > 50:
        try:
            # Нормализуем интенсивность для GLCM
            if np.max(cell_patch) > np.min(cell_patch):
                patch_normalized = ((cell_patch - np.min(cell_patch)) /
                                    (np.max(cell_patch) - np.min(cell_patch)) * 255).astype(np.uint8)
            else:
                patch_normalized = cell_patch.astype(np.uint8)

            glcm = feature.graycomatrix(
                patch_normalized,
                distances=[1, 3, 5],
                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=256,
                symmetric=True,
                normed=True
            )

            texture_features = {
                'contrast': np.mean(feature.graycoprops(glcm, 'contrast')),
                'dissimilarity': np.mean(feature.graycoprops(glcm, 'dissimilarity')),
                'homogeneity': np.mean(feature.graycoprops(glcm, 'homogeneity')),
                'energy': np.mean(feature.graycoprops(glcm, 'energy')),
                'correlation': np.mean(feature.graycoprops(glcm, 'correlation')),
                'ASM': np.mean(feature.graycoprops(glcm, 'ASM')),
            }
        except:
            texture_features = {
                'contrast': 0, 'dissimilarity': 0, 'homogeneity': 0,
                'energy': 0, 'correlation': 0, 'ASM': 0
            }
    else:
        texture_features = {
            'contrast': 0, 'dissimilarity': 0, 'homogeneity': 0,
            'energy': 0, 'correlation': 0, 'ASM': 0
        }

    # Объединяем все признаки
    all_features = {
        'cell_id': cell_id,
        **intensity_features,
        **geometric_features,
        **hu_features,
        **texture_features,
    }

    return all_features


def parse_filename(filename):
    """
    Парсинг информации из имени файла

    Пример: HCT116_WT_24h_0Gy_slide1_field01.jpg
    """
    try:
        # Удаляем расширение
        name = filename.replace('.jpg', '').replace('.png', '')
        parts = name.split('_')

        return {
            'genotype': parts[1] if len(parts) > 1 else 'unknown',
            'time_h': parts[2].replace('h', '') if len(parts) > 2 else 'unknown',
            'dose_gy': parts[3].replace('Gy', '') if len(parts) > 3 else 'unknown',
            'slide': parts[4].replace('slide', '') if len(parts) > 4 else 'unknown',
            'field': parts[5].replace('field', '') if len(parts) > 5 else 'unknown',
        }
    except:
        return {
            'genotype': 'unknown',
            'time_h': 'unknown',
            'dose_gy': 'unknown',
            'slide': 'unknown',
            'field': 'unknown',
        }


def process_single_image(image_path, output_dir, use_existing_masks=True):
    """
    Обработка одного изображения

    Parameters:
    -----------
    image_path : Path
        Путь к изображению
    output_dir : Path
        Директория для сохранения результатов
    use_existing_masks : bool
        Использовать существующие маски если есть
    """

    print(f"\nОбработка: {image_path.name}")

    # Парсинг информации из имени файла
    info = parse_filename(image_path.name)

    # Пути для сохранения масок
    masks_dir = output_dir / 'masks'
    masks_dir.mkdir(exist_ok=True)

    mask_path = masks_dir / f"{image_path.stem}_masks.npy"

    # Загрузка изображения
    image = io.imread(str(image_path))
    if len(image.shape) > 2:
        # Если цветное изображение, конвертируем в grayscale
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

    # Проверяем, есть ли сохраненные маски
    masks = None
    if use_existing_masks and mask_path.exists():
        try:
            masks = np.load(mask_path)
            print(f"  Загружены существующие маски из {mask_path.name}")
        except:
            print(f"  Не удалось загрузить маски, выполняется сегментация...")

    # Если маски не загружены, выполняем сегментацию
    if masks is None:
        try:
            from cellpose import models

            # Проверяем доступность GPU
            import torch
            gpu_available = torch.cuda.is_available()
            print(f"  CUDA доступен: {gpu_available}")

            # Загружаем модель Cellpose
            model = models.CellposeModel(gpu=gpu_available)

            # Выполняем сегментацию
            print(f"  Выполняется сегментация...")
            masks, _, _ = model.eval(
                image,
                diameter=None,
                channels=[0, 0],
                flow_threshold=0.4,
                cellprob_threshold=0.0
            )

            # Сохраняем маски
            np.save(mask_path, masks)
            print(f"  Маски сохранены в {mask_path.name}")

        except Exception as e:
            print(f"  Ошибка при сегментации: {e}")
            return pd.DataFrame()

    # Извлекаем признаки для каждой клетки
    features_list = []
    cell_ids = np.unique(masks)
    cell_ids = cell_ids[cell_ids > 0]  # исключаем фон

    print(f"  Найдено клеток: {len(cell_ids)}")

    for cell_id in tqdm.tqdm(cell_ids, desc="  Извлечение признаков"):
        features = extract_morphological_features(image, masks, cell_id)
        if features is not None:
            # Добавляем информацию об изображении
            features.update(info)
            features['image_filename'] = image_path.name
            features_list.append(features)

    # Создаем DataFrame
    if features_list:
        df = pd.DataFrame(features_list)
        print(f"  Извлечено признаков для {len(df)} клеток")
        return df
    else:
        print(f"  Не удалось извлечь признаки")
        return pd.DataFrame()


def main():
    """Основная функция"""

    print("=" * 60)
    print("Извлечение морфологических признаков клеток")
    print("=" * 60)

    # Пути
    data_dir = project_root / 'data' / 'raw'
    output_dir = project_root / 'results'

    # Проверяем существование директории с данными
    if not data_dir.exists():
        print(f"Ошибка: Директория с данными не найдена: {data_dir}")
        print("Пожалуйста, поместите изображения в data/raw/")
        return

    # Получаем список изображений
    image_files = list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.png'))

    if not image_files:
        print(f"Ошибка: В директории {data_dir} нет изображений")
        return

    print(f"Найдено изображений: {len(image_files)}")

    # Создаем директории для результатов
    features_dir = output_dir / 'features'
    features_dir.mkdir(parents=True, exist_ok=True)

    # Обрабатываем все изображения
    all_features = []

    for i, image_file in enumerate(image_files):
        print(f"\n[{i + 1}/{len(image_files)}] ", end="")
        df = process_single_image(image_file, output_dir, use_existing_masks=True)

        if not df.empty:
            # Сохраняем признаки для каждого изображения отдельно
            img_output_path = features_dir / f"{image_file.stem}_features.csv"
            df.to_csv(img_output_path, index=False)
            print(f"  Признаки сохранены в {img_output_path.name}")

            all_features.append(df)

    # Объединяем все данные
    if all_features:
        full_df = pd.concat(all_features, ignore_index=True)

        # Сохраняем полный датасет
        full_output_path = features_dir / 'all_cells_features.csv'
        full_df.to_csv(full_output_path, index=False)

        print("\n" + "=" * 60)
        print(f"ОБРАБОТКА ЗАВЕРШЕНА!")
        print(f"Всего обработано изображений: {len(image_files)}")
        print(f"Всего клеток: {len(full_df)}")
        print(f"Полный датасет сохранен: {full_output_path}")

        # Базовая статистика
        print("\nБазовая статистика:")
        print(f"  Среднее количество клеток на изображение: {len(full_df) / len(image_files):.1f}")
        print(f"  Столбцы в датасете: {len(full_df.columns)}")

        # Распределение по условиям эксперимента
        if 'genotype' in full_df.columns:
            print("\nРаспределение по генотипам:")
            print(full_df['genotype'].value_counts())

        if 'dose_gy' in full_df.columns:
            print("\nРаспределение по дозам облучения:")
            print(full_df['dose_gy'].value_counts().sort_index())

        if 'time_h' in full_df.columns:
            print("\nРаспределение по времени:")
            print(full_df['time_h'].value_counts().sort_index())

        # Сохраняем сводную статистику
        summary_path = features_dir / 'experiment_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Сводная статистика эксперимента\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Всего изображений: {len(image_files)}\n")
            f.write(f"Всего клеток: {len(full_df)}\n")
            f.write(f"Среднее клеток на изображение: {len(full_df) / len(image_files):.1f}\n\n")

            for col in ['genotype', 'time_h', 'dose_gy', 'slide']:
                if col in full_df.columns:
                    f.write(f"Распределение по {col}:\n")
                    for val, count in full_df[col].value_counts().items():
                        f.write(f"  {col}={val}: {count} клеток ({count / len(full_df) * 100:.1f}%)\n")
                    f.write("\n")

        print(f"\nСводная статистика сохранена: {summary_path}")

        # Создаем простую визуализацию
        create_visualization(full_df, features_dir)

    else:
        print("Не удалось извлечь признаки ни из одного изображения")


def create_visualization(df, output_dir):
    """Создание базовой визуализации признаков"""

    try:
        # Создаем директорию для графиков
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)

        # 1. Гистограмма площадей клеток
        plt.figure(figsize=(10, 6))
        plt.hist(df['area'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Площадь клетки (пиксели)')
        plt.ylabel('Количество клеток')
        plt.title('Распределение площадей клеток')
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / 'area_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Boxplot площади по генотипам
        if 'genotype' in df.columns and len(df['genotype'].unique()) > 1:
            plt.figure(figsize=(8, 6))
            genotypes = df['genotype'].unique()

            # Создаем подвыборки для каждого генотипа
            data = [df[df['genotype'] == g]['area'] for g in genotypes]

            plt.boxplot(data, labels=genotypes)
            plt.xlabel('Генотип')
            plt.ylabel('Площадь клетки (пиксели)')
            plt.title('Распределение площадей клеток по генотипам')
            plt.grid(True, alpha=0.3)
            plt.savefig(plots_dir / 'area_by_genotype.png', dpi=150, bbox_inches='tight')
            plt.close()

        # 3. Scatter plot: площадь vs интенсивность
        plt.figure(figsize=(10, 6))
        plt.scatter(df['area'], df['intensity_mean'], alpha=0.5, s=10)
        plt.xlabel('Площадь клетки')
        plt.ylabel('Средняя интенсивность')
        plt.title('Зависимость интенсивности от размера клетки')
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / 'area_vs_intensity.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Графики сохранены в: {plots_dir}")

    except Exception as e:
        print(f"Ошибка при создании визуализации: {e}")


if __name__ == "__main__":
    main()