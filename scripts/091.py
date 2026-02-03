import sys
import os
import re
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from cellpose import models as cp_models, io
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
from scipy import ndimage
import warnings

warnings.filterwarnings('ignore')


# Настройка CUDA
def safe_torch_cuda():
    """Безопасная инициализация CUDA"""
    try:
        import torch
        if torch.cuda.is_available():
            # Простой тест работы CUDA
            test_tensor = torch.tensor([1.0]).cuda()
            test_tensor = test_tensor * 2
            print(f"✅ CUDA работает: {torch.cuda.get_device_name(0)}")
            return True
        return False
    except Exception as e:
        print(f"⚠️ CUDA недоступна: {e}")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Отключаем CUDA
        return False


CUDA_AVAILABLE = safe_torch_cuda()


# ================= CONFIG =================
class Config:
    # Используем абсолютный путь относительно этого файла
    PROJECT_ROOT = Path(__file__).parent.parent
    RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
    RESULTS_DIR = PROJECT_ROOT / 'results' / 'experiment_analysis'

    # Новые директории для сохранения данных
    MASKS_DIR = RESULTS_DIR / 'masks'
    CROPS_DIR = RESULTS_DIR / 'crops'
    VISUALIZATIONS_DIR = RESULTS_DIR / 'visualizations'
    DATA_DIR = RESULTS_DIR / 'processed_data'

    # Паттерн для парсинга имен файлов
    FILENAME_PATTERN = re.compile(
        r"(?:HCT116[_-]?)?(?P<genotype>WT|CDK8KO|CDK8)[_-](?P<time>\d+)h[_-](?P<dose>\d+)Gy",
        re.IGNORECASE
    )

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MIN_CELL_SIZE = 100
    MAX_CELL_SIZE = 5000

    # Настройки для сохранения данных
    SAVE_MASKS = True
    SAVE_CROPS = True
    SAVE_VISUALIZATIONS = True
    MAX_CROPS_PER_IMAGE = 50  # Максимальное количество кропов на изображение для сохранения

    # Цвета для фаз клеточного цикла
    PHASE_COLORS = {
        'G1': (0, 255, 0),  # Зеленый
        'S': (255, 255, 0),  # Желтый
        'G2M': (255, 165, 0),  # Оранжевый
        'Mitosis': (255, 0, 0),  # Красный
        'SubG1': (128, 128, 128)  # Серый
    }

    # Размеры для сохранения кропов
    CROP_SIZE = (128, 128)  # Стандартный размер для кропов

    def __init__(self):
        # Создаем все необходимые директории
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.MASKS_DIR.mkdir(parents=True, exist_ok=True)
        self.CROPS_DIR.mkdir(parents=True, exist_ok=True)
        self.VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

        print(f"Ищу файлы в: {self.RAW_DATA_DIR.absolute()}")
        print(f"Результаты будут сохранены в: {self.RESULTS_DIR.absolute()}")


# Создаем глобальную конфигурацию
config = Config()


# ================= 1. METADATA PARSING =================
def parse_metadata(filename):
    """Извлекает факторы эксперимента из имени файла"""
    match = config.FILENAME_PATTERN.search(filename)
    if match:
        data = match.groupdict()
        data['dose'] = int(data['dose'])
        data['time'] = data['time']
        # Нормализуем имена генотипов
        if data['genotype'].upper() in ['CDK8KO', 'CDK8']:
            data['genotype'] = 'CDK8KO'
        else:
            data['genotype'] = 'WT'
        return data
    else:
        # Fallback: пытаемся извлечь хотя бы дозу
        print(f"WARNING: Не удалось распарсить имя файла: {filename}")
        return {'genotype': 'Unknown', 'time': 'Unknown', 'dose': 0}


# ================= 2. DEEP FEATURE EXTRACTOR =================
class DeepFeatureExtractor:
    def __init__(self):
        print(f"Загрузка ResNet50 на {config.DEVICE}...")
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.feature_extractor.to(config.DEVICE).eval()

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def get_features(self, crops):
        if not crops:
            return np.array([])

        batch_tensors = []
        for crop in crops:
            try:
                pil_img = Image.fromarray(crop)
                batch_tensors.append(self.transform(pil_img))
            except Exception as e:
                print(f"Ошибка при обработке кропа: {e}")
                continue

        if not batch_tensors:
            return np.array([])

        batch = torch.stack(batch_tensors).to(config.DEVICE)

        with torch.no_grad():
            features = self.feature_extractor(batch).squeeze().cpu().numpy()

        # Если одна клетка, нужно добавить размерность
        if len(crops) == 1:
            features = features.reshape(1, -1)

        return features


# ================= 3. IMAGE PROCESSING UTILITIES =================
def save_mask(mask, filename, original_image_name):
    """Сохраняет маску сегментации"""
    if not config.SAVE_MASKS:
        return

    mask_path = config.MASKS_DIR / f"{original_image_name}_mask_{filename}.png"
    # Нормализуем маску для визуализации
    mask_visual = (mask / (mask.max() if mask.max() > 0 else 1) * 255).astype(np.uint8)
    cv2.imwrite(str(mask_path), mask_visual)
    return str(mask_path)


def save_crop(crop, original_image_name, cell_id, phase=None):
    """Сохраняет кроп клетки"""
    if not config.SAVE_CROPS:
        return None

    # Создаем имя файла
    if phase:
        phase_dir = config.CROPS_DIR / phase
        phase_dir.mkdir(exist_ok=True)
        filename = f"{original_image_name}_cell{cell_id:04d}_{phase}.png"
        save_path = phase_dir / filename
    else:
        filename = f"{original_image_name}_cell{cell_id:04d}.png"
        save_path = config.CROPS_DIR / filename

    # Сохраняем изображение
    cv2.imwrite(str(save_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    return str(save_path)


def extract_cell_features(img, mask, cell_id):
    """Извлекает расширенные признаки клетки"""
    mask_bool = (mask == cell_id)

    # Базовые метрики
    area = np.sum(mask_bool)
    intensity_values = img[mask_bool]
    mean_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else 0
    std_intensity = np.std(intensity_values) if len(intensity_values) > 0 else 0

    # Геометрические признаки
    y_coords, x_coords = np.where(mask_bool)
    if len(y_coords) == 0 or len(x_coords) == 0:
        return None

    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()
    bbox_height = y_max - y_min + 1
    bbox_width = x_max - x_min + 1
    bbox_aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0

    # Центроид
    centroid_y = np.mean(y_coords)
    centroid_x = np.mean(x_coords)

    # Форма
    contours, _ = cv2.findContours(
        mask_bool.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        perimeter = cv2.arcLength(contours[0], True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        # Моменты формы
        moments = cv2.moments(contours[0])
        if moments['m00'] != 0:
            hu_moments = cv2.HuMoments(moments).flatten()
        else:
            hu_moments = np.zeros(7)
    else:
        perimeter = 0
        circularity = 0
        hu_moments = np.zeros(7)

    # Текстура (Haralick features упрощенные)
    if len(img.shape) == 2:
        texture_features = extract_texture_features(img, mask_bool)
    else:
        # Для цветных изображений берем интенсивность
        if len(img.shape) == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray_img = img
        texture_features = extract_texture_features(gray_img, mask_bool)

    features = {
        'cell_id': cell_id,
        'area': area,
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'total_intensity': np.sum(intensity_values),
        'bbox_height': bbox_height,
        'bbox_width': bbox_width,
        'bbox_aspect_ratio': bbox_aspect_ratio,
        'centroid_y': centroid_y,
        'centroid_x': centroid_x,
        'perimeter': perimeter,
        'circularity': circularity,
        'compactness': (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0,
        'solidity': area / cv2.contourArea(contours[0]) if contours and cv2.contourArea(contours[0]) > 0 else 0,
        'extent': area / (bbox_height * bbox_width) if bbox_height * bbox_width > 0 else 0,
        'eccentricity': compute_eccentricity(mask_bool),
        'orientation': compute_orientation(y_coords, x_coords),
        'texture_contrast': texture_features.get('contrast', 0),
        'texture_energy': texture_features.get('energy', 0),
        'texture_homogeneity': texture_features.get('homogeneity', 0),
        'texture_correlation': texture_features.get('correlation', 0),
    }

    # Добавляем моменты Ху
    for i in range(7):
        features[f'hu_moment_{i + 1}'] = hu_moments[i] if i < len(hu_moments) else 0

    return features


def extract_texture_features(img, mask):
    """Извлекает простые текстурные признаки"""
    try:
        # Создаем маску для области клетки
        masked_img = img.copy()
        masked_img[~mask] = 0

        # Вычисляем гистограмму градиентов (упрощенно)
        if np.sum(mask) > 0:
            sobelx = cv2.Sobel(masked_img, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(masked_img, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

            # Базовые статистики текстуры
            texture_values = gradient_magnitude[mask]
            if len(texture_values) > 0:
                contrast = np.std(texture_values)
                energy = np.sum(texture_values ** 2)
                homogeneity = 1.0 / (1.0 + np.var(texture_values)) if np.var(texture_values) > 0 else 0
            else:
                contrast = energy = homogeneity = 0
        else:
            contrast = energy = homogeneity = 0

        # Простая корреляция интенсивности
        if np.sum(mask) > 1:
            intensities = img[mask]
            if len(intensities) > 1:
                correlation = np.corrcoef(intensities[:-1], intensities[1:])[0, 1] if not np.isnan(
                    np.corrcoef(intensities[:-1], intensities[1:])[0, 1]) else 0
            else:
                correlation = 0
        else:
            correlation = 0

        return {
            'contrast': contrast,
            'energy': energy,
            'homogeneity': homogeneity,
            'correlation': correlation
        }
    except:
        return {'contrast': 0, 'energy': 0, 'homogeneity': 0, 'correlation': 0}


def compute_eccentricity(mask):
    """Вычисляет эксцентриситет клетки"""
    y_coords, x_coords = np.where(mask)
    if len(y_coords) < 2:
        return 0

    try:
        # Вычисляем собственные значения ковариационной матрицы
        cov_matrix = np.cov(y_coords, x_coords)
        eigvals = np.linalg.eigvals(cov_matrix)

        if np.max(eigvals) > 0:
            eccentricity = np.sqrt(1 - np.min(eigvals) / np.max(eigvals))
        else:
            eccentricity = 0
    except:
        eccentricity = 0

    return eccentricity


def compute_orientation(y_coords, x_coords):
    """Вычисляет ориентацию клетки"""
    if len(y_coords) < 2:
        return 0

    try:
        # Центрируем координаты
        y_centered = y_coords - np.mean(y_coords)
        x_centered = x_coords - np.mean(x_coords)

        # Вычисляем ковариационную матрицу
        cov_matrix = np.cov(y_centered, x_centered)

        # Собственные векторы и значения
        eigvals, eigvecs = np.linalg.eig(cov_matrix)

        # Ориентация - угол главной оси
        if eigvals[0] > eigvals[1]:
            orientation = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        else:
            orientation = np.arctan2(eigvecs[1, 1], eigvecs[0, 1])
    except:
        orientation = 0

    return orientation


def create_annotated_image(original_img, masks, cell_data, output_path):
    """Создает аннотированное изображение с bounding boxes и подписями фаз"""
    try:
        if len(original_img.shape) == 2:
            annotated_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
        else:
            annotated_img = original_img.copy()

        # Рисуем контуры всех клеток
        contours, _ = cv2.findContours(
            (masks > 0).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated_img, contours, -1, (255, 255, 255), 1)

        # Добавляем bounding boxes и подписи для каждой клетки
        for _, cell in cell_data.iterrows():
            try:
                cell_id = int(cell['cell_id'])
                mask_bool = (masks == cell_id)

                if np.sum(mask_bool) == 0:
                    continue

                # Bounding box
                y_coords, x_coords = np.where(mask_bool)
                y_min, y_max = y_coords.min(), y_coords.max()
                x_min, x_max = x_coords.min(), x_coords.max()

                # Цвет в зависимости от фазы
                phase = cell['phase']
                color = config.PHASE_COLORS.get(phase, (255, 255, 255))

                # Рисуем bounding box
                cv2.rectangle(annotated_img, (x_min, y_min), (x_max, y_max), color, 2)

                # Добавляем подпись с фазой
                label = f"{phase}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1

                # Размер текста
                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # Фон для текста
                cv2.rectangle(annotated_img,
                              (x_min, y_min - text_height - 5),
                              (x_min + text_width, y_min - 2),
                              color, -1)

                # Текст
                cv2.putText(annotated_img, label,
                            (x_min, y_min - 5),
                            font, font_scale, (0, 0, 0), thickness)
            except:
                continue

        # Сохраняем изображение
        cv2.imwrite(str(output_path), annotated_img)
        return annotated_img
    except Exception as e:
        print(f"Ошибка при создании аннотированного изображения: {e}")
        return None


# ================= 4. PROCESSING PIPELINE =================
def process_images():
    print("\n=== Инициализация моделей ===")
    segmentor = cp_models.CellposeModel(gpu=torch.cuda.is_available())
    extractor = DeepFeatureExtractor()

    all_data = []
    deep_features_list = []

    # Словарь для хранения масок и кропов
    image_data_dict = {}

    # Проверяем наличие директории
    if not config.RAW_DATA_DIR.exists():
        print(f"ОШИБКА: Директория не существует: {config.RAW_DATA_DIR.absolute()}")
        print("Создаю директорию...")
        config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Пожалуйста, поместите изображения в: {config.RAW_DATA_DIR.absolute()}")
        return pd.DataFrame(), np.array([]), image_data_dict

    # Ищем файлы
    files = sorted(list(config.RAW_DATA_DIR.glob('*.jpg')) +
                   list(config.RAW_DATA_DIR.glob('*.png')) +
                   list(config.RAW_DATA_DIR.glob('*.tif')) +
                   list(config.RAW_DATA_DIR.glob('*.jpeg')))

    if not files:
        print(f"\nОШИБКА: Не найдено изображений в {config.RAW_DATA_DIR.absolute()}")
        print("Поддерживаемые форматы: .jpg, .png, .tif, .jpeg")
        print("\nСписок файлов в директории:")
        for item in config.RAW_DATA_DIR.iterdir():
            print(f"  - {item.name}")
        return pd.DataFrame(), np.array([]), image_data_dict

    print(f"\n=== Найдено {len(files)} файлов ===")
    for f in files[:5]:  # Показываем первые 5
        print(f"  - {f.name}")
    if len(files) > 5:
        print(f"  ... и еще {len(files) - 5}")

    print("\n=== Обработка изображений ===")
    for fpath in tqdm(files, desc="Сегментация и извлечение признаков"):
        try:
            meta = parse_metadata(fpath.name)
            img = io.imread(str(fpath))

            # Сохраняем оригинальное изображение
            original_image_name = fpath.stem

            print(f"Обработка: {fpath.name}...")

            # 1. Сегментация
            masks, _, _ = segmentor.eval(img, diameter=None, channels=[0, 0])
            n_cells = masks.max()

            if n_cells == 0:
                print(f"\nWARNING: Не найдено клеток в {fpath.name}")
                continue

            # 2. Сохраняем маску
            if config.SAVE_MASKS:
                save_mask(masks, "full", original_image_name)

            # 3. Подготовка RGB версии для кропов
            if len(img.shape) == 2:
                image_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = img

            # 4. Извлечение кропов и метрик
            crops = []
            unique_cells = np.unique(masks)[1:]  # исключаем 0 (фон)

            # Собираем данные для этого изображения
            image_cell_data = []

            for cell_id in unique_cells:
                mask_bool = (masks == cell_id)
                area = np.sum(mask_bool)

                if area < config.MIN_CELL_SIZE or area > config.MAX_CELL_SIZE:
                    continue

                # Координаты bounding box
                y_coords, x_coords = np.where(mask_bool)
                y1, y2 = y_coords.min(), y_coords.max()
                x1, x2 = x_coords.min(), x_coords.max()

                # Добавляем padding
                pad = 10
                y1 = max(0, y1 - pad)
                x1 = max(0, x1 - pad)
                y2 = min(img.shape[0], y2 + pad)
                x2 = min(img.shape[1], x2 + pad)

                # Вырезаем кроп
                crop = image_rgb[y1:y2 + 1, x1:x2 + 1].copy()

                # Ресайз кропа до стандартного размера
                try:
                    crop_resized = cv2.resize(crop, config.CROP_SIZE, interpolation=cv2.INTER_AREA)
                except:
                    crop_resized = crop

                # Зануляем фон в кропе
                local_mask = mask_bool[y1:y2 + 1, x1:x2 + 1]
                if len(local_mask.shape) == 2 and crop_resized.shape[:2] == local_mask.shape:
                    local_mask_3ch = np.repeat(local_mask[:, :, np.newaxis], 3, axis=2)
                    crop_resized[~local_mask_3ch] = 0

                crops.append(crop_resized)

                # Извлекаем расширенные признаки
                cell_features = extract_cell_features(img, masks, cell_id)
                if cell_features is None:
                    continue

                # Базовые метрики
                intensity = np.sum(img[mask_bool])
                contours, _ = cv2.findContours(
                    mask_bool.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                perimeter = cv2.arcLength(contours[0], True) if contours else 1
                circ = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

                # Сохраняем данные клетки
                cell_data = {
                    'filename': fpath.name,
                    'original_image_name': original_image_name,
                    'genotype': meta['genotype'],
                    'dose': meta['dose'],
                    'time': meta['time'],
                    'cell_id': cell_id,
                    'area': area,
                    'total_intensity': intensity,
                    'circularity': circ,
                    'bbox_x1': x1,
                    'bbox_y1': y1,
                    'bbox_x2': x2,
                    'bbox_y2': y2,
                    'bbox_width': x2 - x1,
                    'bbox_height': y2 - y1,
                }

                # Добавляем расширенные признаки
                cell_data.update(cell_features)

                all_data.append(cell_data)
                image_cell_data.append(cell_data.copy())

                # Сохраняем кроп (пока без фазы)
                if config.SAVE_CROPS and len(crops) <= config.MAX_CROPS_PER_IMAGE:
                    crop_path = save_crop(crop_resized, original_image_name, cell_id)
                    cell_data['crop_path'] = crop_path

            # 5. Deep Features для этого изображения
            if crops:
                feats = extractor.get_features(crops)
                if feats.size > 0:
                    # Распределяем deep features по клеткам
                    if len(feats) == len(image_cell_data):
                        for i, cell_data in enumerate(image_cell_data):
                            start_idx = len(all_data) - len(image_cell_data) + i
                            # Сохраняем deep features
                            for j in range(feats.shape[1]):
                                all_data[start_idx][f'deep_feat_{j}'] = feats[i, j]
                    deep_features_list.append(feats)

            # Сохраняем данные для этого изображения
            image_data_dict[original_image_name] = {
                'image_path': str(fpath),
                'mask': masks,
                'cell_data': image_cell_data,
                'metadata': meta
            }

            print(f"  Обработано клеток: {len(image_cell_data)}")

        except Exception as e:
            print(f"\nОШИБКА при обработке {fpath.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Проверка на пустоту
    if not all_data:
        print("\nОШИБКА: Не удалось извлечь ни одной клетки из изображений!")
        return pd.DataFrame(), np.array([]), image_data_dict

    df = pd.DataFrame(all_data)

    if not deep_features_list:
        print("\nWARNING: Deep features не извлечены. Использую только базовые метрики.")
        deep_features = np.zeros((len(df), 1))  # заглушка
    else:
        deep_features = np.vstack(deep_features_list)

    print(f"\n=== Извлечено {len(df)} клеток ===")
    print(f"Размерность deep features: {deep_features.shape}")
    print(f"Количество колонок в данных: {len(df.columns)}")

    return df, deep_features, image_data_dict


# ================= 5. CALIBRATED CLASSIFICATION =================
def calibrate_and_classify(df, deep_features):
    if df.empty:
        print("Нет данных для классификации!")
        return df

    print("\n=== Калибровка и классификация ===")

    # Используем больше признаков для классификации
    feature_columns = ['area', 'total_intensity', 'circularity', 'mean_intensity', 'std_intensity',
                       'bbox_aspect_ratio', 'compactness', 'solidity', 'extent', 'eccentricity']

    # Оставляем только существующие колонки
    available_features = [col for col in feature_columns if col in df.columns]

    if len(available_features) < 2:
        print("WARNING: Недостаточно признаков для классификации. Использую базовые.")
        available_features = ['area', 'total_intensity']

    X = np.log1p(df[available_features].values)

    # Выбираем калибровочную выборку (0 Gy)
    calibration_mask = (df['dose'] == 0)
    n_control = calibration_mask.sum()

    print(f"Контрольных клеток (0 Gy): {n_control}")

    if n_control < 50:
        print("WARNING: Мало контрольных клеток! Использую все данные для обучения.")
        X_train = X
    else:
        X_train = X[calibration_mask]

    # Обучаем GMM с большим числом компонент
    print("Обучение Gaussian Mixture Model...")
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42, n_init=10)
    gmm.fit(X_train)

    # Сортируем кластеры по размеру (area)
    # Находим индекс area в признаках
    if 'area' in available_features:
        area_idx = available_features.index('area')
        means = gmm.means_[:, area_idx]
    else:
        # Если area нет, используем первый признак
        means = gmm.means_[:, 0]

    sorted_indices = np.argsort(means)

    # Сопоставляем кластеры фазам (больше кластеров для лучшего разделения)
    mapping = {}
    if len(sorted_indices) >= 5:
        mapping = {
            sorted_indices[0]: 'SubG1',
            sorted_indices[1]: 'G1',
            sorted_indices[2]: 'S',
            sorted_indices[3]: 'G2M',
            sorted_indices[4]: 'Mitosis'
        }
    elif len(sorted_indices) >= 3:
        mapping = {
            sorted_indices[0]: 'G1',
            sorted_indices[1]: 'S',
            sorted_indices[2]: 'G2M'
        }
    else:
        mapping = {i: f'Cluster_{i}' for i in range(len(sorted_indices))}

    # Предсказание для всех клеток
    all_labels = gmm.predict(X)
    df['phase'] = [mapping.get(l, f'Cluster_{l}') for l in all_labels]
    df['gmm_confidence'] = gmm.predict_proba(X).max(axis=1)
    df['gmm_cluster'] = all_labels

    # Уточнение: Митоз
    g2m_mask = (df['phase'] == 'G2M')
    if g2m_mask.any():
        mitosis_mask = g2m_mask & (df['circularity'] > 0.85) & (df['area'] > df[g2m_mask]['area'].median())
        df.loc[mitosis_mask, 'phase'] = 'Mitosis'

    # Уточнение: SubG1 (апоптоз)
    if n_control > 0:
        g1_control = df[calibration_mask & (df['phase'] == 'G1')]['area']
        if len(g1_control) > 0:
            g1_threshold = g1_control.quantile(0.25) * 0.5
            df.loc[df['area'] < g1_threshold, 'phase'] = 'SubG1'

    print("\nРаспределение фаз:")
    print(df['phase'].value_counts())

    # Сохраняем параметры GMM для дальнейшего использования
    gmm_data = {
        'means': gmm.means_,
        'covariances': gmm.covariances_,
        'weights': gmm.weights_,
        'mapping': mapping,
        'feature_names': available_features
    }

    with open(config.DATA_DIR / 'gmm_model.pkl', 'wb') as f:
        pickle.dump(gmm_data, f)

    print(f"\nМодель GMM сохранена: {config.DATA_DIR / 'gmm_model.pkl'}")

    return df


# ================= 6. SAVE ALL DATA =================
def save_all_data(df, image_data_dict):
    """Сохраняет все данные для дальнейшего анализа"""
    print("\n=== Сохранение всех данных ===")

    # 1. Сохраняем полный DataFrame
    output_csv = config.DATA_DIR / 'full_cell_data.csv'
    df.to_csv(output_csv, index=False)
    print(f"Полные данные сохранены: {output_csv}")

    # 2. Сохраняем сводную статистику
    summary = df.groupby(['genotype', 'time', 'dose', 'phase']).agg({
        'cell_id': 'count',
        'area': ['mean', 'std'],
        'circularity': ['mean', 'std'],
        'mean_intensity': ['mean', 'std'],
        'gmm_confidence': 'mean'
    }).round(3)

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    summary.to_csv(config.DATA_DIR / 'summary_statistics.csv', index=False)
    print(f"Сводная статистика: {config.DATA_DIR / 'summary_statistics.csv'}")

    # 3. Сохраняем данные изображений (без самих изображений, только метаданные)
    image_metadata = {}
    for img_name, img_data in image_data_dict.items():
        image_metadata[img_name] = {
            'image_path': img_data['image_path'],
            'metadata': img_data['metadata'],
            'n_cells': len(img_data['cell_data']) if 'cell_data' in img_data else 0
        }

    with open(config.DATA_DIR / 'image_metadata.json', 'w') as f:
        json.dump(image_metadata, f, indent=2)

    # 4. Сохраняем кропы с фазами
    if config.SAVE_CROPS:
        print("\nСохранение кропов с фазами...")
        for _, row in df.iterrows():
            if 'crop_path' in row and pd.notnull(row['crop_path']):
                # Читаем сохраненный кроп
                try:
                    crop = cv2.imread(row['crop_path'])
                    if crop is not None:
                        # Сохраняем с фазой
                        save_crop(crop, row['original_image_name'],
                                  int(row['cell_id']), row['phase'])
                except:
                    pass

    # 5. Сохраняем аннотированные изображения
    if config.SAVE_VISUALIZATIONS:
        print("\nСоздание аннотированных изображений...")
        for img_name, img_data in image_data_dict.items():
            try:
                # Загружаем оригинальное изображение
                img = io.imread(img_data['image_path'])
                mask = img_data['mask']

                # Получаем данные клеток для этого изображения
                img_cells = df[df['original_image_name'] == img_name]

                if len(img_cells) > 0:
                    # Создаем аннотированное изображение
                    output_path = config.VISUALIZATIONS_DIR / f"{img_name}_annotated.png"
                    create_annotated_image(img, mask, img_cells, output_path)

                    # Также сохраняем изображение с маской
                    mask_visual = (mask / mask.max() * 255).astype(np.uint8)
                    if len(mask_visual.shape) == 2:
                        mask_visual = cv2.cvtColor(mask_visual, cv2.COLOR_GRAY2RGB)
                    cv2.imwrite(str(config.VISUALIZATIONS_DIR / f"{img_name}_mask_visual.png"), mask_visual)

            except Exception as e:
                print(f"Ошибка при создании аннотированного изображения для {img_name}: {e}")

    print(f"\nВсе данные сохранены в: {config.DATA_DIR}")

    return df


# ================= 7. REPORTS =================
def generate_reports(df):
    if df.empty:
        return

    print("\n=== Генерация отчетов ===")

    # 1. Сводная таблица
    summary = df.groupby(['genotype', 'time', 'dose', 'phase']).size().reset_index(name='count')
    total_per_condition = df.groupby(['genotype', 'time', 'dose']).size().reset_index(name='total')
    summary = summary.merge(total_per_condition, on=['genotype', 'time', 'dose'])
    summary['percentage'] = (summary['count'] / summary['total'] * 100).round(2)

    summary.to_csv(config.RESULTS_DIR / 'phase_distribution_summary.csv', index=False)
    print(f"Таблица сохранена: {config.RESULTS_DIR / 'phase_distribution_summary.csv'}")

    # 2. Dose-Response график
    plt.figure(figsize=(12, 6))

    # G2M + Mitosis
    df['is_G2M'] = df['phase'].isin(['G2M', 'Mitosis'])
    g2_data = df.groupby(['genotype', 'time', 'dose'])['is_G2M'].mean().reset_index()
    g2_data['is_G2M'] *= 100

    for genotype in df['genotype'].unique():
        for time in df['time'].unique():
            subset = g2_data[(g2_data['genotype'] == genotype) & (g2_data['time'] == time)]
            if not subset.empty:
                plt.plot(subset['dose'], subset['is_G2M'],
                         marker='o', linewidth=2,
                         label=f"{genotype} {time}h")

    plt.xlabel('Доза облучения (Gy)', fontsize=12)
    plt.ylabel('% клеток в G2/M', fontsize=12)
    plt.title('Доза-зависимый G2 блок', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / 'dose_response_G2M.png', dpi=300)
    plt.savefig(config.VISUALIZATIONS_DIR / 'dose_response_G2M.png', dpi=300)
    print(f"График сохранен: {config.RESULTS_DIR / 'dose_response_G2M.png'}")

    # 3. Stacked bar
    pivot = summary.pivot_table(
        index=['genotype', 'dose', 'time'],
        columns='phase',
        values='percentage',
        fill_value=0
    )

    phase_order = ['SubG1', 'G1', 'S', 'G2M', 'Mitosis']
    phase_order = [p for p in phase_order if p in pivot.columns]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, time_val in enumerate(df['time'].unique()):
        ax = axes[idx] if len(df['time'].unique()) > 1 else axes
        data = pivot.xs(time_val, level='time')[phase_order]
        data.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
        ax.set_title(f'Распределение фаз ({time_val}h)', fontsize=12)
        ax.set_ylabel('Процент клеток (%)')
        ax.set_xlabel('Генотип и доза')
        ax.legend(title='Фаза', bbox_to_anchor=(1.05, 1))
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / 'phase_distribution_bars.png', dpi=300)
    plt.savefig(config.VISUALIZATIONS_DIR / 'phase_distribution_bars.png', dpi=300)
    print(f"График сохранен: {config.RESULTS_DIR / 'phase_distribution_bars.png'}")

    # 4. Scatter plot признаков
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Area vs Circularity
    for phase in df['phase'].unique():
        phase_data = df[df['phase'] == phase]
        axes[0, 0].scatter(phase_data['area'], phase_data['circularity'],
                           alpha=0.5, label=phase, s=10)
    axes[0, 0].set_xlabel('Area')
    axes[0, 0].set_ylabel('Circularity')
    axes[0, 0].set_title('Area vs Circularity by Phase')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Intensity vs Area
    for genotype in df['genotype'].unique():
        genotype_data = df[df['genotype'] == genotype]
        axes[0, 1].scatter(genotype_data['area'], genotype_data['mean_intensity'],
                           alpha=0.5, label=genotype, s=10)
    axes[0, 1].set_xlabel('Area')
    axes[0, 1].set_ylabel('Mean Intensity')
    axes[0, 1].set_title('Area vs Intensity by Genotype')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Dose vs G2M percentage
    dose_g2m = df.groupby(['dose', 'genotype'])['is_G2M'].mean().reset_index()
    for genotype in dose_g2m['genotype'].unique():
        genotype_data = dose_g2m[dose_g2m['genotype'] == genotype]
        axes[1, 0].plot(genotype_data['dose'], genotype_data['is_G2M'] * 100,
                        marker='o', label=genotype, linewidth=2)
    axes[1, 0].set_xlabel('Dose (Gy)')
    axes[1, 0].set_ylabel('% G2M/Mitosis')
    axes[1, 0].set_title('Dose Response by Genotype')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Feature correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        # Берем не более 10 числовых колонок для лучшей читаемости
        numeric_cols_subset = numeric_cols[:min(10, len(numeric_cols))]
        corr_matrix = df[numeric_cols_subset].corr()
        im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 1].set_title('Feature Correlation Matrix')
        plt.colorbar(im, ax=axes[1, 1])

        # Добавляем текстовые значения
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                axes[1, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                ha='center', va='center', color='white', fontsize=8)

    plt.tight_layout()
    plt.savefig(config.VISUALIZATIONS_DIR / 'feature_analysis.png', dpi=300)
    print(f"График анализа признаков сохранен: {config.VISUALIZATIONS_DIR / 'feature_analysis.png'}")

def main():
print("=" * 60)
print("РАСШИРЕННЫЙ АНАЛИЗ КЛЕТОЧНОГО ЦИКЛА")
print("=" * 60)
print(f"Директория результатов: {config.RESULTS_DIR}")
print(f"Сохранение масок: {config.SAVE_MASKS}")
print(f"Сохранение кропов: {config.SAVE_CROPS}")
print(f"Сохранение визуализаций: {config.SAVE_VISUALIZATIONS}")
print("=" * 60)