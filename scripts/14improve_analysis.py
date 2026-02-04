import sys
import os
import re
import json
import logging
import warnings
import numpy as np
import pandas as pd
import cv2
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from cellpose import models as cp_models, io
import matplotlib.pyplot as plt
import seaborn as sns

# Игнорируем предупреждения
warnings.filterwarnings('ignore')


# ================= CONFIGURATION =================
class Config:
    # Структура проекта
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'

    # Уникальная папка для каждого запуска
    RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
    RESULTS_DIR = PROJECT_ROOT / 'results' / f"analysis_{RUN_ID}"

    # Подпапки
    CHECKPOINTS_DIR = RESULTS_DIR / 'checkpoints'
    MASKS_DIR = RESULTS_DIR / 'masks'
    LOGS_DIR = RESULTS_DIR / 'logs'
    PLOTS_DIR = RESULTS_DIR / 'plots'
    INTERMEDIATE_DIR = RESULTS_DIR / 'intermediate'

    # Настройки Cellpose
    USE_GPU_CELLPOSE = torch.cuda.is_available()
    MIN_CELL_SIZE = 100
    MAX_CELL_SIZE = 5000
    SAVE_MASKS = True
    SAVE_SEGMENTATION_VIS = True

    # Параметры классификации
    INTENSITY_METRIC = 'mean_intensity'  # 'mean_intensity' или 'total_intensity'
    USE_ADVANCED_FEATURES = True  # Использовать расширенные морфологические признаки
    EXTRACT_ENTROPY = True  # Извлекать энтропию текстуры
    EXTRACT_ELLIPSE = True  # Извлекать параметры эллипса

    # Пороги для классификации
    SUBG1_DISTANCE_THRESHOLD = 0.4  # Минимальное расстояние для выделения SubG1 (log units)
    SUBG1_WEIGHT_THRESHOLD = 0.01  # Минимальный вес кластера для SubG1
    MITOSIS_CIRCULARITY_QUANTILE = 0.80  # Квантиль для выделения митоза
    MITOSIS_MIN_CIRCULARITY = 0.85  # Абсолютный минимум округлости для митоза

    # Нормализация
    NORMALIZATION_METHOD = 'per_sample'  # 'per_sample' или 'global'
    USE_KDE_FOR_PEAKS = True  # Использовать KDE для поиска пиков

    # Классификация
    MIN_CELLS_FOR_CLASSIFICATION = 50  # Минимальное количество клеток для классификации

    # Паттерн имени файла
    FILENAME_PATTERN = re.compile(
        r"(?:HCT116[_-]?)?(?P<genotype>WT|CDK8KO|CDK8|p53KO|p53KO_CDK8KO)[_-](?P<time>\d+)h[_-](?P<dose>\d+)Gy",
        re.IGNORECASE
    )

    # Ground Truth для валидации
    GROUND_TRUTH_DATA = [
        {"Genotype": "WT", "Dose": 0, "Phase": "SubG1", "Value": 6.9},
        {"Genotype": "WT", "Dose": 0, "Phase": "G1", "Value": 61.4},
        {"Genotype": "WT", "Dose": 0, "Phase": "G2M", "Value": 20.4},
        {"Genotype": "WT", "Dose": 4, "Phase": "SubG1", "Value": 22.9},
        {"Genotype": "WT", "Dose": 4, "Phase": "G1", "Value": 26.4},
        {"Genotype": "WT", "Dose": 4, "Phase": "G2M", "Value": 45.9},
        {"Genotype": "WT", "Dose": 10, "Phase": "SubG1", "Value": 36.3},
        {"Genotype": "WT", "Dose": 10, "Phase": "G1", "Value": 20.1},
        {"Genotype": "WT", "Dose": 10, "Phase": "G2M", "Value": 36.1},
        {"Genotype": "CDK8KO", "Dose": 0, "Phase": "SubG1", "Value": 9.0},
        {"Genotype": "CDK8KO", "Dose": 0, "Phase": "G1", "Value": 63.4},
        {"Genotype": "CDK8KO", "Dose": 0, "Phase": "G2M", "Value": 16.4},
        {"Genotype": "CDK8KO", "Dose": 4, "Phase": "SubG1", "Value": 35.4},
        {"Genotype": "CDK8KO", "Dose": 4, "Phase": "G1", "Value": 21.8},
        {"Genotype": "CDK8KO", "Dose": 4, "Phase": "G2M", "Value": 33.2},
        {"Genotype": "CDK8KO", "Dose": 10, "Phase": "SubG1", "Value": 48.1},
        {"Genotype": "CDK8KO", "Dose": 10, "Phase": "G1", "Value": 23.2},
        {"Genotype": "CDK8KO", "Dose": 10, "Phase": "G2M", "Value": 19.4},
    ]

    # Цвета для фаз
    PHASE_COLORS = {
        'SubG1': 'gray',
        'G1': 'blue',
        'S': 'purple',
        'G2M': 'green',
        'Mitosis': 'red'
    }

    def __init__(self):
        # Создание всех необходимых директорий
        for d in [self.RESULTS_DIR, self.CHECKPOINTS_DIR, self.MASKS_DIR,
                  self.LOGS_DIR, self.PLOTS_DIR, self.INTERMEDIATE_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        # Настройка логирования
        self.setup_logging()

        logger.info("=" * 70)
        logger.info(f"Запуск анализа: {self.RUN_ID}")
        logger.info(f"Проект: {self.PROJECT_ROOT}")
        logger.info(f"Данные: {self.RAW_DATA_DIR}")
        logger.info(f"Результаты: {self.RESULTS_DIR}")
        logger.info(f"GPU для Cellpose: {self.USE_GPU_CELLPOSE}")
        logger.info(f"Метод нормализации: {self.NORMALIZATION_METHOD}")
        logger.info("=" * 70)

    def setup_logging(self):
        """Настройка системы логирования"""
        log_file = self.LOGS_DIR / f'analysis_{self.RUN_ID}.log'

        # Создаем логгер
        self.logger = logging.getLogger('CellCycleAnalysis')
        self.logger.setLevel(logging.DEBUG)

        # Форматер (без юникодных символов для Windows)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Файловый обработчик (все сообщения)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Консольный обработчик (только INFO и выше)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Используем простой форматер для консоли (без проблемных символов)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)

        # Очищаем существующие обработчики и добавляем новые
        self.logger.handlers = []
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Глобальный логгер для использования в коде
        global logger
        logger = self.logger


config = Config()
logger = config.logger


# ================= 1. METADATA PARSING =================

class MetadataParser:
    """Парсер метаданных из имен файлов"""

    @staticmethod
    def parse(filename: str) -> Dict[str, Any]:
        """Извлечение метаданных из имени файла"""
        match = config.FILENAME_PATTERN.search(filename)
        if match:
            data = match.groupdict()
            data['dose'] = int(data['dose'])
            data['time'] = int(data['time'])
            geno = data['genotype'].upper()

            # Нормализация генотипов
            if 'CDK8' in geno and 'P53' not in geno and 'KO' in geno:
                data['genotype'] = 'CDK8KO'
            elif 'WT' in geno:
                data['genotype'] = 'WT'
            elif 'P53KO' in geno:
                data['genotype'] = 'p53KO'
            else:
                data['genotype'] = geno  # Сохраняем оригинальное название

            logger.debug(f"Парсинг {filename}: {data}")
            return data

        logger.warning(f"Не удалось распарсить имя файла: {filename}")
        return {'genotype': 'Unknown', 'time': 0, 'dose': 0}


# ================= 2. IMAGE PROCESSING WITH CHECKPOINTS =================

class EnhancedImageProcessor:
    """Улучшенный процессор изображений с системой чекпоинтов"""

    def __init__(self):
        logger.info(f"Инициализация Cellpose (GPU={config.USE_GPU_CELLPOSE})...")
        try:
            self.segmentor = cp_models.CellposeModel(
                gpu=config.USE_GPU_CELLPOSE,
                model_type='cyto'
            )
            logger.info("Cellpose модель успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки Cellpose: {e}")
            raise

        # Система чекпоинтов
        self.checkpoint_file = config.CHECKPOINTS_DIR / 'processed_files.json'
        self.processed_files = self._load_checkpoint()

    def _load_checkpoint(self) -> set:
        """Загрузка списка обработанных файлов"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return set(data)
            except Exception as e:
                logger.warning(f"Ошибка загрузки чекпоинта: {e}")
        return set()

    def _save_checkpoint(self, filename: str):
        """Сохранение чекпоинта"""
        self.processed_files.add(filename)
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(list(self.processed_files), f, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения чекпоинта: {e}")

    def _extract_cell_features(self, img: np.ndarray, mask: np.ndarray, cell_id: int) -> Optional[Dict]:
        """Извлечение признаков из отдельной клетки"""
        mask_bool = (mask == cell_id)

        # Базовые метрики
        area = np.sum(mask_bool)

        # Фильтрация по размеру
        if area < config.MIN_CELL_SIZE or area > config.MAX_CELL_SIZE:
            return None

        # Интенсивностные признаки
        intensity_values = img[mask_bool]
        total_intensity = np.sum(intensity_values)
        mean_intensity = np.mean(intensity_values)
        std_intensity = np.std(intensity_values)

        # Морфологические признаки
        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        contour = contours[0]
        perimeter = cv2.arcLength(contour, True)

        # Окружность
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        # Инициализация дополнительных признаков
        features = {
            'area': area,
            'total_intensity': total_intensity,
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'circularity': circularity,
            'perimeter': perimeter,
        }

        # Дополнительные морфологические признаки
        if config.EXTRACT_ELLIPSE and len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                (center, axes, orientation) = ellipse
                major_axis = max(axes)
                minor_axis = min(axes)
                eccentricity = np.sqrt(1 - (minor_axis ** 2) / (major_axis ** 2)) if major_axis > 0 else 0
                features['eccentricity'] = eccentricity
                features['ellipse_major'] = major_axis
                features['ellipse_minor'] = minor_axis
            except:
                features['eccentricity'] = 0

        # Текстура (энтропия)
        if config.EXTRACT_ENTROPY:
            hist, _ = np.histogram(intensity_values, bins=32, range=(0, 255))
            hist = hist / hist.sum() if hist.sum() > 0 else hist
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            features['entropy'] = entropy

        # Solidness (отношение площади к выпуклой оболочке)
        try:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            features['solidity'] = solidity
        except:
            features['solidity'] = 0

        return features

    def process_single_image(self, fpath: Path) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
        """Обработка одного изображения"""
        logger.info(f"Обработка: {fpath.name}")

        try:
            # Загрузка изображения
            img = io.imread(str(fpath))
            if len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Парсинг метаданных
            meta = MetadataParser.parse(fpath.name)

            # Сегментация
            logger.debug(f"Сегментация {fpath.name}")
            masks, flows, _ = self.segmentor.eval(
                img,
                diameter=None,
                channels=[0, 0],
                flow_threshold=0.4,
                cellprob_threshold=0.0
            )

            # Статистика сегментации
            n_cells = len(np.unique(masks)) - 1
            logger.info(f"  Сегментировано клеток: {n_cells}")

            # Извлечение признаков
            cells_data = []
            unique_cells = np.unique(masks)[1:]  # Исключаем фон (0)

            for cell_id in unique_cells:
                features = self._extract_cell_features(img, masks, cell_id)
                if features is not None:
                    cell_data = {
                        **meta,
                        'filename': fpath.name,
                        'cell_id': cell_id,
                        **features
                    }
                    cells_data.append(cell_data)

            df_image = pd.DataFrame(cells_data)

            # Сохранение промежуточных результатов
            if not df_image.empty:
                # CSV с данными клеток
                csv_file = config.INTERMEDIATE_DIR / f"{fpath.stem}_cells.csv"
                df_image.to_csv(csv_file, index=False)
                logger.debug(f"  Сохранены данные: {csv_file}")

                # Статистика по файлу
                stats = {
                    'filename': fpath.name,
                    'total_cells': n_cells,
                    'valid_cells': len(df_image),
                    'mean_area': float(df_image['area'].mean()),
                    'mean_intensity': float(df_image['mean_intensity'].mean())
                }
                stats_file = config.INTERMEDIATE_DIR / f"{fpath.stem}_stats.json"
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)

            # Сохранение масок
            if config.SAVE_MASKS:
                mask_file = config.MASKS_DIR / f"{fpath.stem}_mask.npy"
                np.save(str(mask_file), masks)
                logger.debug(f"  Сохранена маска: {mask_file}")

            # Визуализация сегментации
            if config.SAVE_SEGMENTATION_VIS and not df_image.empty:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                axes[0].imshow(img, cmap='gray')
                axes[0].set_title('Исходное изображение')
                axes[0].axis('off')

                axes[1].imshow(masks, cmap='tab20c')
                axes[1].set_title(f'Сегментация ({len(df_image)} клеток)')
                axes[1].axis('off')

                axes[2].imshow(flows[0], cmap='jet')
                axes[2].set_title('Поток')
                axes[2].axis('off')

                plt.tight_layout()
                vis_file = config.MASKS_DIR / f"{fpath.stem}_segmentation.png"
                plt.savefig(vis_file, dpi=150, bbox_inches='tight')
                plt.close()
                logger.debug(f"  Сохранена визуализация: {vis_file}")

            logger.info(f"  Успешно обработано клеток: {len(df_image)}")
            return df_image, masks

        except Exception as e:
            logger.error(f"Ошибка обработки {fpath.name}: {e}", exc_info=True)
            return None, None

    def process_directory(self) -> pd.DataFrame:
        """Обработка всех изображений в директории"""
        # Поиск файлов
        extensions = ['*.jpg', '*.png', '*.tif', '*.tiff', '*.jpeg']
        files = []
        for ext in extensions:
            files.extend(list(config.RAW_DATA_DIR.glob(ext)))

        files = sorted(files)

        if not files:
            logger.error(f"Файлы не найдены в {config.RAW_DATA_DIR}")
            return pd.DataFrame()

        logger.info(f"Найдено файлов: {len(files)}")
        logger.info(f"Уже обработано: {len(self.processed_files)}")

        all_data = []
        processed_count = 0
        failed_count = 0

        # Проверка существующих CSV файлов (чекпоинты)
        for fpath in files:
            csv_file = config.INTERMEDIATE_DIR / f"{fpath.stem}_cells.csv"
            if csv_file.exists() and fpath.name in self.processed_files:
                try:
                    df_existing = pd.read_csv(csv_file)
                    all_data.append(df_existing)
                    processed_count += 1
                    logger.info(f"Загружено из чекпоинта: {fpath.name} ({len(df_existing)} клеток)")
                except Exception as e:
                    logger.warning(f"Ошибка загрузки чекпоинта {csv_file}: {e}")

        # Обработка новых файлов
        files_to_process = [f for f in files if f.name not in self.processed_files]

        if files_to_process:
            logger.info(f"Новых файлов для обработки: {len(files_to_process)}")

            for fpath in tqdm(files_to_process, desc="Обработка изображений"):
                df_image, _ = self.process_single_image(fpath)

                if df_image is not None and not df_image.empty:
                    all_data.append(df_image)
                    self._save_checkpoint(fpath.name)
                    processed_count += 1
                else:
                    failed_count += 1
                    logger.warning(f"Не удалось обработать файл: {fpath.name}")

        # Объединение данных
        if all_data:
            df_all = pd.concat(all_data, ignore_index=True)
            logger.info(f"Обработка завершена. Успешно: {processed_count}, неудачно: {failed_count}")
            logger.info(f"Всего клеток: {len(df_all)}")

            # Сохранение сырых данных
            raw_data_file = config.INTERMEDIATE_DIR / 'all_cells_raw.csv'
            df_all.to_csv(raw_data_file, index=False)
            logger.info(f"Сохранены все сырые данные: {raw_data_file}")

            return df_all

        logger.warning("Нет данных после обработки")
        return pd.DataFrame()


# ================= 3. INTENSITY NORMALIZATION =================

class IntensityNormalizer:
    """Нормализация интенсивности между образцами"""

    @staticmethod
    def normalize_per_sample(df: pd.DataFrame) -> pd.DataFrame:
        """
        Per-sample нормализация:
        1. Логарифмирует интенсивность.
        2. Находит G1 пик для каждого файла с помощью KDE.
        3. Выравнивает все файлы по общему референсу.
        """
        logger.info("=== Нормализация интенсивности (per-sample) ===")

        if df.empty:
            return df

        df = df.copy()

        # 1. Логарифмирование интенсивности
        intensity_col = config.INTENSITY_METRIC
        df['log_intensity'] = np.log1p(df[intensity_col])

        # 2. Поиск пиков по файлам
        file_peaks = {}
        file_stats = {}

        for fname in tqdm(df['filename'].unique(), desc="Поиск пиков G1"):
            subset = df[df['filename'] == fname]['log_intensity'].values

            if len(subset) < 10:  # Слишком мало клеток для надежного определения
                file_peaks[fname] = np.median(subset)
                continue

            if config.USE_KDE_FOR_PEAKS:
                # Использование KDE для более точного определения пика
                try:
                    kde = gaussian_kde(subset)
                    x_range = np.linspace(subset.min(), subset.max(), 200)
                    y_kde = kde(x_range)
                    peak_idx = np.argmax(y_kde)
                    peak_val = x_range[peak_idx]
                    file_peaks[fname] = peak_val
                except Exception as e:
                    logger.warning(f"KDE не удалось для {fname}: {e}. Используем медиану.")
                    file_peaks[fname] = np.median(subset)
            else:
                # Использование гистограммы
                hist, bin_edges = np.histogram(subset, bins=50)
                peak_idx = np.argmax(hist)
                peak_val = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
                file_peaks[fname] = peak_val

            # Сохранение статистики
            file_stats[fname] = {
                'peak_value': float(file_peaks[fname]),
                'n_cells': int(len(subset)),
                'mean_intensity': float(np.mean(subset)),
                'std_intensity': float(np.std(subset))
            }

        # Сохранение статистики пиков
        stats_file = config.INTERMEDIATE_DIR / 'peak_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(file_stats, f, indent=2, ensure_ascii=False)
        logger.debug(f"Сохранена статистика пиков: {stats_file}")

        # 3. Выбор референса
        # Используем медиану пиков контрольных образцов (0 Gy)
        control_files = df[df['dose'] == 0]['filename'].unique()
        control_peaks = [file_peaks[f] for f in control_files if f in file_peaks]

        if control_peaks:
            reference_peak = np.median(control_peaks)
            logger.info(f"Референсный G1 пик (контроль 0 Gy): {reference_peak:.3f}")
        else:
            # Если нет контроля, используем медиану всех образцов
            reference_peak = np.median(list(file_peaks.values()))
            logger.info(f"Референсный G1 пик (все образцы): {reference_peak:.3f}")

        # 4. Выравнивание
        def apply_normalization(row):
            fname = row['filename']
            if fname in file_peaks:
                shift = reference_peak - file_peaks[fname]
                return row['log_intensity'] + shift
            return row['log_intensity']

        df['log_intensity_norm'] = df.apply(apply_normalization, axis=1)

        # 5. Преобразование обратно в линейную шкалу для визуализации
        df['intensity_linear_norm'] = np.expm1(df['log_intensity_norm'])

        # Статистика после нормализации
        logger.info(f"Нормализация завершена:")
        logger.info(f"  Диапазон до: {df['log_intensity'].min():.2f} - {df['log_intensity'].max():.2f}")
        logger.info(f"  Диапазон после: {df['log_intensity_norm'].min():.2f} - {df['log_intensity_norm'].max():.2f}")

        return df

    @staticmethod
    def normalize_global(df: pd.DataFrame) -> pd.DataFrame:
        """Глобальная нормализация (альтернативный метод)"""
        logger.info("=== Глобальная нормализация ===")

        df = df.copy()
        intensity_col = config.INTENSITY_METRIC

        # Логарифмирование
        df['log_intensity'] = np.log1p(df[intensity_col])

        # Находим G1 пик на всех данных контроля
        control_data = df[df['dose'] == 0]['log_intensity'].values

        if len(control_data) > 0:
            # Используем медиану контроля как референс
            reference_peak = np.median(control_data)

            # Сдвигаем все данные
            df['log_intensity_norm'] = df['log_intensity'] - reference_peak

            logger.info(f"Глобальный референс: {reference_peak:.3f}")
        else:
            # Если нет контроля, центрируем вокруг медианы
            reference_peak = df['log_intensity'].median()
            df['log_intensity_norm'] = df['log_intensity'] - reference_peak
            logger.warning("Нет контрольных образцов (0 Gy). Используется медиана всех данных.")

        # Линейная шкала для визуализации
        df['intensity_linear_norm'] = np.expm1(df['log_intensity_norm'])

        return df

    @staticmethod
    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Основной метод нормализации"""
        if config.NORMALIZATION_METHOD == 'per_sample':
            return IntensityNormalizer.normalize_per_sample(df)
        else:
            return IntensityNormalizer.normalize_global(df)


# ================= 4. IMPROVED ADAPTIVE CLASSIFICATION =================

class CellPhaseClassifier:
    """Улучшенный адаптивный классификатор фаз клеточного цикла"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.classification_results = {}

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Подготовка признаков для классификации"""
        # Всегда используем интенсивность
        features = ['log_intensity_norm']

        # Добавляем дополнительные признаки если они включены
        if config.USE_ADVANCED_FEATURES:
            additional_features = []
            for feature in ['circularity', 'area', 'eccentricity', 'entropy']:
                if feature in df.columns:
                    additional_features.append(feature)
            features.extend(additional_features)

        X = df[features].copy()

        # Заполнение пропусков медианой
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())

        # Нормализация
        if config.USE_ADVANCED_FEATURES and len(features) > 1:
            X_scaled = self.scaler.fit_transform(X)
            return X_scaled
        else:
            return X.values

    def _find_optimal_clusters(self, intensities: np.ndarray) -> int:
        """Определение оптимального числа кластеров"""
        # Биологически обоснованный подход:
        # 1. Всегда есть G1 и G2/M (минимум 2 кластера)
        # 2. Может быть S фаза между ними (3 кластера)
        # 3. Может быть SubG1 (4 кластера)

        # Анализируем распределение интенсивностей
        hist, bin_edges = np.histogram(intensities, bins=30)

        # Находим пики
        peaks, properties = find_peaks(hist, height=np.max(hist) * 0.1, distance=5)

        # Число пиков определяет число основных популяций
        n_peaks = len(peaks)

        if n_peaks >= 3:
            # Наблюдаем минимум 3 пика: SubG1, G1, G2 или G1, S, G2
            return min(4, n_peaks)
        elif n_peaks == 2:
            # 2 пика: G1 и G2, возможно SubG1 или S не выражены
            # Проверяем расстояние между пиками
            peak_positions = [(bin_edges[p] + bin_edges[p + 1]) / 2 for p in peaks]
            distance = abs(peak_positions[1] - peak_positions[0])

            if distance > np.log(1.8):  # Если расстояние > log(1.8) ~ 0.59
                # Большое расстояние, вероятно G1 и G2, S не выражена
                return 2
            else:
                # Близкие пики, возможно G1 и S или S и G2
                return 3
        else:
            # 1 пик или меньше
            return 2  # Минимум 2 кластера

    def _refine_classification_with_biology(self, df: pd.DataFrame, labels: np.ndarray,
                                            cluster_means: np.ndarray) -> pd.Series:
        """Уточнение классификации с учетом биологических знаний"""
        # Сортируем кластеры по интенсивности
        sorted_idx = np.argsort(cluster_means)

        # Базовое отображение: самый левый -> SubG1, следующий -> G1, и т.д.
        phase_map = {}

        n_clusters = len(cluster_means)

        if n_clusters == 2:
            # Только G1 и G2M
            phase_map = {
                sorted_idx[0]: 'G1',
                sorted_idx[1]: 'G2M'
            }
        elif n_clusters == 3:
            # G1, S, G2M
            phase_map = {
                sorted_idx[0]: 'G1',
                sorted_idx[1]: 'S',
                sorted_idx[2]: 'G2M'
            }
        elif n_clusters == 4:
            # SubG1, G1, S, G2M
            phase_map = {
                sorted_idx[0]: 'SubG1',
                sorted_idx[1]: 'G1',
                sorted_idx[2]: 'S',
                sorted_idx[3]: 'G2M'
            }

        # Применяем маппинг
        phases = pd.Series([phase_map.get(l, 'Unknown') for l in labels], index=df.index)

        # Пост-обработка: проверяем SubG1
        if 'SubG1' not in phases.values and n_clusters >= 3:
            # Если SubG1 не выделился, но есть клетки с очень низкой интенсивностью
            g1_mean = cluster_means[sorted_idx[0]]
            subg1_threshold = g1_mean - 0.5  # Эмпирический порог

            # Находим клетки ниже порога
            subg1_mask = df['log_intensity_norm'] < subg1_threshold
            if subg1_mask.sum() > len(df) * 0.01:  # Если >1% клеток
                phases.loc[subg1_mask] = 'SubG1'

        return phases

    def classify_adaptive_gmm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Адаптивная GMM классификация с определением числа компонент"""
        logger.info("=== Адаптивная классификация фаз (GMM) ===")

        df = df.copy()
        df['phase'] = 'Unknown'

        # Группировка по экспериментальным условиям
        groups = df.groupby(['genotype', 'dose', 'time'])

        results = []

        for (genotype, dose, time), group in tqdm(groups, desc="Классификация групп"):
            if len(group) < config.MIN_CELLS_FOR_CLASSIFICATION:
                logger.warning(f"Слишком мало клеток для {genotype}, {dose}Gy, {time}h: {len(group)}. Пропускаем.")
                group['phase'] = 'Unknown'
                results.append(group)
                continue

            logger.debug(f"Классификация: {genotype}, {dose}Gy, {time}h ({len(group)} клеток)")

            # Подготовка признаков
            X = self._prepare_features(group)

            # Определение оптимального числа кластеров
            intensities = group['log_intensity_norm'].values
            n_components = self._find_optimal_clusters(intensities)

            # Начальная инициализация центров
            if n_components == 4:
                # SubG1, G1, S, G2M
                percentiles = np.percentile(intensities, [10, 35, 65, 90])
            elif n_components == 3:
                # G1, S, G2M
                percentiles = np.percentile(intensities, [25, 50, 75])
            else:  # n_components == 2
                # G1, G2M
                percentiles = np.percentile(intensities, [30, 70])

            try:
                # Обучение GMM
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='full',
                    means_init=percentiles.reshape(-1, 1) if X.shape[1] == 1 else None,
                    random_state=42,
                    n_init=3,
                    max_iter=200
                )

                gmm.fit(X)
                labels = gmm.predict(X)

                # Получаем средние значения интенсивности для каждого кластера
                if X.shape[1] == 1:
                    cluster_means = gmm.means_.flatten()
                else:
                    # Если используется несколько признаков, берем только интенсивность
                    cluster_means = np.array([group.iloc[labels == i]['log_intensity_norm'].mean()
                                              for i in range(n_components)])

                # Уточнение классификации
                phases = self._refine_classification_with_biology(group, labels, cluster_means)
                group['phase'] = phases.values

                # Выделение митоза
                g2m_mask = group['phase'] == 'G2M'
                if g2m_mask.sum() > 10:
                    g2m_cells = group[g2m_mask]

                    # Определяем порог округлости для митоза
                    circ_thresh = g2m_cells['circularity'].quantile(config.MITOSIS_CIRCULARITY_QUANTILE)
                    final_thresh = max(circ_thresh, config.MITOSIS_MIN_CIRCULARITY)

                    mitosis_mask = g2m_mask & (group['circularity'] > final_thresh)
                    group.loc[mitosis_mask, 'phase'] = 'Mitosis'

                # Статистика
                phase_counts = group['phase'].value_counts().to_dict()
                phase_percentages = {k: v / len(group) * 100 for k, v in phase_counts.items()}

                # Сохранение результатов
                key = f"{genotype}_{dose}Gy_{time}h"
                self.classification_results[key] = {
                    'n_cells': int(len(group)),
                    'n_components': int(n_components),
                    'phase_counts': phase_counts,
                    'phase_percentages': phase_percentages
                }

                logger.info(f"  {genotype}, {dose}Gy, {time}h: использовано {n_components} кластеров")
                for phase, count in sorted(phase_counts.items()):
                    pct = count / len(group) * 100
                    logger.info(f"    {phase}: {count} клеток ({pct:.1f}%)")

            except Exception as e:
                logger.error(f"Ошибка классификации {genotype}, {dose}Gy, {time}h: {e}")
                group['phase'] = 'Unknown'

            results.append(group)

        # Объединение всех групп
        df_classified = pd.concat(results, ignore_index=True)

        # Сохранение результатов классификации
        results_file = config.INTERMEDIATE_DIR / 'classification_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.classification_results, f, indent=2, ensure_ascii=False)
        logger.debug(f"Сохранены результаты классификации: {results_file}")

        return df_classified

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Основной метод классификации"""
        return self.classify_adaptive_gmm(df)


# ================= 5. VISUALIZATION =================

class VisualizationEngine:
    """Движок визуализации с линейными шкалами"""

    @staticmethod
    def generate_all_plots(df: pd.DataFrame):
        """Генерация всех графиков"""
        logger.info("=== Генерация графиков ===")

        # Установка стиля
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10

        try:
            # 1. Псевдо-цитометрия (линейная шкала)
            VisualizationEngine.plot_pseudo_cytometry(df)

            # 2. Дозовые зависимости
            VisualizationEngine.plot_dose_response(df)

            # 3. Стековые диаграммы
            VisualizationEngine.plot_stacked_bars(df)

            # 4. Распределение признаков
            VisualizationEngine.plot_feature_distributions(df)

            # 5. Матрица корреляции
            VisualizationEngine.plot_correlation_matrix(df)

            logger.info("Все графики успешно сгенерированы")

        except Exception as e:
            logger.error(f"Ошибка при генерации графиков: {e}", exc_info=True)

    @staticmethod
    def plot_pseudo_cytometry(df: pd.DataFrame):
        """График псевдо-цитометрии с линейными осями"""
        logger.info("  Генерация графиков псевдо-цитометрии")

        # Уникальные генотипы и дозы
        genotypes = sorted(df['genotype'].unique())
        doses = sorted(df['dose'].unique())

        # Создаем сетку графиков
        n_rows = len(genotypes)
        n_cols = len(doses)

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4 * n_cols, 3.5 * n_rows),
            squeeze=False
        )

        for i, genotype in enumerate(genotypes):
            for j, dose in enumerate(doses):
                ax = axes[i, j]
                subset = df[(df['genotype'] == genotype) & (df['dose'] == dose)]

                if subset.empty:
                    ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
                    ax.set_title(f"{genotype}, {dose} Gy")
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                    continue

                # Гистограмма по фазам
                for phase, color in config.PHASE_COLORS.items():
                    phase_subset = subset[subset['phase'] == phase]
                    if not phase_subset.empty:
                        ax.hist(
                            phase_subset['intensity_linear_norm'],
                            bins=50,
                            alpha=0.6,
                            density=True,
                            color=color,
                            label=phase,
                            histtype='stepfilled'
                        )

                # Настройки осей
                ax.set_xlabel('Интенсивность (линейная)')
                ax.set_ylabel('Плотность')
                ax.set_title(f"{genotype}, {dose} Gy")
                ax.grid(True, alpha=0.3)

                # Убираем научную нотацию
                ax.ticklabel_format(style='plain', axis='x')

                # Ограничиваем легенду для экономии места
                if i == 0 and j == 0:
                    ax.legend(loc='upper right', fontsize=8, ncol=2)

        plt.tight_layout()
        output_file = config.PLOTS_DIR / 'pseudo_flow_cytometry.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"    Сохранен: {output_file}")

    @staticmethod
    def plot_dose_response(df: pd.DataFrame):
        """Графики дозовой зависимости для каждой фазы"""
        logger.info("  Генерация графиков дозовой зависимости")

        # Подготовка данных (игнорируем время)
        phase_percentages = df.groupby(['genotype', 'dose', 'phase']).size().unstack(fill_value=0)
        phase_percentages = phase_percentages.div(phase_percentages.sum(axis=1), axis=0) * 100

        # Основные фазы для анализа
        main_phases = ['SubG1', 'G1', 'S', 'G2M']

        # Создаем фигуру
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, phase in enumerate(main_phases):
            ax = axes[idx]

            for genotype in df['genotype'].unique():
                if genotype in phase_percentages.index.get_level_values(0):
                    # Агрегируем по времени (берем среднее)
                    genotype_data = phase_percentages.loc[genotype]
                    if isinstance(genotype_data, pd.Series):
                        # Если только одна доза
                        if phase in genotype_data.index:
                            ax.scatter([genotype_data.name], [genotype_data[phase]],
                                       label=genotype, s=100)
                    else:
                        # Несколько доз
                        if phase in genotype_data.columns:
                            ax.plot(
                                genotype_data.index,
                                genotype_data[phase],
                                'o-',
                                label=genotype,
                                linewidth=2,
                                markersize=8,
                                alpha=0.8
                            )

            ax.set_xlabel('Доза облучения (Gy)', fontsize=11)
            ax.set_ylabel(f'% клеток в {phase}', fontsize=11)
            ax.set_title(f'Дозовая зависимость: {phase}', fontsize=13)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

            # Добавляем сетку на фон
            ax.set_axisbelow(True)

        plt.tight_layout()
        output_file = config.PLOTS_DIR / 'dose_response_curves.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"    Сохранены графики дозовой зависимости")

    @staticmethod
    def plot_stacked_bars(df: pd.DataFrame):
        """Стековые диаграммы распределения фаз"""
        logger.info("  Генерация стековых диаграмм")

        # Агрегируем по времени
        phase_dist = df.groupby(['genotype', 'dose', 'phase']).size().unstack(fill_value=0)
        phase_dist = phase_dist.div(phase_dist.sum(axis=1), axis=0) * 100

        # Упорядочивание фаз
        phase_order = ['SubG1', 'G1', 'S', 'G2M', 'Mitosis']
        for phase in phase_order:
            if phase not in phase_dist.columns:
                phase_dist[phase] = 0

        phase_dist = phase_dist[phase_order]

        # Для каждого генотипа
        for genotype in df['genotype'].unique():
            if genotype not in phase_dist.index.get_level_values(0):
                continue

            subset = phase_dist.loc[genotype]

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = [config.PHASE_COLORS.get(phase, 'gray') for phase in phase_order]

            if isinstance(subset, pd.Series):
                # Если только одна доза
                subset = subset.to_frame().T

            subset.plot(
                kind='bar',
                stacked=True,
                ax=ax,
                color=colors,
                width=0.8,
                edgecolor='black',
                linewidth=0.5
            )

            ax.set_xlabel('Доза облучения (Gy)', fontsize=12)
            ax.set_ylabel('Процент клеток (%)', fontsize=12)
            ax.set_title(f'Распределение фаз клеточного цикла: {genotype}', fontsize=14)
            ax.legend(title='Фаза', loc='upper left', bbox_to_anchor=(1.02, 1))
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            output_file = config.PLOTS_DIR / f'stacked_bars_{genotype}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            logger.debug(f"    Сохранена стековая диаграмма для {genotype}")

    @staticmethod
    def plot_feature_distributions(df: pd.DataFrame):
        """Распределение признаков по фазам"""
        logger.info("  Генерация распределений признаков")

        # Выбираем ключевые признаки
        features_to_plot = ['area', 'intensity_linear_norm', 'circularity']

        # Создаем фигуру
        n_features = min(3, len(features_to_plot))
        fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 4))

        if n_features == 1:
            axes = [axes]

        for idx, feature in enumerate(features_to_plot[:n_features]):
            ax = axes[idx]

            for phase, color in config.PHASE_COLORS.items():
                if phase not in ['SubG1', 'G1', 'S', 'G2M']:
                    continue

                subset = df[df['phase'] == phase]
                if not subset.empty and feature in subset.columns:
                    # Используем KDE для гладкого отображения
                    try:
                        sns.kdeplot(
                            data=subset,
                            x=feature,
                            color=color,
                            label=phase,
                            ax=ax,
                            alpha=0.6,
                            linewidth=2
                        )
                    except:
                        # Если KDE не работает, используем гистограмму
                        ax.hist(
                            subset[feature],
                            bins=30,
                            alpha=0.5,
                            density=True,
                            label=phase,
                            color=color
                        )

            # Форматирование названия признака
            feature_name = feature.replace('_', ' ').title()
            if feature == 'intensity_linear_norm':
                feature_name = 'Intensity (Normalized)'

            ax.set_xlabel(feature_name, fontsize=11)
            ax.set_ylabel('Плотность', fontsize=11)
            ax.set_title(f'Распределение {feature_name}', fontsize=13)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = config.PLOTS_DIR / 'feature_distributions.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"    Сохранены распределения признаков")

    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame):
        """Матрица корреляции признаков"""
        logger.info("  Генерация матрицы корреляции")

        # Выбираем числовые колонки
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Ограничиваем количество признаков для читаемости
        important_features = ['log_intensity_norm', 'area', 'circularity',
                              'mean_intensity', 'total_intensity']

        # Фильтруем только существующие колонки
        features_to_use = [f for f in important_features if f in df.columns]

        if len(features_to_use) < 2:
            logger.warning("Недостаточно признаков для матрицы корреляции")
            return

        # Вычисляем корреляцию
        corr_matrix = df[features_to_use].corr()

        # Визуализация
        plt.figure(figsize=(10, 8))

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8}
        )

        plt.title('Матрица корреляции признаков', fontsize=14)
        plt.tight_layout()

        output_file = config.PLOTS_DIR / 'correlation_matrix.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"    Сохранена матрица корреляции")


# ================= 6. VALIDATION & REPORTING =================

class ValidationEngine:
    """Движок валидации и генерации отчетов"""

    @staticmethod
    def validate_with_ground_truth(df: pd.DataFrame) -> Dict[str, float]:
        """Сравнение с Ground Truth данными"""
        logger.info("=== Валидация с Ground Truth ===")

        # Наши результаты (игнорируем время и агрегируем по дозам)
        our_results = df.groupby(['genotype', 'dose', 'phase']).size().unstack(fill_value=0)
        our_results = our_results.div(our_results.sum(axis=1), axis=0) * 100

        # Ground Truth
        gt_df = pd.DataFrame(config.GROUND_TRUTH_DATA)
        gt_pivot = gt_df.pivot_table(
            index=['Genotype', 'Dose'],
            columns='Phase',
            values='Value'
        ).fillna(0)

        # Объединяем S и G2M для сравнения (так как в GT часто они вместе)
        if 'S' in our_results.columns and 'S' not in gt_pivot.columns:
            # Если в GT нет S, объединяем S и G2M в наших данных
            our_results['G2M'] = our_results.get('S', 0) + our_results.get('G2M', 0)
            if 'S' in our_results.columns:
                our_results = our_results.drop(columns=['S'])

        # Общие индексы и фазы для сравнения
        common_indices = set(our_results.index).intersection(set(gt_pivot.index))
        common_phases = set(our_results.columns).intersection(set(gt_pivot.columns))

        if not common_indices or not common_phases:
            logger.warning("Нет общих данных для сравнения с Ground Truth")
            return {'MAE': 0.0, 'R2': 0.0}

        # Сбор данных для сравнения
        our_values = []
        gt_values = []
        comparison_data = []

        for idx in common_indices:
            genotype, dose = idx
            for phase in common_phases:
                if phase in our_results.columns and phase in gt_pivot.columns:
                    our_val = our_results.loc[idx, phase]
                    gt_val = gt_pivot.loc[idx, phase]

                    our_values.append(float(our_val))
                    gt_values.append(float(gt_val))
                    comparison_data.append({
                        'genotype': str(genotype),
                        'dose': int(dose),
                        'phase': str(phase),
                        'our_value': float(our_val),
                        'gt_value': float(gt_val),
                        'difference': float(abs(our_val - gt_val))
                    })

        if not our_values or not gt_values:
            logger.warning("Не удалось получить значения для сравнения")
            return {'MAE': 0.0, 'R2': 0.0}

        # Расчет метрик
        our_array = np.array(our_values)
        gt_array = np.array(gt_values)

        # MAE (Mean Absolute Error)
        mae = float(np.mean(np.abs(our_array - gt_array)))

        # R² (Coefficient of Determination)
        ss_res = float(np.sum((our_array - gt_array) ** 2))
        ss_tot = float(np.sum((gt_array - np.mean(gt_array)) ** 2))
        if ss_tot != 0:
            r2 = float(1 - (ss_res / ss_tot))
        else:
            r2 = 0.0

        logger.info(f"Результаты валидации:")
        logger.info(f"  MAE: {mae:.2f}%")
        logger.info(f"  R^2: {r2:.3f}")  # Используем ^ вместо ²

        # Визуализация сравнения
        comparison_df = pd.DataFrame(comparison_data)

        plt.figure(figsize=(10, 8))

        for phase in comparison_df['phase'].unique():
            subset = comparison_df[comparison_df['phase'] == phase]
            color = config.PHASE_COLORS.get(phase, 'gray')
            plt.scatter(subset['gt_value'], subset['our_value'],
                        label=phase, s=100, alpha=0.7, color=color)

        # Линия идеального совпадения
        max_val = max(max(our_values), max(gt_values)) * 1.1
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Идеальное совпадение')

        plt.xlabel('Ground Truth (%)', fontsize=12)
        plt.ylabel('Наши результаты (%)', fontsize=12)
        plt.title(f'Сравнение с Ground Truth\nMAE = {mae:.1f}%, R^2 = {r2:.3f}', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = config.PLOTS_DIR / 'validation_scatter.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        # Сохранение данных сравнения
        comparison_df.to_csv(config.INTERMEDIATE_DIR / 'validation_comparison.csv', index=False)

        return {'MAE': mae, 'R2': r2}

    @staticmethod
    def calculate_summary_metrics(df: pd.DataFrame) -> Dict[str, Any]:
        """Расчет сводных метрик анализа"""
        logger.info("Расчет сводных метрик...")

        metrics = {}

        # Основные метрики
        metrics['total_cells'] = int(len(df))
        metrics['total_samples'] = int(len(df['filename'].unique()))
        metrics['total_genotypes'] = int(len(df['genotype'].unique()))
        metrics['total_doses'] = int(len(df['dose'].unique()))
        metrics['total_time_points'] = int(len(df['time'].unique()))

        # Распределение фаз
        phase_counts = df['phase'].value_counts()
        phase_percentages = {str(k): float(v) for k, v in (phase_counts / len(df) * 100).round(2).items()}
        metrics['phase_distribution'] = phase_percentages

        # Качество сегментации
        if 'circularity' in df.columns:
            metrics['mean_circularity'] = float(df['circularity'].mean().round(3))
            metrics['median_circularity'] = float(df['circularity'].median().round(3))

        if 'area' in df.columns:
            metrics['mean_area'] = float(df['area'].mean().round(1))
            metrics['median_area'] = float(df['area'].median().round(1))

        # Интенсивность
        if 'mean_intensity' in df.columns:
            metrics['mean_intensity'] = float(df['mean_intensity'].mean().round(1))

        # Статистика по экспериментальным условиям
        cells_per_condition = df.groupby(['genotype', 'dose']).size()
        # Преобразуем кортежи в строки для JSON
        cells_per_condition_dict = {}
        for idx, count in cells_per_condition.items():
            key = f"{idx[0]}_{idx[1]}Gy"
            cells_per_condition_dict[key] = int(count)
        metrics['cells_per_condition'] = cells_per_condition_dict

        # Валидационные метрики
        validation_metrics = ValidationEngine.validate_with_ground_truth(df)
        metrics.update(validation_metrics)

        # Сохранение метрик
        metrics_file = config.RESULTS_DIR / 'analysis_metrics.json'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        logger.info(f"Сохранены метрики анализа: {metrics_file}")

        return metrics

    @staticmethod
    def generate_comprehensive_report(df: pd.DataFrame, metrics: Dict[str, Any]):
        """Генерация комплексного отчета"""
        logger.info("=== Генерация отчета ===")

        report_file = config.RESULTS_DIR / 'analysis_report.txt'

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ОТЧЕТ ОБ АНАЛИЗЕ ФАЗ КЛЕТОЧНОГО ЦИКЛА\n")
            f.write("=" * 80 + "\n\n")

            # Основная информация
            f.write("ОСНОВНАЯ ИНФОРМАЦИЯ:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Идентификатор запуска: {config.RUN_ID}\n")
            f.write(f"Всего проанализировано клеток: {metrics.get('total_cells', 0)}\n")
            f.write(f"Количество образцов: {metrics.get('total_samples', 0)}\n")
            f.write(f"Генотипы: {', '.join(df['genotype'].unique())}\n")
            f.write(f"Дозы облучения: {', '.join(map(str, sorted(df['dose'].unique())))} Gy\n")
            f.write(f"Временные точки: {', '.join(map(str, sorted(df['time'].unique())))} часов\n")
            f.write("\n")

            # Качество данных
            f.write("КАЧЕСТВО ДАННЫХ:\n")
            f.write("-" * 40 + "\n")
            if 'mean_circularity' in metrics:
                f.write(f"Средняя округлость клеток: {metrics['mean_circularity']}\n")
            if 'mean_area' in metrics:
                f.write(f"Средняя площадь клеток: {metrics['mean_area']} пикселей\n")
            if 'mean_intensity' in metrics:
                f.write(f"Средняя интенсивность: {metrics['mean_intensity']}\n")
            f.write("\n")

            # Распределение фаз
            f.write("РАСПРЕДЕЛЕНИЕ ФАЗ (ВСЕ ДАННЫЕ):\n")
            f.write("-" * 40 + "\n")
            phase_dist = metrics.get('phase_distribution', {})
            for phase in ['SubG1', 'G1', 'S', 'G2M', 'Mitosis']:
                if phase in phase_dist:
                    f.write(f"{phase:<10} {phase_dist[phase]:>6.1f}%\n")
            f.write("\n")

            # Валидация
            if 'MAE' in metrics and 'R2' in metrics:
                f.write("ВАЛИДАЦИЯ С GROUND TRUTH:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Средняя абсолютная ошибка (MAE): {metrics['MAE']:.2f}%\n")
                f.write(f"Коэффициент детерминации (R^2): {metrics['R2']:.3f}\n")

                # Интерпретация MAE
                if metrics['MAE'] < 10:
                    f.write("Интерпретация: Отличное соответствие (<10%)\n")
                elif metrics['MAE'] < 20:
                    f.write("Интерпретация: Хорошее соответствие (10-20%)\n")
                elif metrics['MAE'] < 30:
                    f.write("Интерпретация: Удовлетворительное соответствие (20-30%)\n")
                else:
                    f.write("Интерпретация: Требует улучшения (>30%)\n")
                f.write("\n")

            # Конфигурация анализа
            f.write("КОНФИГУРАЦИЯ АНАЛИЗА:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Метод нормализации: {config.NORMALIZATION_METHOD}\n")
            f.write(f"Использование KDE для пиков: {config.USE_KDE_FOR_PEAKS}\n")
            f.write(f"Расширенные признаки: {config.USE_ADVANCED_FEATURES}\n")
            f.write(f"GPU для Cellpose: {config.USE_GPU_CELLPOSE}\n")
            f.write("\n")

            # Файлы
            f.write("СОЗДАННЫЕ ФАЙЛЫ:\n")
            f.write("-" * 40 + "\n")

            # Подсчет файлов по типам
            file_types = {}
            total_size = 0
            for item in config.RESULTS_DIR.rglob('*'):
                if item.is_file():
                    ext = item.suffix.lower()
                    if ext not in file_types:
                        file_types[ext] = 0
                    file_types[ext] += 1
                    total_size += item.stat().st_size

            for ext, count in sorted(file_types.items()):
                if ext:
                    f.write(f"{ext.upper()[1:]:<10} {count:>4} файлов\n")

            f.write(f"\nВсего файлов: {sum(file_types.values())}\n")
            f.write(f"Общий размер: {total_size / (1024 ** 2):.1f} MB\n")
            f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("АНАЛИЗ ЗАВЕРШЕН\n")
            f.write("=" * 80 + "\n")

        logger.info(f"Отчет сохранен: {report_file}")

        # Краткий отчет для консоли
        print("\n" + "=" * 70)
        print("КРАТКИЙ ОТЧЕТ:")
        print("=" * 70)
        print(f"Всего клеток: {metrics.get('total_cells', 0)}")
        print(f"Образцов: {metrics.get('total_samples', 0)}")

        if 'MAE' in metrics:
            print(f"Точность (MAE): {metrics['MAE']:.1f}%")
            print(f"Качество (R^2): {metrics['R2']:.3f}")

        print(f"\nРаспределение фаз:")
        phase_dist = metrics.get('phase_distribution', {})
        for phase in ['SubG1', 'G1', 'S', 'G2M', 'Mitosis']:
            if phase in phase_dist:
                print(f"  {phase}: {phase_dist[phase]:.1f}%")

        print(f"\nРезультаты сохранены в: {config.RESULTS_DIR}")
        print("=" * 70)


# ================= 7. MAIN EXECUTION =================

def main():
    """Основная функция выполнения анализа"""
    try:
        # Шаг 1: Обработка изображений
        logger.info("\n[ШАГ 1] Обработка изображений и сегментация")
        processor = EnhancedImageProcessor()
        df_raw = processor.process_directory()

        if df_raw.empty:
            logger.error("Нет данных для анализа!")
            return

        logger.info(f"Всего обработано клеток: {len(df_raw)}")

        # Шаг 2: Нормализация интенсивности
        logger.info("\n[ШАГ 2] Нормализация интенсивности")
        normalizer = IntensityNormalizer()
        df_norm = normalizer.normalize(df_raw)

        # Шаг 3: Классификация фаз
        logger.info("\n[ШАГ 3] Классификация фаз клеточного цикла")
        classifier = CellPhaseClassifier()
        df_classified = classifier.classify(df_norm)

        # Шаг 4: Сохранение финальных данных
        logger.info("\n[ШАГ 4] Сохранение данных")
        final_data_file = config.RESULTS_DIR / 'final_classified_data.csv'
        df_classified.to_csv(final_data_file, index=False)
        logger.info(f"Финальные данные сохранены: {final_data_file}")

        # Шаг 5: Визуализация
        logger.info("\n[ШАГ 5] Визуализация результатов")
        VisualizationEngine.generate_all_plots(df_classified)

        # Шаг 6: Валидация и отчет
        logger.info("\n[ШАГ 6] Валидация и генерация отчета")
        metrics = ValidationEngine.calculate_summary_metrics(df_classified)
        ValidationEngine.generate_comprehensive_report(df_classified, metrics)

        # Итоговая статистика
        logger.info("\n" + "=" * 70)
        logger.info("АНАЛИЗ УСПЕШНО ЗАВЕРШЕН!")
        logger.info("=" * 70)

        # Вывод времени выполнения
        end_time = datetime.now()
        start_time_str = config.RUN_ID  # RUN_ID содержит время начала
        start_time = datetime.strptime(start_time_str, '%Y%m%d_%H%M%S')
        duration = end_time - start_time

        logger.info(f"Время выполнения: {duration}")
        logger.info(f"Результаты сохранены в: {config.RESULTS_DIR}")

    except KeyboardInterrupt:
        logger.info("Анализ прерван пользователем")
        print("\nАнализ прерван. Промежуточные результаты сохранены.")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        print(f"\nПроизошла ошибка: {e}")
        print("Детали смотрите в лог-файле.")
        # Не поднимаем исключение дальше, чтобы не прерывать выполнение
        return


if __name__ == "__main__":
    main()