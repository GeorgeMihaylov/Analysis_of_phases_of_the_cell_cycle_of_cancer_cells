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
class KellyAuraConfig:
    # Структура проекта
    PROJECT_ROOT = Path(__file__).resolve().parent.parent  # scripts -> project root
    RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'kelly_auranofin'

    # Уникальная папка для каждого запуска
    RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
    RESULTS_DIR = PROJECT_ROOT / 'results' / f"kelly_aura_{RUN_ID}"

    # Подпапки
    CHECKPOINTS_DIR = RESULTS_DIR / 'checkpoints'
    MASKS_DIR = RESULTS_DIR / 'masks'
    LOGS_DIR = RESULTS_DIR / 'logs'
    PLOTS_DIR = RESULTS_DIR / 'plots'
    INTERMEDIATE_DIR = RESULTS_DIR / 'intermediate'

    # Настройки Cellpose
    USE_GPU_CELLPOSE = torch.cuda.is_available()
    MIN_CELL_SIZE = 80  # Уменьшено для округлых клеток
    MAX_CELL_SIZE = 6000  # Увеличено для возможных агрегатов
    SAVE_MASKS = True
    SAVE_SEGMENTATION_VIS = True

    # Параметры классификации
    INTENSITY_METRIC = 'mean_intensity'
    USE_ADVANCED_FEATURES = True
    EXTRACT_ENTROPY = True
    EXTRACT_ELLIPSE = True

    # Пороги для классификации
    SUBG1_DISTANCE_THRESHOLD = 0.35  # Немного уменьшено для ауранофина
    SUBG1_WEIGHT_THRESHOLD = 0.005  # Уменьшено, так как может быть много SubG1
    MITOSIS_CIRCULARITY_QUANTILE = 0.85  # Повышено для ауранофина
    MITOSIS_MIN_CIRCULARITY = 0.90  # Повышено для округлых клеток

    # Нормализация
    NORMALIZATION_METHOD = 'per_sample'
    USE_KDE_FOR_PEAKS = True

    # Паттерн имени файла для Kelly + ауранофин
    FILENAME_PATTERN = re.compile(
        r"\d+\)\s*Kelly\s+(?P<condition>ctrl|aura)\s*(?P<concentration>\d+\.?\d*)?\s*uM\s*(?P<time>\d+)h",
        re.IGNORECASE
    )

    # Ground Truth из табличных данных биологов
    GROUND_TRUTH_DATA = [
        # 2h
        {"Condition": "Ctrl", "Concentration": 0, "Time": 2, "Phase": "SubG1", "Value": 3.56},
        {"Condition": "Ctrl", "Concentration": 0, "Time": 2, "Phase": "G1", "Value": 58.71},
        {"Condition": "Ctrl", "Concentration": 0, "Time": 2, "Phase": "G2M", "Value": 37.10},
        {"Condition": "Aura", "Concentration": 0.5, "Time": 2, "Phase": "SubG1", "Value": 2.94},
        {"Condition": "Aura", "Concentration": 0.5, "Time": 2, "Phase": "G1", "Value": 59.23},
        {"Condition": "Aura", "Concentration": 0.5, "Time": 2, "Phase": "G2M", "Value": 37.33},
        {"Condition": "Aura", "Concentration": 1.0, "Time": 2, "Phase": "SubG1", "Value": 5.53},
        {"Condition": "Aura", "Concentration": 1.0, "Time": 2, "Phase": "G1", "Value": 64.88},
        {"Condition": "Aura", "Concentration": 1.0, "Time": 2, "Phase": "G2M", "Value": 29.08},
        {"Condition": "Aura", "Concentration": 2.0, "Time": 2, "Phase": "SubG1", "Value": 7.53},
        {"Condition": "Aura", "Concentration": 2.0, "Time": 2, "Phase": "G1", "Value": 64.16},
        {"Condition": "Aura", "Concentration": 2.0, "Time": 2, "Phase": "G2M", "Value": 27.70},
        # 6h (исключаем из анализа, но оставляем для полноты)
        {"Condition": "Ctrl", "Concentration": 0, "Time": 6, "Phase": "SubG1", "Value": 3.06},
        {"Condition": "Ctrl", "Concentration": 0, "Time": 6, "Phase": "G1", "Value": 57.67},
        {"Condition": "Ctrl", "Concentration": 0, "Time": 6, "Phase": "G2M", "Value": 38.70},
        {"Condition": "Aura", "Concentration": 0.5, "Time": 6, "Phase": "SubG1", "Value": 8.48},
        {"Condition": "Aura", "Concentration": 0.5, "Time": 6, "Phase": "G1", "Value": 58.12},
        {"Condition": "Aura", "Concentration": 0.5, "Time": 6, "Phase": "G2M", "Value": 32.36},
        {"Condition": "Aura", "Concentration": 1.0, "Time": 6, "Phase": "SubG1", "Value": 16.05},
        {"Condition": "Aura", "Concentration": 1.0, "Time": 6, "Phase": "G1", "Value": 51.39},
        {"Condition": "Aura", "Concentration": 1.0, "Time": 6, "Phase": "G2M", "Value": 31.10},
        {"Condition": "Aura", "Concentration": 2.0, "Time": 6, "Phase": "SubG1", "Value": 21.09},
        {"Condition": "Aura", "Concentration": 2.0, "Time": 6, "Phase": "G1", "Value": 53.52},
        {"Condition": "Aura", "Concentration": 2.0, "Time": 6, "Phase": "G2M", "Value": 24.04},
        # 24h
        {"Condition": "Ctrl", "Concentration": 0, "Time": 24, "Phase": "SubG1", "Value": 7.59},
        {"Condition": "Ctrl", "Concentration": 0, "Time": 24, "Phase": "G1", "Value": 62.00},
        {"Condition": "Ctrl", "Concentration": 0, "Time": 24, "Phase": "G2M", "Value": 29.72},
        {"Condition": "Aura", "Concentration": 0.5, "Time": 24, "Phase": "SubG1", "Value": 21.98},
        {"Condition": "Aura", "Concentration": 0.5, "Time": 24, "Phase": "G1", "Value": 57.08},
        {"Condition": "Aura", "Concentration": 0.5, "Time": 24, "Phase": "G2M", "Value": 20.51},
        {"Condition": "Aura", "Concentration": 1.0, "Time": 24, "Phase": "SubG1", "Value": 40.71},
        {"Condition": "Aura", "Concentration": 1.0, "Time": 24, "Phase": "G1", "Value": 50.61},
        {"Condition": "Aura", "Concentration": 1.0, "Time": 24, "Phase": "G2M", "Value": 8.59},
        {"Condition": "Aura", "Concentration": 2.0, "Time": 24, "Phase": "SubG1", "Value": 62.65},
        {"Condition": "Aura", "Concentration": 2.0, "Time": 24, "Phase": "G1", "Value": 28.89},
        {"Condition": "Aura", "Concentration": 2.0, "Time": 24, "Phase": "G2M", "Value": 8.06},
    ]

    # Фазы для анализа (убрали S, так как в таблице данных нет S)
    PHASES = ['SubG1', 'G1', 'G2M', 'Mitosis']

    # Цвета для фаз
    PHASE_COLORS = {
        'SubG1': 'gray',
        'G1': 'blue',
        'S': 'purple',
        'G2M': 'green',
        'Mitosis': 'red'
    }

    # Концентрации ауранофина
    CONCENTRATIONS = [0, 0.5, 1.0, 2.0]

    # Временные точки (исключаем 6h)
    TIMEPOINTS = [2, 24]

    def __init__(self, exclude_6h=True):
        self.exclude_6h = exclude_6h

        # Создание всех необходимых директорий
        for d in [self.RESULTS_DIR, self.CHECKPOINTS_DIR, self.MASKS_DIR,
                  self.LOGS_DIR, self.PLOTS_DIR, self.INTERMEDIATE_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        # Настройка логирования
        self.setup_logging()

        logger.info("=" * 70)
        logger.info(f"Анализ Kelly с ауранофином: {self.RUN_ID}")
        logger.info(f"Проект: {self.PROJECT_ROOT}")
        logger.info(f"Данные: {self.RAW_DATA_DIR}")
        logger.info(f"Результаты: {self.RESULTS_DIR}")
        logger.info(f"Исключить 6h: {self.exclude_6h}")
        logger.info("=" * 70)

    def setup_logging(self):
        """Настройка системы логирования"""
        log_file = self.LOGS_DIR / f'kelly_aura_{self.RUN_ID}.log'

        # Создаем логгер
        self.logger = logging.getLogger('KellyAuraAnalysis')
        self.logger.setLevel(logging.DEBUG)

        # Форматер
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Файловый обработчик (все сообщения)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Консольный обработчик (только INFO и выше)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # Очищаем существующие обработчики и добавляем новые
        self.logger.handlers = []
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Глобальный логгер для использования в коде
        global logger
        logger = self.logger


config = KellyAuraConfig(exclude_6h=True)
logger = config.logger


# ================= 1. METADATA PARSING =================

class KellyMetadataParser:
    """Парсер метаданных для Kelly + ауранофин"""

    @staticmethod
    def parse(filename: str) -> Dict[str, Any]:
        """Извлечение метаданных из имени файла"""
        # Убираем номер и скобки в начале
        clean_name = re.sub(r'^\d+\)\s*', '', filename)

        match = config.FILENAME_PATTERN.search(clean_name)
        if match:
            data = match.groupdict()

            # Преобразование типов
            data['time'] = int(data['time'])

            # Обработка концентрации
            condition = data['condition'].lower()
            if condition == 'ctrl':
                data['concentration'] = 0.0
                data['condition'] = 'Ctrl'
            else:
                try:
                    data['concentration'] = float(data['concentration']) if data['concentration'] else 0.0
                except:
                    data['concentration'] = 0.0
                data['condition'] = 'Aura'

            # Пропускаем 6h, если установлено в конфиге
            if config.exclude_6h and data['time'] == 6:
                logger.debug(f"Пропуск 6h файла: {filename}")
                return None

            # Создаем уникальный идентификатор образца
            data['sample_id'] = f"Kelly_{data['condition']}_{data['concentration']}uM_{data['time']}h"

            logger.debug(f"Парсинг {filename}: {data}")
            return data

        # Попробуем альтернативный парсинг для простых случаев
        alt_patterns = [
            r"(?P<condition>ctrl|control|control)",
            r"(?P<concentration>\d+\.?\d*)\s*uM",
            r"(?P<time>\d+)h"
        ]

        logger.warning(f"Не удалось распарсить имя файла по паттерну: {filename}")

        # Ручное извлечение из известных имен
        filename_lower = filename.lower()

        data = {
            'condition': 'Unknown',
            'concentration': 0.0,
            'time': 0,
            'sample_id': 'Unknown'
        }

        # Определяем время
        for time in [2, 6, 24]:
            if f"{time}h" in filename_lower:
                data['time'] = time
                break

        # Определяем условие
        if 'ctrl' in filename_lower:
            data['condition'] = 'Ctrl'
            data['concentration'] = 0.0
        elif 'aura' in filename_lower:
            data['condition'] = 'Aura'
            # Ищем концентрацию
            for conc in [0.5, 1, 2]:
                if f"{conc}u" in filename_lower or f"{conc} µ" in filename_lower:
                    data['concentration'] = float(conc)
                    break

        data['sample_id'] = f"Kelly_{data['condition']}_{data['concentration']}uM_{data['time']}h"

        if config.exclude_6h and data['time'] == 6:
            return None

        return data


# ================= 2. IMAGE PROCESSING =================

class KellyImageProcessor:
    """Процессор изображений для Kelly с улучшенной обработкой морфологии"""

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

    def _extract_enhanced_features(self, img: np.ndarray, mask: np.ndarray, cell_id: int) -> Optional[Dict]:
        """Извлечение расширенных признаков для клеток Kelly"""
        mask_bool = (mask == cell_id)

        # Базовые метрики
        area = np.sum(mask_bool)

        # Фильтрация по размеру
        if area < config.MIN_CELL_SIZE or area > config.MAX_CELL_SIZE:
            return None

        # Интенсивностные признаки
        intensity_values = img[mask_bool]
        if len(intensity_values) == 0:
            return None

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

        # Дополнительные морфологические признаки для ауранофина
        features = {
            'area': area,
            'total_intensity': total_intensity,
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'circularity': circularity,
            'perimeter': perimeter,
        }

        # Эксцентриситет (для веретеновидных vs округлых)
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
                features['axis_ratio'] = major_axis / minor_axis if minor_axis > 0 else 1
            except:
                features['eccentricity'] = 0
                features['axis_ratio'] = 1

        # Текстура (энтропия)
        if config.EXTRACT_ENTROPY:
            hist, _ = np.histogram(intensity_values, bins=32, range=(0, 255))
            hist = hist / hist.sum() if hist.sum() > 0 else hist
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            features['entropy'] = entropy

        # Компактность (отношение площади к выпуклой оболочке)
        try:
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            features['solidity'] = solidity
        except:
            features['solidity'] = 0

        # Форм-фактор (для оценки формы)
        features['form_factor'] = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        # Вариация интенсивности
        features['intensity_cv'] = std_intensity / mean_intensity if mean_intensity > 0 else 0

        return features

    def process_single_image(self, fpath: Path) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
        """Обработка одного изображения"""
        logger.info(f"Обработка: {fpath.name}")

        try:
            # Парсинг метаданных
            meta = KellyMetadataParser.parse(fpath.name)
            if meta is None:  # Пропуск 6h файлов
                return None, None

            # Загрузка изображения
            img = io.imread(str(fpath))
            if len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

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
                features = self._extract_enhanced_features(img, masks, cell_id)
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
                    'sample_id': meta.get('sample_id', 'Unknown'),
                    'condition': meta.get('condition', 'Unknown'),
                    'concentration': meta.get('concentration', 0),
                    'time': meta.get('time', 0),
                    'total_cells': n_cells,
                    'valid_cells': len(df_image),
                    'mean_area': float(df_image['area'].mean()),
                    'mean_intensity': float(df_image['mean_intensity'].mean()),
                    'mean_circularity': float(df_image['circularity'].mean())
                }
                stats_file = config.INTERMEDIATE_DIR / f"{fpath.stem}_stats.json"
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=2)

            # Сохранение масок
            if config.SAVE_MASKS:
                mask_file = config.MASKS_DIR / f"{fpath.stem}_mask.npy"
                np.save(str(mask_file), masks)
                logger.debug(f"  Сохранена маска: {mask_file}")

            # Визуализация сегментации
            if config.SAVE_SEGMENTATION_VIS and not df_image.empty:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                axes[0].imshow(img, cmap='gray')
                axes[0].set_title(f"{meta.get('sample_id', 'Unknown')}")
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

    def process_all_images(self) -> pd.DataFrame:
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
            raw_data_file = config.INTERMEDIATE_DIR / 'kelly_cells_raw.csv'
            df_all.to_csv(raw_data_file, index=False)
            logger.info(f"Сохранены все сырые данные: {raw_data_file}")

            return df_all

        logger.warning("Нет данных после обработки")
        return pd.DataFrame()


# ================= 3. INTENSITY NORMALIZATION =================

class KellyIntensityNormalizer:
    """Нормализация интенсивности для данных Kelly"""

    @staticmethod
    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Нормализация интенсивности по контрольным образцам"""
        logger.info("=== Нормализация интенсивности ===")

        if df.empty:
            return df

        df = df.copy()

        # 1. Логарифмирование интенсивности
        intensity_col = config.INTENSITY_METRIC
        df['log_intensity'] = np.log1p(df[intensity_col])

        # 2. Поиск G1 пика для каждого контрольного образца
        control_files = df[(df['condition'] == 'Ctrl') & (df['time'].isin(config.TIMEPOINTS))]['filename'].unique()

        if len(control_files) == 0:
            logger.warning("Нет контрольных образцов. Используем медиану всех данных.")
            reference_peak = df['log_intensity'].median()
        else:
            control_peaks = []

            for fname in control_files:
                subset = df[df['filename'] == fname]['log_intensity'].values

                if len(subset) < 10:
                    control_peaks.append(np.median(subset))
                    continue

                # Использование KDE для определения пика
                try:
                    kde = gaussian_kde(subset)
                    x_range = np.linspace(subset.min(), subset.max(), 200)
                    y_kde = kde(x_range)
                    peak_idx = np.argmax(y_kde)
                    peak_val = x_range[peak_idx]
                    control_peaks.append(peak_val)
                except Exception as e:
                    logger.warning(f"KDE не удалось для {fname}: {e}. Используем медиану.")
                    control_peaks.append(np.median(subset))

            reference_peak = np.median(control_peaks)
            logger.info(f"Референсный G1 пик (контроль): {reference_peak:.3f}")

        # 3. Выравнивание всех данных относительно референсного пика
        # Находим пик для каждого файла
        file_peaks = {}

        for fname in df['filename'].unique():
            subset = df[df['filename'] == fname]['log_intensity'].values

            if len(subset) < 5:
                file_peaks[fname] = np.median(subset)
                continue

            if config.USE_KDE_FOR_PEAKS:
                try:
                    kde = gaussian_kde(subset)
                    x_range = np.linspace(subset.min(), subset.max(), 200)
                    y_kde = kde(x_range)
                    peak_idx = np.argmax(y_kde)
                    peak_val = x_range[peak_idx]
                    file_peaks[fname] = peak_val
                except:
                    file_peaks[fname] = np.median(subset)
            else:
                file_peaks[fname] = np.median(subset)

        # 4. Применение нормализации
        def apply_normalization(row):
            fname = row['filename']
            if fname in file_peaks:
                shift = reference_peak - file_peaks[fname]
                return row['log_intensity'] + shift
            return row['log_intensity']

        df['log_intensity_norm'] = df.apply(apply_normalization, axis=1)

        # 5. Преобразование обратно в линейную шкалу
        df['intensity_linear_norm'] = np.expm1(df['log_intensity_norm'])

        # Статистика
        logger.info(f"Нормализация завершена:")
        logger.info(f"  Диапазон до: {df['log_intensity'].min():.2f} - {df['log_intensity'].max():.2f}")
        logger.info(f"  Диапазон после: {df['log_intensity_norm'].min():.2f} - {df['log_intensity_norm'].max():.2f}")

        return df


# ================= 4. CELL CYCLE CLASSIFICATION =================

class KellyCellClassifier:
    """Классификатор фаз клеточного цикла для данных Kelly"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.classification_results = {}

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Подготовка признаков для классификации"""
        # Базовые признаки
        base_features = ['log_intensity_norm', 'circularity', 'area']

        # Дополнительные признаки
        additional_features = []
        for feature in ['eccentricity', 'entropy', 'axis_ratio', 'intensity_cv']:
            if feature in df.columns:
                additional_features.append(feature)

        all_features = base_features + additional_features
        X = df[all_features].copy()

        # Заполнение пропусков медианой
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())

        # Нормализация
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Классификация фаз клеточного цикла"""
        logger.info("=== Классификация фаз клеточного цикла ===")

        df = df.copy()
        df['phase'] = 'Unknown'

        # Группировка по условиям эксперимента
        groups = df.groupby(['condition', 'concentration', 'time'])

        results = []

        for (condition, concentration, time), group in tqdm(groups, desc="Классификация групп"):
            if len(group) < 30:
                logger.warning(
                    f"Слишком мало клеток для {condition}, {concentration}uM, {time}h: {len(group)}. Пропускаем.")
                group['phase'] = 'Unknown'
                results.append(group)
                continue

            logger.debug(f"Классификация: {condition}, {concentration}uM, {time}h ({len(group)} клеток)")

            try:
                # Подготовка признаков
                X = self._prepare_features(group)

                # Используем 3 компонента (SubG1, G1, G2M) так как в таблице данных нет S
                n_components = 3

                # Инициализация по перцентилям интенсивности
                intensities = group['log_intensity_norm'].values
                percentiles = np.percentile(intensities, [10, 50, 90])
                means_init = percentiles.reshape(-1, 1)

                # GMM классификация
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='full',
                    means_init=means_init,
                    random_state=42,
                    n_init=3,
                    max_iter=200
                )

                gmm.fit(X)
                means = gmm.means_.flatten()
                weights = gmm.weights_

                # Сортируем компоненты по интенсивности
                sorted_idx = np.argsort(means)

                # Сопоставляем компоненты фазам
                mapping = {
                    sorted_idx[0]: 'SubG1',  # Самый низкий пик
                    sorted_idx[1]: 'G1',  # Средний пик
                    sorted_idx[2]: 'G2M'  # Высокий пик
                }

                # Предсказание фаз
                labels = gmm.predict(X)
                group['phase'] = [mapping.get(l, 'Unknown') for l in labels]

                # Дополнительная обработка SubG1
                # Проверяем, действительно ли SubG1 отделен от G1
                subg1_mean = means[sorted_idx[0]]
                g1_mean = means[sorted_idx[1]]
                distance = g1_mean - subg1_mean

                if distance < config.SUBG1_DISTANCE_THRESHOLD:
                    # Объединяем SubG1 с G1
                    group.loc[group['phase'] == 'SubG1', 'phase'] = 'G1'
                    logger.debug(f"  SubG1 объединен с G1 (расстояние: {distance:.3f})")

                # Выделение митоза внутри G2M по высокой округлости
                g2m_mask = group['phase'] == 'G2M'
                if g2m_mask.sum() > 5:
                    g2m_cells = group[g2m_mask]

                    # Определяем порог округлости
                    circ_thresh = g2m_cells['circularity'].quantile(config.MITOSIS_CIRCULARITY_QUANTILE)
                    final_thresh = max(circ_thresh, config.MITOSIS_MIN_CIRCULARITY)

                    # Также проверяем интенсивность (митозные клетки обычно ярче)
                    intensity_thresh = g2m_cells['log_intensity_norm'].quantile(0.8)

                    mitosis_mask = (
                            g2m_mask &
                            (group['circularity'] > final_thresh) &
                            (group['log_intensity_norm'] > intensity_thresh)
                    )

                    group.loc[mitosis_mask, 'phase'] = 'Mitosis'
                    logger.debug(f"  Выделено митозных клеток: {mitosis_mask.sum()}")

                # Сохранение статистики классификации
                phase_counts = group['phase'].value_counts().to_dict()
                self.classification_results[f"{condition}_{concentration}uM_{time}h"] = {
                    'n_cells': len(group),
                    'phases': phase_counts
                }

                logger.info(f"  {condition}, {concentration}uM, {time}h: {phase_counts}")

            except Exception as e:
                logger.error(f"Ошибка классификации {condition}, {concentration}uM, {time}h: {e}")
                group['phase'] = 'Unknown'

            results.append(group)

        # Объединение всех групп
        df_classified = pd.concat(results, ignore_index=True)

        # Сохранение результатов классификации
        results_file = config.INTERMEDIATE_DIR / 'classification_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.classification_results, f, indent=2)
        logger.debug(f"Сохранены результаты классификации: {results_file}")

        return df_classified


# ================= 5. VISUALIZATION =================

class KellyVisualization:
    """Визуализация результатов для Kelly + ауранофин"""

    @staticmethod
    def generate_all_plots(df: pd.DataFrame):
        """Генерация всех графиков"""
        logger.info("=== Генерация графиков ===")

        # Установка стиля
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.titlesize'] = 13
        plt.rcParams['axes.labelsize'] = 11

        try:
            # 1. Псевдо-цитометрия
            KellyVisualization.plot_pseudo_cytometry(df)

            # 2. Концентрационные зависимости
            KellyVisualization.plot_concentration_response(df)

            # 3. Временные зависимости
            KellyVisualization.plot_time_course(df)

            # 4. Морфологические изменения
            KellyVisualization.plot_morphology_changes(df)

            # 5. Валидация с ground truth
            KellyVisualization.plot_validation_curves(df)

            # 6. Сводные графики
            KellyVisualization.plot_summary_grid(df)

            logger.info("Все графики успешно сгенерированы")

        except Exception as e:
            logger.error(f"Ошибка при генерации графиков: {e}", exc_info=True)

    @staticmethod
    def plot_pseudo_cytometry(df: pd.DataFrame):
        """Графики псевдо-цитометрии"""
        logger.info("  Генерация графиков псевдо-цитометрии")

        # Фильтруем только интересующие временные точки
        df_plot = df[df['time'].isin(config.TIMEPOINTS)]

        # Создаем сетку графиков
        times = sorted(df_plot['time'].unique())
        concentrations = sorted(df_plot['concentration'].unique())

        fig, axes = plt.subplots(
            len(times), len(concentrations),
            figsize=(4.5 * len(concentrations), 3.5 * len(times)),
            squeeze=False
        )

        for i, time in enumerate(times):
            for j, conc in enumerate(concentrations):
                ax = axes[i, j]
                subset = df_plot[(df_plot['time'] == time) & (df_plot['concentration'] == conc)]

                if subset.empty:
                    ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center')
                    ax.set_title(f"{time}h, {conc}µM")
                    continue

                # Гистограмма по фазам
                for phase, color in config.PHASE_COLORS.items():
                    phase_subset = subset[subset['phase'] == phase]
                    if not phase_subset.empty:
                        ax.hist(
                            phase_subset['intensity_linear_norm'],
                            bins=40,
                            alpha=0.6,
                            density=True,
                            color=color,
                            label=phase,
                            histtype='stepfilled',
                            edgecolor='black',
                            linewidth=0.5
                        )

                ax.set_xlabel('Интенсивность')
                ax.set_ylabel('Плотность')
                ax.set_title(f"{time}h, {conc}µM")
                ax.grid(True, alpha=0.3)
                ax.ticklabel_format(style='plain', axis='x')

                # Легенда только в первом графике
                if i == 0 and j == 0:
                    ax.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        output_file = config.PLOTS_DIR / 'pseudo_cytometry_grid.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"    Сохранен: {output_file}")

    @staticmethod
    def plot_concentration_response(df: pd.DataFrame):
        """Графики зависимости от концентрации"""
        logger.info("  Генерация графиков концентрационной зависимости")

        # Подготовка данных
        df_summary = df.groupby(['condition', 'concentration', 'time', 'phase']).size().unstack(fill_value=0)
        df_summary = df_summary.div(df_summary.sum(axis=1), axis=0) * 100

        # Основные фазы
        phases_to_plot = ['SubG1', 'G1', 'G2M']

        # Создаем графики для каждого времени
        for time in config.TIMEPOINTS:
            fig, axes = plt.subplots(1, len(phases_to_plot), figsize=(15, 5))

            for idx, phase in enumerate(phases_to_plot):
                ax = axes[idx] if len(phases_to_plot) > 1 else axes

                # Для контроля и ауранофина
                for condition in ['Ctrl', 'Aura']:
                    if condition not in df_summary.index.get_level_values(0):
                        continue

                    # Фильтруем по условию и времени
                    condition_data = df_summary.loc[condition]
                    if time in condition_data.index.get_level_values(1):
                        time_data = condition_data.xs(time, level=1)

                        if phase in time_data.columns:
                            concentrations = sorted(time_data.index)
                            phase_values = [time_data.loc[c, phase] if phase in time_data.columns else 0
                                            for c in concentrations]

                            color = 'blue' if condition == 'Ctrl' else 'red'
                            linestyle = '-' if condition == 'Ctrl' else '--'

                            ax.plot(concentrations, phase_values,
                                    color=color, linestyle=linestyle,
                                    marker='o', linewidth=2,
                                    markersize=8, label=condition)

                ax.set_xlabel('Концентрация (µM)', fontsize=11)
                ax.set_ylabel(f'% клеток в {phase}', fontsize=11)
                ax.set_title(f'{time}h: {phase}', fontsize=13)
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xticks([0, 0.5, 1, 2])

            plt.tight_layout()
            output_file = config.PLOTS_DIR / f'concentration_response_{time}h.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

        logger.info(f"    Сохранены графики концентрационной зависимости")

    @staticmethod
    def plot_time_course(df: pd.DataFrame):
        """Графики временного хода"""
        logger.info("  Генерация графиков временного хода")

        # Подготовка данных
        phase_percentages = df.groupby(['concentration', 'time', 'phase']).size().unstack(fill_value=0)
        phase_percentages = phase_percentages.div(phase_percentages.sum(axis=1), axis=0) * 100

        # Фазы для анализа
        phases = ['SubG1', 'G1', 'G2M']

        fig, axes = plt.subplots(1, len(phases), figsize=(16, 5))

        for idx, phase in enumerate(phases):
            ax = axes[idx] if len(phases) > 1 else axes

            for conc in config.CONCENTRATIONS:
                if conc == 0:  # Контроль
                    continue

                if conc in phase_percentages.index.get_level_values(0):
                    conc_data = phase_percentages.loc[conc]

                    # Собираем значения для каждого времени
                    times = sorted(conc_data.index)
                    values = [conc_data.loc[t, phase] if phase in conc_data.columns else 0 for t in times]

                    ax.plot(times, values, 'o-', label=f'{conc}µM', linewidth=2, markersize=8)

            # Контрольная группа
            if 0 in phase_percentages.index.get_level_values(0):
                ctrl_data = phase_percentages.loc[0]
                times = sorted(ctrl_data.index)
                values = [ctrl_data.loc[t, phase] if phase in ctrl_data.columns else 0 for t in times]
                ax.plot(times, values, 'ko--', label='Ctrl', linewidth=2, markersize=8)

            ax.set_xlabel('Время (часы)', fontsize=11)
            ax.set_ylabel(f'% клеток в {phase}', fontsize=11)
            ax.set_title(f'Динамика {phase}', fontsize=13)
            ax.legend(title='Концентрация')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(config.TIMEPOINTS)

        plt.tight_layout()
        output_file = config.PLOTS_DIR / 'time_course_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"    Сохранен график временного хода")

    @staticmethod
    def plot_morphology_changes(df: pd.DataFrame):
        """Анализ морфологических изменений"""
        logger.info("  Генерация графиков морфологических изменений")

        # Фильтруем по временным точкам
        df_plot = df[df['time'].isin(config.TIMEPOINTS)]

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Округлость vs концентрация
        ax1 = axes[0, 0]
        for time in config.TIMEPOINTS:
            subset = df_plot[df_plot['time'] == time]
            means = subset.groupby('concentration')['circularity'].mean()
            stds = subset.groupby('concentration')['circularity'].std()

            concentrations = sorted(subset['concentration'].unique())
            means_values = [means[c] if c in means.index else 0 for c in concentrations]
            stds_values = [stds[c] if c in stds.index else 0 for c in concentrations]

            ax1.errorbar(concentrations, means_values, yerr=stds_values,
                         marker='o', capsize=5, label=f'{time}h', linewidth=2)

        ax1.set_xlabel('Концентрация (µM)', fontsize=11)
        ax1.set_ylabel('Средняя округлость', fontsize=11)
        ax1.set_title('Изменение округлости клеток', fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks([0, 0.5, 1, 2])

        # 2. Площадь vs концентрация
        ax2 = axes[0, 1]
        for time in config.TIMEPOINTS:
            subset = df_plot[df_plot['time'] == time]
            means = subset.groupby('concentration')['area'].mean()

            concentrations = sorted(subset['concentration'].unique())
            means_values = [means[c] if c in means.index else 0 for c in concentrations]

            ax2.plot(concentrations, means_values, 'o-', label=f'{time}h', linewidth=2)

        ax2.set_xlabel('Концентрация (µM)', fontsize=11)
        ax2.set_ylabel('Средняя площадь', fontsize=11)
        ax2.set_title('Изменение площади клеток', fontsize=13)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks([0, 0.5, 1, 2])

        # 3. Диаграмма рассеяния: округлость vs интенсивность
        ax3 = axes[1, 0]
        scatter_data = df_plot[df_plot['concentration'] == 2.0]  # Самая высокая концентрация

        if not scatter_data.empty:
            for phase in ['SubG1', 'G1', 'G2M']:
                phase_data = scatter_data[scatter_data['phase'] == phase]
                if not phase_data.empty:
                    ax3.scatter(phase_data['circularity'], phase_data['intensity_linear_norm'],
                                alpha=0.5, label=phase, s=30)

        ax3.set_xlabel('Округлость', fontsize=11)
        ax3.set_ylabel('Интенсивность', fontsize=11)
        ax3.set_title('Морфология при 2µM Aura', fontsize=13)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Распределение форм-фактора
        ax4 = axes[1, 1]
        if 'form_factor' in df_plot.columns:
            for conc in [0, 2]:  # Контроль и максимальная концентрация
                subset = df_plot[df_plot['concentration'] == conc]
                if not subset.empty:
                    ax4.hist(subset['form_factor'], bins=30, alpha=0.6,
                             label=f'{conc}µM', density=True)

        ax4.set_xlabel('Форм-фактор', fontsize=11)
        ax4.set_ylabel('Плотность', fontsize=11)
        ax4.set_title('Распределение форм-фактора', fontsize=13)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = config.PLOTS_DIR / 'morphology_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"    Сохранен анализ морфологии")

    @staticmethod
    def plot_validation_curves(df: pd.DataFrame):
        """Сравнение с ground truth данными"""
        logger.info("  Генерация графиков валидации")

        # Наши результаты
        our_results = df.groupby(['condition', 'concentration', 'time', 'phase']).size().unstack(fill_value=0)
        our_results = our_results.div(our_results.sum(axis=1), axis=0) * 100

        # Ground Truth данные
        gt_data = []
        for item in config.GROUND_TRUTH_DATA:
            if item['Time'] in config.TIMEPOINTS:  # Исключаем 6h
                gt_data.append(item)

        gt_df = pd.DataFrame(gt_data)
        gt_pivot = gt_df.pivot_table(
            index=['Condition', 'Concentration', 'Time'],
            columns='Phase',
            values='Value'
        ).fillna(0)

        # Фазы для сравнения
        phases_to_compare = ['SubG1', 'G1', 'G2M']

        # Создаем графики для каждой фазы
        for phase in phases_to_compare:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Собираем данные для графика
            comparisons = []

            for idx in set(our_results.index).intersection(set(gt_pivot.index)):
                condition, conc, time = idx

                if phase in our_results.columns and phase in gt_pivot.columns:
                    our_val = our_results.loc[idx, phase]
                    gt_val = gt_pivot.loc[idx, phase]

                    comparisons.append({
                        'condition': condition,
                        'concentration': conc,
                        'time': time,
                        'our_value': our_val,
                        'gt_value': gt_val,
                        'difference': our_val - gt_val
                    })

            if comparisons:
                comp_df = pd.DataFrame(comparisons)

                # Разные цвета для контроля и ауранофина
                colors = {'Ctrl': 'blue', 'Aura': 'red'}

                for condition in comp_df['condition'].unique():
                    subset = comp_df[comp_df['condition'] == condition]
                    ax.scatter(subset['gt_value'], subset['our_value'],
                               label=condition, s=100, alpha=0.7,
                               color=colors.get(condition, 'gray'))

                # Линия идеального совпадения
                max_val = max(comp_df['gt_value'].max(), comp_df['our_value'].max()) * 1.1
                ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5,
                        label='Идеальное совпадение')

                # Расчет R²
                from sklearn.metrics import r2_score
                r2 = r2_score(comp_df['gt_value'], comp_df['our_value'])

                ax.set_xlabel('Ground Truth (%)', fontsize=12)
                ax.set_ylabel('Наши результаты (%)', fontsize=12)
                ax.set_title(f'Сравнение с Ground Truth: {phase}\nR² = {r2:.3f}', fontsize=14)
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)

                # Добавляем аннотации для больших расхождений
                for _, row in comp_df.iterrows():
                    if abs(row['difference']) > 15:
                        label = f"{row['concentration']}µM {row['time']}h"
                        ax.annotate(label, (row['gt_value'], row['our_value']),
                                    fontsize=8, alpha=0.7)

                plt.tight_layout()
                output_file = config.PLOTS_DIR / f'validation_{phase}.png'
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()

        logger.info(f"    Сохранены графики валидации")

    @staticmethod
    def plot_summary_grid(df: pd.DataFrame):
        """Сводная сетка графиков"""
        logger.info("  Генерация сводной сетки графиков")

        fig = plt.figure(figsize=(18, 12))

        # 1. SubG1 зависимость от концентрации и времени
        ax1 = plt.subplot(2, 3, 1)
        subg1_data = df.groupby(['concentration', 'time'])['phase'].apply(
            lambda x: (x == 'SubG1').mean() * 100
        ).unstack()

        for time in subg1_data.columns:
            ax1.plot(subg1_data.index, subg1_data[time], 'o-', label=f'{time}h')

        ax1.set_xlabel('Концентрация (µM)')
        ax1.set_ylabel('% SubG1 клеток')
        ax1.set_title('Индукция апоптоза')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Соотношение G1/G2M
        ax2 = plt.subplot(2, 3, 2)
        g1_data = df.groupby(['concentration', 'time'])['phase'].apply(
            lambda x: (x == 'G1').mean() * 100
        ).unstack()

        g2m_data = df.groupby(['concentration', 'time'])['phase'].apply(
            lambda x: (x.isin(['G2M', 'Mitosis'])).mean() * 100
        ).unstack()

        for time in g1_data.columns:
            if time in g2m_data.columns:
                ratio = g1_data[time] / (g2m_data[time] + 1e-10)
                ax2.plot(g1_data.index, ratio, 'o-', label=f'{time}h')

        ax2.set_xlabel('Концентрация (µM)')
        ax2.set_ylabel('Соотношение G1/G2M')
        ax2.set_title('Нарушение клеточного цикла')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Средняя округлость
        ax3 = plt.subplot(2, 3, 3)
        circ_data = df.groupby(['concentration', 'time'])['circularity'].mean().unstack()

        for time in circ_data.columns:
            ax3.plot(circ_data.index, circ_data[time], 'o-', label=f'{time}h')

        ax3.set_xlabel('Концентрация (µM)')
        ax3.set_ylabel('Средняя округлость')
        ax3.set_title('Морфологические изменения')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Распределение фаз для 24h
        ax4 = plt.subplot(2, 3, 4)
        time_24h = df[df['time'] == 24]
        phase_dist_24h = time_24h.groupby(['concentration', 'phase']).size().unstack(fill_value=0)
        phase_dist_24h = phase_dist_24h.div(phase_dist_24h.sum(axis=1), axis=0) * 100

        colors = [config.PHASE_COLORS.get(p, 'gray') for p in ['SubG1', 'G1', 'G2M']]
        phase_dist_24h[['SubG1', 'G1', 'G2M']].plot(kind='bar', stacked=True, ax=ax4, color=colors)

        ax4.set_xlabel('Концентрация (µM)')
        ax4.set_ylabel('Процент клеток (%)')
        ax4.set_title('Распределение фаз (24h)')
        ax4.legend(title='Фаза')
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. Корреляция округлости с SubG1
        ax5 = plt.subplot(2, 3, 5)
        for conc in df['concentration'].unique():
            subset = df[df['concentration'] == conc]
            if not subset.empty:
                is_subg1 = subset['phase'] == 'SubG1'
                subg1_circularity = subset.loc[is_subg1, 'circularity'].mean() if is_subg1.any() else 0
                subg1_percent = is_subg1.mean() * 100

                ax5.scatter(subg1_percent, subg1_circularity, s=100, alpha=0.7, label=f'{conc}µM')

        ax5.set_xlabel('% SubG1 клеток')
        ax5.set_ylabel('Средняя округлость SubG1')
        ax5.set_title('Корреляция апоптоза с морфологией')
        ax5.legend(title='Концентрация')
        ax5.grid(True, alpha=0.3)

        # 6. Сводная таблица количества клеток
        ax6 = plt.subplot(2, 3, 6)
        cell_counts = df.groupby(['concentration', 'time']).size().unstack(fill_value=0)

        im = ax6.imshow(cell_counts.values, cmap='YlOrRd', aspect='auto')
        ax6.set_xticks(range(len(cell_counts.columns)))
        ax6.set_xticklabels(cell_counts.columns)
        ax6.set_yticks(range(len(cell_counts.index)))
        ax6.set_yticklabels(cell_counts.index)
        ax6.set_xlabel('Время (часы)')
        ax6.set_ylabel('Концентрация (µM)')
        ax6.set_title('Количество клеток по условиям')

        # Добавляем значения в ячейки
        for i in range(len(cell_counts.index)):
            for j in range(len(cell_counts.columns)):
                ax6.text(j, i, int(cell_counts.iloc[i, j]),
                         ha='center', va='center', color='black', fontsize=9)

        plt.colorbar(im, ax=ax6)

        plt.tight_layout()
        output_file = config.PLOTS_DIR / 'summary_grid.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"    Сохранена сводная сетка графиков")


# ================= 6. VALIDATION & REPORTING =================

class KellyValidation:
    """Валидация и генерация отчетов для Kelly"""

    @staticmethod
    def calculate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
        """Расчет метрик анализа"""
        logger.info("Расчет метрик анализа...")

        metrics = {}

        # Основные метрики
        metrics['total_cells'] = len(df)
        metrics['total_samples'] = len(df['filename'].unique())
        metrics['total_conditions'] = len(df['condition'].unique())
        metrics['total_concentrations'] = len(df['concentration'].unique())
        metrics['total_times'] = len(df['time'].unique())

        # Распределение фаз
        phase_counts = df['phase'].value_counts()
        phase_percentages = (phase_counts / len(df) * 100).round(2).to_dict()
        metrics['phase_distribution'] = phase_percentages

        # Морфологические метрики
        if 'circularity' in df.columns:
            metrics['mean_circularity'] = float(df['circularity'].mean().round(3))
            metrics['median_circularity'] = float(df['circularity'].median().round(3))

        if 'area' in df.columns:
            metrics['mean_area'] = float(df['area'].mean().round(1))
            metrics['median_area'] = float(df['area'].median().round(1))

        # Концентрационные зависимости
        for conc in df['concentration'].unique():
            subset = df[df['concentration'] == conc]
            if len(subset) > 0:
                metrics[f'conc_{conc}_cells'] = len(subset)
                subg1_percent = (subset['phase'] == 'SubG1').mean() * 100
                metrics[f'conc_{conc}_subg1_percent'] = float(subg1_percent.round(2))

        # Сравнение с ground truth
        validation_results = KellyValidation.compare_with_ground_truth(df)
        metrics.update(validation_results)

        # Сохранение метрик
        metrics_file = config.RESULTS_DIR / 'analysis_metrics.json'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        logger.info(f"Сохранены метрики анализа: {metrics_file}")

        return metrics

    @staticmethod
    def compare_with_ground_truth(df: pd.DataFrame) -> Dict[str, Any]:
        """Сравнение с табличными данными биологов"""
        logger.info("=== Сравнение с табличными данными ===")

        # Наши результаты
        our_results = df.groupby(['condition', 'concentration', 'time', 'phase']).size().unstack(fill_value=0)
        our_results = our_results.div(our_results.sum(axis=1), axis=0) * 100

        # Ground Truth данные (исключаем 6h если нужно)
        gt_data = []
        for item in config.GROUND_TRUTH_DATA:
            if item['Time'] in config.TIMEPOINTS or not config.exclude_6h:
                gt_data.append(item)

        gt_df = pd.DataFrame(gt_data)
        gt_pivot = gt_df.pivot_table(
            index=['Condition', 'Concentration', 'Time'],
            columns='Phase',
            values='Value'
        ).fillna(0)

        # Общие индексы для сравнения
        common_indices = set(our_results.index).intersection(set(gt_pivot.index))
        phases_to_compare = ['SubG1', 'G1', 'G2M']

        if not common_indices:
            logger.warning("Нет общих данных для сравнения с Ground Truth")
            return {'validation_MAE': 0, 'validation_R2': 0}

        # Сбор данных для сравнения
        our_values = []
        gt_values = []
        comparison_data = []

        for idx in common_indices:
            condition, conc, time = idx

            for phase in phases_to_compare:
                if phase in our_results.columns and phase in gt_pivot.columns:
                    our_val = our_results.loc[idx, phase] if phase in our_results.columns else 0
                    gt_val = gt_pivot.loc[idx, phase] if phase in gt_pivot.columns else 0

                    our_values.append(our_val)
                    gt_values.append(gt_val)

                    comparison_data.append({
                        'condition': condition,
                        'concentration': conc,
                        'time': time,
                        'phase': phase,
                        'our_value': our_val,
                        'gt_value': gt_val,
                        'difference': our_val - gt_val,
                        'absolute_error': abs(our_val - gt_val)
                    })

        if not our_values:
            logger.warning("Не удалось получить значения для сравнения")
            return {'validation_MAE': 0, 'validation_R2': 0}

        # Расчет метрик
        from sklearn.metrics import mean_absolute_error, r2_score

        mae = mean_absolute_error(gt_values, our_values)
        r2 = r2_score(gt_values, our_values)

        logger.info(f"Результаты сравнения с табличными данными:")
        logger.info(f"  MAE: {mae:.2f}%")
        logger.info(f"  R²: {r2:.3f}")

        # Сохранение данных сравнения
        comparison_df = pd.DataFrame(comparison_data)
        comparison_file = config.INTERMEDIATE_DIR / 'ground_truth_comparison.csv'
        comparison_df.to_csv(comparison_file, index=False)
        logger.info(f"Сохранены данные сравнения: {comparison_file}")

        return {
            'validation_MAE': float(mae),
            'validation_R2': float(r2),
            'validation_n_comparisons': len(comparison_data)
        }

    @staticmethod
    def generate_report(df: pd.DataFrame, metrics: Dict[str, Any]):
        """Генерация комплексного отчета"""
        logger.info("=== Генерация отчета ===")

        report_file = config.RESULTS_DIR / 'kelly_aura_analysis_report.txt'

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ОТЧЕТ ОБ АНАЛИЗЕ ВОЗДЕЙСТВИЯ АУРАНОФИНА НА КЛЕТКИ KELLY\n")
            f.write("=" * 80 + "\n\n")

            # Основная информация
            f.write("ОСНОВНАЯ ИНФОРМАЦИЯ:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Идентификатор запуска: {config.RUN_ID}\n")
            f.write(f"Всего проанализировано клеток: {metrics.get('total_cells', 0)}\n")
            f.write(f"Количество образцов: {metrics.get('total_samples', 0)}\n")
            f.write(f"Концентрации ауранофина: {', '.join([str(c) for c in config.CONCENTRATIONS])} µM\n")
            f.write(f"Временные точки: {', '.join([str(t) for t in config.TIMEPOINTS])} часов\n")
            f.write(f"Исключены 6h образцы: {config.exclude_6h}\n")
            f.write("\n")

            # Качество данных
            f.write("КАЧЕСТВО ДАННЫХ:\n")
            f.write("-" * 40 + "\n")
            if 'mean_circularity' in metrics:
                f.write(f"Средняя округлость клеток: {metrics['mean_circularity']}\n")
            if 'mean_area' in metrics:
                f.write(f"Средняя площадь клеток: {metrics['mean_area']} пикселей\n")
            f.write("\n")

            # Распределение фаз
            f.write("ОБЩЕЕ РАСПРЕДЕЛЕНИЕ ФАЗ:\n")
            f.write("-" * 40 + "\n")
            phase_dist = metrics.get('phase_distribution', {})
            for phase in ['SubG1', 'G1', 'G2M', 'Mitosis']:
                if phase in phase_dist:
                    f.write(f"{phase:<10} {phase_dist[phase]:>6.1f}%\n")
            f.write("\n")

            # Концентрационные зависимости
            f.write("КОНЦЕНТРАЦИОННЫЕ ЗАВИСИМОСТИ:\n")
            f.write("-" * 40 + "\n")
            for conc in config.CONCENTRATIONS:
                key = f'conc_{conc}_subg1_percent'
                if key in metrics:
                    f.write(f"{conc:>4} µM: {metrics[key]:>5.1f}% SubG1 клеток\n")
            f.write("\n")

            # Валидация с табличными данными
            if 'validation_MAE' in metrics:
                f.write("СРАВНЕНИЕ С ТАБЛИЧНЫМИ ДАННЫМИ БИОЛОГОВ:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Средняя абсолютная ошибка (MAE): {metrics['validation_MAE']:.2f}%\n")
                f.write(f"Коэффициент детерминации (R²): {metrics['validation_R2']:.3f}\n")
                f.write(f"Количество сравнений: {metrics.get('validation_n_comparisons', 0)}\n")

                # Интерпретация
                mae_val = metrics['validation_MAE']
                if mae_val < 10:
                    f.write("Интерпретация: Отличное соответствие с табличными данными\n")
                elif mae_val < 20:
                    f.write("Интерпретация: Хорошее соответствие с табличными данными\n")
                elif mae_val < 30:
                    f.write("Интерпретация: Удовлетворительное соответствие\n")
                else:
                    f.write("Интерпретация: Требует дополнительной калибровки\n")
                f.write("\n")

            # Ключевые выводы
            f.write("КЛЮЧЕВЫЕ ВЫВОДЫ:\n")
            f.write("-" * 40 + "\n")

            # Анализ SubG1 (апоптоз)
            subg1_percent = phase_dist.get('SubG1', 0)
            if subg1_percent > 30:
                f.write("• Выраженная индукция апоптоза (SubG1 > 30%)\n")
            elif subg1_percent > 15:
                f.write("• Умеренная индукция апоптоза (SubG1 > 15%)\n")
            elif subg1_percent > 5:
                f.write("• Слабая индукция апоптоза (SubG1 > 5%)\n")
            else:
                f.write("• Низкий уровень апоптоза\n")

            # Концентрационная зависимость
            highest_conc = max(config.CONCENTRATIONS)
            if highest_conc > 0:
                highest_key = f'conc_{highest_conc}_subg1_percent'
                if highest_key in metrics:
                    highest_subg1 = metrics[highest_key]
                    ctrl_key = 'conc_0_subg1_percent'
                    ctrl_subg1 = metrics.get(ctrl_key, 0)

                    if highest_subg1 > ctrl_subg1 * 3:
                        f.write("• Сильная концентрационная зависимость апоптоза\n")
                    elif highest_subg1 > ctrl_subg1 * 2:
                        f.write("• Умеренная концентрационная зависимость апоптоза\n")
                    else:
                        f.write("• Слабая концентрационная зависимость апоптоза\n")

            # Временная динамика
            f.write("• Анализ включает временные точки 2h и 24h\n")

            # Морфологические изменения
            if 'mean_circularity' in metrics:
                if metrics['mean_circularity'] > 0.7:
                    f.write("• Преобладают округлые формы клеток\n")
                elif metrics['mean_circularity'] > 0.5:
                    f.write("• Смешанные морфологии клеток\n")
                else:
                    f.write("• Преобладают веретеновидные формы клеток\n")

            f.write("\n")

            # Рекомендации
            f.write("РЕКОМЕНДАЦИИ:\n")
            f.write("-" * 40 + "\n")
            f.write("1. Для улучшения точности классификации:\n")
            f.write("   - Увеличить количество анализируемых клеток\n")
            f.write("   - Провести калибровку на Flow cytometry данных\n")
            f.write("2. Для дальнейшего анализа:\n")
            f.write("   - Добавить промежуточные концентрации\n")
            f.write("   - Изучить более ранние временные точки\n")
            f.write("   - Проанализировать другие морфологические параметры\n")
            f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО\n")
            f.write("=" * 80 + "\n")

        logger.info(f"Отчет сохранен: {report_file}")

        # Краткий отчет для консоли
        print("\n" + "=" * 70)
        print("КРАТКИЙ ОТЧЕТ - KELLY + АУРАНОФИН:")
        print("=" * 70)
        print(f"Всего клеток: {metrics.get('total_cells', 0)}")
        print(f"Образцов: {metrics.get('total_samples', 0)}")

        if 'validation_MAE' in metrics:
            print(f"Сравнение с табличными данными:")
            print(f"  MAE: {metrics['validation_MAE']:.1f}%")
            print(f"  R²: {metrics['validation_R2']:.3f}")

        print(f"\nРаспределение фаз:")
        phase_dist = metrics.get('phase_distribution', {})
        for phase in ['SubG1', 'G1', 'G2M', 'Mitosis']:
            if phase in phase_dist:
                print(f"  {phase}: {phase_dist[phase]:.1f}%")

        print(f"\nРезультаты сохранены в: {config.RESULTS_DIR}")
        print("=" * 70)


# ================= 7. MAIN EXECUTION =================

def main_kelly_analysis():
    """Основная функция анализа данных Kelly с ауранофином"""
    try:
        # Шаг 1: Обработка изображений
        logger.info("\n[ШАГ 1] Обработка изображений и сегментация")
        processor = KellyImageProcessor()
        df_raw = processor.process_all_images()

        if df_raw.empty:
            logger.error("Нет данных для анализа!")
            return

        # Шаг 2: Нормализация интенсивности
        logger.info("\n[ШАГ 2] Нормализация интенсивности")
        normalizer = KellyIntensityNormalizer()
        df_norm = normalizer.normalize(df_raw)

        # Шаг 3: Классификация фаз
        logger.info("\n[ШАГ 3] Классификация фаз клеточного цикла")
        classifier = KellyCellClassifier()
        df_classified = classifier.classify(df_norm)

        # Шаг 4: Сохранение финальных данных
        logger.info("\n[ШАГ 4] Сохранение данных")
        final_data_file = config.RESULTS_DIR / 'kelly_final_classified_data.csv'
        df_classified.to_csv(final_data_file, index=False)
        logger.info(f"Финальные данные сохранены: {final_data_file}")

        # Шаг 5: Визуализация
        logger.info("\n[ШАГ 5] Визуализация результатов")
        KellyVisualization.generate_all_plots(df_classified)

        # Шаг 6: Валидация и отчет
        logger.info("\n[ШАГ 6] Валидация и генерация отчета")
        metrics = KellyValidation.calculate_metrics(df_classified)
        KellyValidation.generate_report(df_classified, metrics)

        # Итоговая статистика
        logger.info("\n" + "=" * 70)
        logger.info("АНАЛИЗ ДАННЫХ KELLY С АУРАНОФИНОМ УСПЕШНО ЗАВЕРШЕН!")
        logger.info("=" * 70)

        # Вывод времени выполнения
        end_time = datetime.now()
        start_time_str = config.RUN_ID
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
        raise


if __name__ == "__main__":
    main_kelly_analysis()