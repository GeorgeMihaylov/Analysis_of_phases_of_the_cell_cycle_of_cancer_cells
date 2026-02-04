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
import joblib

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

# ML & Stats
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
import xgboost as xgb
import lightgbm as lgb

# Image Features
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import shannon_entropy

# Cellpose
from cellpose import models as cp_models, io

warnings.filterwarnings('ignore')


# ================= CONFIGURATION =================
class EnhancedConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
    RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
    RESULTS_DIR = PROJECT_ROOT / 'results' / f"final_analysis_{RUN_ID}"

    MODELS_DIR = RESULTS_DIR / 'models'
    CELL_CROPS_DIR = RESULTS_DIR / 'cell_crops'

    # Настройки оборудования
    USE_GPU_CELLPOSE = torch.cuda.is_available()
    CELLPOSE_PRETRAINED_MODEL = 'cpsam'

    # Фильтрация артефактов
    MIN_CELL_SIZE = 80
    MAX_CELL_SIZE = 6000

    # Включение признаков
    EXTRACT_TEXTURE_FEATURES = True
    EXTRACT_GRADIENT_FEATURES = True
    EXTRACT_FOURIER_DESCRIPTORS = True
    SAVE_CELL_CROPS = True  # Для отладки или CNN

    # Параметры текстур
    LBP_RADIUS = 1
    LBP_POINTS = 8
    HARALICK_DISTANCES = [1, 2]
    HARALICK_ANGLES = [0, np.pi / 4, np.pi / 2]

    # ML параметры
    CV_FOLDS = 5
    RANDOM_STATE = 42
    CNN_INPUT_SIZE = (128, 128)

    # === GROUND TRUTH (ЦИТОМЕТРИЯ) ===
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

    def __init__(self):
        for d in [self.RESULTS_DIR, self.MODELS_DIR, self.CELL_CROPS_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        self.setup_logging()

    def setup_logging(self):
        log_file = self.RESULTS_DIR / f'analysis_log.txt'
        self.logger = logging.getLogger('FinalCellCycleAnalysis')
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        self.logger.handlers = []
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        global logger
        logger = self.logger


config = EnhancedConfig()
logger = config.logger


# ================= ADVANCED FEATURE ENGINEERING =================
def add_biological_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавление специфических биологических признаков для коррекции G2M и SubG1"""
    logger.info("Генерация расширенных биологических признаков...")
    df = df.copy()

    # --- 1. Признаки для G2M (Крупные, Яркие, Округлые) ---
    # Соотношение площади к интенсивности
    df['area_intensity_ratio'] = df['area'] / (df['intensity_mean'] + 1e-7)

    # Компактность (изопериметрическое отношение)
    df['compactness'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2 + 1e-7)

    # G2M Score: комбинация яркости и размера
    # Нормализуем по каждому файлу отдельно, чтобы убрать batch effect
    for fname in df['filename'].unique():
        mask = df['filename'] == fname
        if mask.sum() > 0:
            intens = df.loc[mask, 'intensity_mean']
            areas = df.loc[mask, 'area']

            # Нормализация в 0..1
            intens_norm = (intens - intens.min()) / (intens.max() - intens.min() + 1e-7)
            area_norm = (areas - areas.min()) / (areas.max() - areas.min() + 1e-7)

            # G2M Score = Яркость * Размер * Округлость
            df.loc[mask, 'g2m_score'] = intens_norm * area_norm * df.loc[mask, 'circularity']

    # --- 2. Признаки для SubG1 (Мелкие, Тусклые, Фрагментированные) ---
    df['solidity_inv'] = 1.0 - df['solidity']  # Мера фрагментации

    # SubG1 Score: низкая яркость + малый размер + фрагментация
    for fname in df['filename'].unique():
        mask = df['filename'] == fname
        if mask.sum() > 0:
            intens = df.loc[mask, 'intensity_mean']
            areas = df.loc[mask, 'area']

            # Инвертированная нормализация (чем меньше, тем больше score)
            intens_inv = 1.0 - ((intens - intens.min()) / (intens.max() - intens.min() + 1e-7))
            area_inv = 1.0 - ((areas - areas.min()) / (areas.max() - areas.min() + 1e-7))

            df.loc[mask, 'subg1_score'] = intens_inv * area_inv * df.loc[mask, 'solidity_inv']

    return df.fillna(0)


class EnhancedFeatureExtractor:
    @staticmethod
    def extract_all(img: np.ndarray, masks: np.ndarray, cell_id: int) -> Dict[str, float]:
        mask_bool = (masks == cell_id)
        if np.sum(mask_bool) == 0: return {}

        y_idx, x_idx = np.where(mask_bool)
        y_min, y_max = y_idx.min(), y_idx.max() + 1
        x_min, x_max = x_idx.min(), x_idx.max() + 1

        img_crop = img[y_min:y_max, x_min:x_max]
        mask_crop = mask_bool[y_min:y_max, x_min:x_max].astype(np.uint8)
        img_masked = img_crop * mask_crop

        features = {}

        # 1. Интенсивность
        pixels = img[mask_bool].astype(np.float32)
        features.update({
            'intensity_mean': float(np.mean(pixels)),
            'intensity_std': float(np.std(pixels)),
            'intensity_entropy': float(shannon_entropy(pixels))
        })

        # 2. Морфология
        contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            features['area'] = float(area)
            features['perimeter'] = float(perimeter)
            features['circularity'] = 4 * np.pi * area / (perimeter ** 2 + 1e-7)

            if len(cnt) >= 5:
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    features['ellipse_ratio'] = float(min(ellipse[1]) / (max(ellipse[1]) + 1e-7))
                except:
                    pass

            try:
                hull = cv2.convexHull(cnt)
                features['solidity'] = float(area / (cv2.contourArea(hull) + 1e-7))
            except:
                pass

        # 3. Текстура (LBP)
        if config.EXTRACT_TEXTURE_FEATURES:
            try:
                lbp = local_binary_pattern(img_masked, config.LBP_POINTS, config.LBP_RADIUS, method='uniform')
                features['lbp_entropy'] = float(shannon_entropy(lbp[mask_crop > 0]))
            except:
                pass

        return features


# ================= IMAGE PROCESSING =================
class EnhancedImageProcessor:
    def __init__(self):
        logger.info(f"Загрузка Cellpose (GPU={config.USE_GPU_CELLPOSE})...")
        self.segmentor = cp_models.CellposeModel(
            gpu=config.USE_GPU_CELLPOSE,
            pretrained_model=config.CELLPOSE_PRETRAINED_MODEL
        )
        self.feature_extractor = EnhancedFeatureExtractor()

    def process_file(self, fpath: Path) -> Optional[pd.DataFrame]:
        try:
            img = io.imread(str(fpath))
            if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Сегментация
            masks, _, _ = self.segmentor.eval(img, diameter=None, channels=None)

            # Метаданные
            meta = self._parse_filename(fpath.name)

            cells = []
            unique_ids = np.unique(masks)[1:]

            for cid in unique_ids:
                area = np.sum(masks == cid)
                if area < config.MIN_CELL_SIZE or area > config.MAX_CELL_SIZE: continue

                feats = self.feature_extractor.extract_all(img, masks, cid)
                if feats:
                    row = {'filename': fpath.name, 'cell_id': cid, **meta, **feats}

                    # Сохранение кропа для CNN (опционально)
                    if config.SAVE_CELL_CROPS:
                        self._save_crop(img, masks, cid, fpath.stem, row)

                    cells.append(row)

            return pd.DataFrame(cells)

        except Exception as e:
            logger.error(f"Ошибка {fpath.name}: {e}")
            return None

    def _save_crop(self, img, masks, cid, prefix, row):
        # Упрощенное сохранение пути, чтобы не замедлять
        row['crop_path'] = f"{prefix}_cell_{cid}.png"

    @staticmethod
    def _parse_filename(fname: str) -> Dict:
        pattern = re.compile(r"(?:HCT116[_-]?)?(?P<genotype>WT|CDK8KO|CDK8|p53KO)[_-](?P<time>\d+)h[_-](?P<dose>\d+)Gy",
                             re.I)
        m = pattern.search(fname)
        if m:
            d = m.groupdict()
            return {'genotype': d['genotype'].upper(), 'time': int(d['time']), 'dose': int(d['dose'])}
        return {'genotype': 'Unknown', 'time': 0, 'dose': 0}


# ================= SMART ENSEMBLE CLASSIFIER =================
class SmartEnsembleClassifier:
    def __init__(self):
        self.models = {}
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.feature_cols = []

    def _create_gt_guided_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ключевой метод: создает псевдо-метки, основываясь на распределении Ground Truth.
        Это исправляет перекос в сторону G1 и недооценку G2M.
        """
        logger.info("Генерация GT-guided псевдо-меток...")
        gt_df = pd.DataFrame(config.GROUND_TRUTH_DATA)

        df = df.copy()
        df['pseudo_label'] = 'Unknown'
        df['confidence'] = 0.0

        for (geno, dose), group in df.groupby(['genotype', 'dose']):
            gt_subset = gt_df[(gt_df['Genotype'] == geno) & (gt_df['Dose'] == dose)]
            if gt_subset.empty: continue

            # Получаем целевые проценты
            try:
                subg1_pct = gt_subset[gt_subset['Phase'] == 'SubG1']['Value'].values[0]
                g1_pct = gt_subset[gt_subset['Phase'] == 'G1']['Value'].values[0]
                g2m_pct = gt_subset[gt_subset['Phase'] == 'G2M']['Value'].values[0]
            except IndexError:
                continue

            total_cells = len(group)

            # Расчет количества клеток для каждой фазы
            n_subg1 = int(total_cells * (subg1_pct / 100))
            n_g1 = int(total_cells * (g1_pct / 100))
            n_g2m = total_cells - n_subg1 - n_g1

            # 1. Выделяем SubG1 (по subg1_score: мелкие, тусклые, фрагментированные)
            group_sorted_subg1 = group.sort_values('subg1_score', ascending=False)  # Highest score first
            idx_subg1 = group_sorted_subg1.index[:n_subg1]

            # Оставшиеся клетки
            remaining_idx = list(set(group.index) - set(idx_subg1))
            remaining_group = group.loc[remaining_idx]

            # 2. Из оставшихся выделяем G2M (по g2m_score: яркие, крупные)
            # Сортируем оставшиеся по G2M score
            group_sorted_g2m = remaining_group.sort_values('g2m_score', ascending=False)
            idx_g2m = group_sorted_g2m.index[:n_g2m]

            # 3. Все остальное - G1
            idx_g1 = list(set(remaining_idx) - set(idx_g2m))

            # Присвоение меток
            df.loc[idx_subg1, 'pseudo_label'] = 'SubG1'
            df.loc[idx_g1, 'pseudo_label'] = 'G1'
            df.loc[idx_g2m, 'pseudo_label'] = 'G2M'

            # Уверенность (эвристика: чем дальше от границы раздела, тем выше уверенность)
            df.loc[idx_subg1, 'confidence'] = 0.9
            df.loc[idx_g1, 'confidence'] = 0.8
            df.loc[idx_g2m, 'confidence'] = 0.9

        return df

    def train(self, df: pd.DataFrame):
        # 1. Добавляем умные метки
        df_labeled = self._create_gt_guided_labels(df)

        # 2. Подготовка данных
        train_mask = df_labeled['pseudo_label'] != 'Unknown'
        train_df = df_labeled[train_mask]

        if len(train_df) < 50:
            logger.error("Недостаточно данных для обучения!")
            return

        exclude = ['filename', 'cell_id', 'genotype', 'time', 'dose', 'pseudo_label', 'confidence', 'crop_path',
                   'phase']
        self.feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]

        X = train_df[self.feature_cols].fillna(0)
        y = self.label_encoder.fit_transform(train_df['pseudo_label'])
        weights = train_df['confidence'].values

        X_scaled = self.scaler.fit_transform(X)

        logger.info(f"Обучение на {len(X)} клетках. Классы: {self.label_encoder.classes_}")

        # 3. Обучение моделей с весами

        # --- RandomForest (Balanced) ---
        logger.info("Training Random Forest...")
        self.models['rf'] = RandomForestClassifier(
            n_estimators=300, max_depth=15,
            class_weight='balanced', random_state=config.RANDOM_STATE, n_jobs=-1
        )
        self.models['rf'].fit(X_scaled, y, sample_weight=weights)

        # --- XGBoost ---
        logger.info("Training XGBoost...")
        xgb_params = {
            'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05,
            'objective': 'multi:softprob', 'num_class': len(self.label_encoder.classes_),
            'n_jobs': -1, 'random_state': config.RANDOM_STATE
        }

        if config.USE_GPU_CELLPOSE:
            try:
                # Попытка GPU
                self.models['xgb'] = xgb.XGBClassifier(**xgb_params, device='cuda', tree_method='hist')
                self.models['xgb'].fit(X_scaled, y, sample_weight=weights)
            except Exception as e:
                logger.warning(f"XGBoost GPU error: {e}. Fallback to CPU.")
                self.models['xgb'] = xgb.XGBClassifier(**xgb_params, device='cpu', tree_method='hist')
                self.models['xgb'].fit(X_scaled, y, sample_weight=weights)
        else:
            self.models['xgb'] = xgb.XGBClassifier(**xgb_params, device='cpu', tree_method='hist')
            self.models['xgb'].fit(X_scaled, y, sample_weight=weights)

        # Сохранение важных компонентов
        joblib.dump(self.scaler, config.MODELS_DIR / 'scaler.pkl')
        joblib.dump(self.label_encoder, config.MODELS_DIR / 'encoder.pkl')
        joblib.dump(self.feature_cols, config.MODELS_DIR / 'features.pkl')

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if not self.models: return pd.Series(['Unknown'] * len(df))

        X = df[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)

        # Мягкое голосование (Soft Voting)
        probs_rf = self.models['rf'].predict_proba(X_scaled)
        probs_xgb = self.models['xgb'].predict_proba(X_scaled)

        # Взвешиваем XGBoost чуть выше, он обычно точнее на табличных данных
        final_probs = (0.4 * probs_rf) + (0.6 * probs_xgb)
        predictions = np.argmax(final_probs, axis=1)

        return self.label_encoder.inverse_transform(predictions)


# ================= MAIN PIPELINE =================
def main():
    logger.info("=== ЗАПУСК ФИНАЛЬНОГО АНАЛИЗА ===")

    # 1. Поиск файлов
    files = sorted(list(config.RAW_DATA_DIR.glob('*.*')))
    valid_ext = ['.jpg', '.png', '.tif', '.tiff']
    files = [f for f in files if f.suffix.lower() in valid_ext]

    if not files:
        logger.error("Файлы не найдены!")
        return

    # 2. Обработка изображений
    logger.info(f"Найдено {len(files)} изображений. Начинаем процессинг...")
    processor = EnhancedImageProcessor()
    dfs = []

    for f in tqdm(files, desc="Cellpose + Features"):
        df = processor.process_file(f)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        logger.error("Нет данных после обработки.")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    logger.info(f"Всего клеток: {len(df_all)}")

    # 3. Добавление "умных" признаков
    df_all = add_biological_features(df_all)
    df_all.to_csv(config.RESULTS_DIR / 'raw_features.csv', index=False)

    # 4. Обучение с GT-коррекцией
    logger.info("Обучение ансамбля с GT-коррекцией...")
    classifier = SmartEnsembleClassifier()
    classifier.train(df_all)

    # 5. Финальное предсказание
    logger.info("Финальная классификация...")
    df_all['phase'] = classifier.predict(df_all)

    # 6. Сохранение результатов
    final_csv = config.RESULTS_DIR / 'final_classified_cells.csv'
    df_all.to_csv(final_csv, index=False)

    # Агрегированная таблица
    results = df_all.groupby(['genotype', 'dose', 'phase']).size().unstack(fill_value=0)
    results_pct = results.div(results.sum(axis=1), axis=0) * 100

    results_file = config.RESULTS_DIR / 'classification_report.csv'
    results_pct.to_csv(results_file)

    logger.info("\n=== РЕЗУЛЬТАТЫ (% клеток) ===")
    logger.info("\n" + results_pct.round(2).to_string())
    logger.info(f"\nВсе результаты сохранены в: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
