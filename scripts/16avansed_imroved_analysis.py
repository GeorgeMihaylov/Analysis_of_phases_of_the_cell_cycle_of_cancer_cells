import re
import logging
import warnings
import numpy as np
import pandas as pd
import cv2
import torch

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from tqdm import tqdm

from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score

import xgboost as xgb
import lightgbm as lgb

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import shannon_entropy

from cellpose import models as cp_models, io

warnings.filterwarnings('ignore')


# ================= ENHANCED CONFIGURATION =================
class EnhancedConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
    RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
    RESULTS_DIR = PROJECT_ROOT / 'results' / f"enhanced_analysis_{RUN_ID}"

    MODELS_DIR = RESULTS_DIR / 'models'
    CELL_CROPS_DIR = RESULTS_DIR / 'cell_crops'

    USE_GPU_CELLPOSE = torch.cuda.is_available()
    CELLPOSE_PRETRAINED_MODEL = 'cpsam'
    MIN_CELL_SIZE = 100
    MAX_CELL_SIZE = 5000

    EXTRACT_TEXTURE_FEATURES = True
    EXTRACT_GRADIENT_FEATURES = True
    EXTRACT_FOURIER_DESCRIPTORS = True
    SAVE_CELL_CROPS = True

    LBP_RADIUS = 1
    LBP_POINTS = 8
    HARALICK_DISTANCES = [1, 2, 3]
    HARALICK_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    CV_FOLDS = 5
    RANDOM_STATE = 42

    CNN_INPUT_SIZE = (128, 128)

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
        log_file = self.RESULTS_DIR / f'enhanced_analysis_{self.RUN_ID}.log'
        self.logger = logging.getLogger('EnhancedCellCycleAnalysis')
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        self.logger.handlers = []
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        global logger
        logger = self.logger


config = EnhancedConfig()
logger = config.logger


# ================= FEATURES =================
class EnhancedFeatureExtractor:
    @staticmethod
    def extract_texture_lbp(img_region: np.ndarray) -> Dict[str, float]:
        lbp = local_binary_pattern(
            img_region,
            config.LBP_POINTS,
            config.LBP_RADIUS,
            method='uniform'
        )
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=config.LBP_POINTS + 2,
            range=(0, config.LBP_POINTS + 2)
        )
        hist = hist.astype(float) / (hist.sum() + 1e-7)
        return {f'lbp_bin_{i}': float(hist[i]) for i in range(len(hist))}

    @staticmethod
    def extract_haralick(img_region: np.ndarray) -> Dict[str, float]:
        if img_region.size == 0:
            return {}
        vmin, vmax = float(np.min(img_region)), float(np.max(img_region))
        if vmax <= vmin:
            return {}
        img_normalized = ((img_region - vmin) / (vmax - vmin) * 255).astype(np.uint8)

        glcm = graycomatrix(
            img_normalized,
            distances=config.HARALICK_DISTANCES,
            angles=config.HARALICK_ANGLES,
            levels=256,
            symmetric=True,
            normed=True
        )

        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        features = {}
        for prop in properties:
            values = graycoprops(glcm, prop)
            features[f'haralick_{prop}_mean'] = float(np.mean(values))
            features[f'haralick_{prop}_std'] = float(np.std(values))
        return features

    @staticmethod
    def extract_gradient_features(img_region: np.ndarray) -> Dict[str, float]:
        if img_region.size == 0:
            return {}
        img_region = img_region.astype(np.float32)

        grad_x = cv2.Sobel(img_region, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_region, cv2.CV_32F, 0, 1, ksize=3)

        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        gradient_direction = np.arctan2(grad_y, grad_x)

        return {
            'gradient_mean': float(np.mean(gradient_magnitude)),
            'gradient_std': float(np.std(gradient_magnitude)),
            'gradient_max': float(np.max(gradient_magnitude)),
            'gradient_direction_entropy': float(shannon_entropy(gradient_direction))
        }

    @staticmethod
    def extract_fourier_descriptors(contour: np.ndarray, n_descriptors: int = 10) -> Dict[str, float]:
        contour_complex = contour[:, 0, 0] + 1j * contour[:, 0, 1]
        fourier_result = np.fft.fft(contour_complex)
        fourier_descriptors = np.abs(fourier_result)[:n_descriptors]
        if fourier_descriptors[0] != 0:
            fourier_descriptors = fourier_descriptors / fourier_descriptors[0]
        return {f'fourier_desc_{i}': float(fourier_descriptors[i]) for i in range(len(fourier_descriptors))}

    @staticmethod
    def extract_intensity_features(pixels: np.ndarray) -> Dict[str, float]:
        if pixels.size == 0:
            return {}
        pixels = pixels.astype(np.float32)
        s = pd.Series(pixels.ravel())
        return {
            'intensity_mean': float(np.mean(pixels)),
            'intensity_median': float(np.median(pixels)),
            'intensity_std': float(np.std(pixels)),
            'intensity_min': float(np.min(pixels)),
            'intensity_max': float(np.max(pixels)),
            'intensity_range': float(np.ptp(pixels)),
            'intensity_iqr': float(np.percentile(pixels, 75) - np.percentile(pixels, 25)),
            'intensity_skewness': float(s.skew()),
            'intensity_kurtosis': float(s.kurtosis()),
            'intensity_entropy': float(shannon_entropy(pixels))
        }

    @staticmethod
    def extract_morphological_features(mask_region: np.ndarray, contour: np.ndarray) -> Dict[str, float]:
        area = float(np.sum(mask_region > 0))
        perimeter = float(cv2.arcLength(contour, True))

        features = {
            'area': float(area),
            'perimeter': float(perimeter),
            'circularity': float(4 * np.pi * area / (perimeter ** 2 + 1e-7)),
        }

        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                (_, _), axes, orientation = ellipse
                major_axis = float(max(axes))
                minor_axis = float(min(axes))

                features.update({
                    'ellipse_major': float(major_axis),
                    'ellipse_minor': float(minor_axis),
                    'ellipse_ratio': float(minor_axis / (major_axis + 1e-7)),
                    'ellipse_eccentricity': float(np.sqrt(max(0.0, 1 - (minor_axis ** 2) / (major_axis ** 2 + 1e-7)))),
                    'ellipse_orientation': float(orientation)
                })
            except Exception:
                pass

        try:
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            hull_perimeter = float(cv2.arcLength(hull, True))
            features.update({
                'solidity': float(area / (hull_area + 1e-7)),
                'convexity': float(hull_perimeter / (perimeter + 1e-7))
            })
        except Exception:
            pass

        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        for i, hu in enumerate(hu_moments):
            features[f'hu_moment_{i}'] = float(-np.sign(hu) * np.log10(np.abs(hu) + 1e-10))

        return features


# ================= IMAGE PROCESSING =================
class EnhancedImageProcessor:
    def __init__(self):
        logger.info(f"Инициализация CellposeModel (GPU={config.USE_GPU_CELLPOSE}, pretrained={config.CELLPOSE_PRETRAINED_MODEL})...")
        self.segmentor = cp_models.CellposeModel(
            gpu=config.USE_GPU_CELLPOSE,
            pretrained_model=config.CELLPOSE_PRETRAINED_MODEL,
            model_type=None
        )
        self.feature_extractor = EnhancedFeatureExtractor()
        logger.info("Cellpose и Feature Extractor готовы")

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        vmin, vmax = float(np.min(img)), float(np.max(img))
        if vmax <= vmin:
            return np.zeros_like(img, dtype=np.uint8)
        out = (img - vmin) / (vmax - vmin) * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)

    def _save_cell_crop(self, img_region: np.ndarray, mask_region: np.ndarray,
                        cell_id: int, crop_prefix: str) -> str:
        img_masked = img_region.astype(np.float32) * mask_region.astype(np.float32)
        img_resized = cv2.resize(img_masked, config.CNN_INPUT_SIZE, interpolation=cv2.INTER_AREA)
        img_u8 = self._to_uint8(img_resized)

        crop_file = config.CELL_CROPS_DIR / f"{crop_prefix}_cell_{cell_id}.png"
        cv2.imwrite(str(crop_file), img_u8)
        return str(crop_file)

    def _extract_all_features(self, img: np.ndarray, masks: np.ndarray,
                              cell_id: int, crop_prefix: str) -> Optional[Dict[str, Any]]:
        mask_bool = (masks == cell_id)

        area = int(np.sum(mask_bool))
        if area < config.MIN_CELL_SIZE or area > config.MAX_CELL_SIZE:
            return None

        y_idx, x_idx = np.where(mask_bool)
        if y_idx.size == 0 or x_idx.size == 0:
            return None

        y_min, y_max = int(y_idx.min()), int(y_idx.max()) + 1
        x_min, x_max = int(x_idx.min()), int(x_idx.max()) + 1

        img_region = img[y_min:y_max, x_min:x_max]
        mask_region = mask_bool[y_min:y_max, x_min:x_max].astype(np.uint8)

        contours, _ = cv2.findContours(mask_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)

        features: Dict[str, Any] = {}
        features.update(self.feature_extractor.extract_intensity_features(img[mask_bool]))
        features.update(self.feature_extractor.extract_morphological_features(mask_region, contour))

        if config.EXTRACT_TEXTURE_FEATURES:
            try:
                img_masked = img_region.astype(np.float32) * mask_region.astype(np.float32)
                features.update(self.feature_extractor.extract_texture_lbp(img_masked))
            except Exception as e:
                logger.debug(f"LBP extraction failed: {e}")

        if config.EXTRACT_TEXTURE_FEATURES:
            try:
                img_masked = img_region.astype(np.float32) * mask_region.astype(np.float32)
                features.update(self.feature_extractor.extract_haralick(img_masked))
            except Exception as e:
                logger.debug(f"Haralick extraction failed: {e}")

        if config.EXTRACT_GRADIENT_FEATURES:
            try:
                features.update(self.feature_extractor.extract_gradient_features(img_region))
            except Exception as e:
                logger.debug(f"Gradient extraction failed: {e}")

        if config.EXTRACT_FOURIER_DESCRIPTORS and len(contour) >= 10:
            try:
                features.update(self.feature_extractor.extract_fourier_descriptors(contour))
            except Exception as e:
                logger.debug(f"Fourier extraction failed: {e}")

        if config.SAVE_CELL_CROPS:
            features['crop_path'] = self._save_cell_crop(img_region, mask_region, cell_id, crop_prefix)

        return features

    def process_single_image(self, fpath: Path) -> Optional[pd.DataFrame]:
        logger.info(f"Обработка: {fpath.name}")

        try:
            img = io.imread(str(fpath))

            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif img.ndim > 3:
                raise ValueError(f"Неподдерживаемая размерность изображения: {img.shape}")

            out = self.segmentor.eval(
                img,
                diameter=None,
                channels=None,
                flow_threshold=0.4,
                cellprob_threshold=0.0,
                progress=None
            )
            if isinstance(out, tuple) and len(out) == 3:
                masks, flows, styles = out
            elif isinstance(out, tuple) and len(out) == 4:
                masks, flows, styles, _diams = out
            else:
                raise ValueError(f"Неожиданный формат выхода segmentor.eval: type={type(out)}")

            unique_cells = np.unique(masks)
            unique_cells = unique_cells[unique_cells != 0]
            logger.info(f"  Сегментировано: {len(unique_cells)} клеток")

            meta = self._parse_metadata(fpath.name)
            crop_prefix = fpath.stem

            cells_data = []
            for cell_id in tqdm(unique_cells, desc="Извлечение признаков", leave=False):
                feat = self._extract_all_features(img, masks, int(cell_id), crop_prefix)
                if feat is None:
                    continue
                cells_data.append({
                    'filename': fpath.name,
                    'cell_id': int(cell_id),
                    **meta,
                    **feat
                })

            df_image = pd.DataFrame(cells_data)
            logger.info(f"  Извлечено признаков для {len(df_image)} клеток")
            return df_image

        except Exception as e:
            logger.error(f"Ошибка обработки {fpath.name}: {e}", exc_info=True)
            return None

    @staticmethod
    def _parse_metadata(filename: str) -> Dict[str, Any]:
        pattern = re.compile(
            r"(?:HCT116[_-]?)?"
            r"(?P<genotype>WT|CDK8KO|CDK8|p53KO|p53KO_CDK8KO)"
            r"[_-](?P<time>\d+)h"
            r"[_-](?P<dose>\d+)Gy",
            re.IGNORECASE
        )
        match = pattern.search(filename)
        if match:
            data = match.groupdict()
            return {
                'genotype': str(data['genotype']).upper(),
                'time': int(data['time']),
                'dose': int(data['dose'])
            }
        return {'genotype': 'Unknown', 'time': 0, 'dose': 0}


# ================= ENSEMBLE (FIXED XGBOOST GPU CONFIG) =================
class EnsembleClassifier:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()

        self.feature_cols: List[str] = []
        self.feature_medians: Optional[pd.Series] = None
        self.feature_importance: Dict[str, Dict[str, float]] = {}

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        exclude_cols = ['filename', 'cell_id', 'genotype', 'time', 'dose',
                        'phase', 'crop_path', 'log_intensity_norm']
        return [c for c in df.columns if c not in exclude_cols]

    def _prepare_features(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        if fit:
            self.feature_cols = self._get_feature_cols(df)

        if not self.feature_cols:
            raise ValueError("feature_cols пустой: сначала вызовите train_ensemble().")

        X = df.reindex(columns=self.feature_cols).copy()

        if fit:
            self.feature_medians = X.median(numeric_only=True)

        med = self.feature_medians if self.feature_medians is not None else X.median(numeric_only=True)
        X = X.fillna(med)

        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    def _create_pseudo_labels(self, df: pd.DataFrame) -> pd.Series:
        gt_df = pd.DataFrame(config.GROUND_TRUTH_DATA)
        labels = pd.Series('Unknown', index=df.index, dtype='object')

        for (genotype, dose), group in df.groupby(['genotype', 'dose']):
            gt_subset = gt_df[(gt_df['Genotype'] == genotype) & (gt_df['Dose'] == dose)]
            if gt_subset.empty or 'intensity_mean' not in group.columns:
                continue

            intensities = group['intensity_mean'].values.reshape(-1, 1)
            n_components = len(gt_subset)

            gmm = GaussianMixture(n_components=n_components, random_state=config.RANDOM_STATE)
            gmm.fit(intensities)
            cluster_labels = gmm.predict(intensities)

            cluster_means = [float(intensities[cluster_labels == i].mean()) for i in range(n_components)]
            sorted_clusters = np.argsort(cluster_means)

            phase_map = {int(cluster_idx): str(gt_subset.iloc[i]['Phase']) for i, cluster_idx in enumerate(sorted_clusters)}
            labels.loc[group.index] = pd.Series([phase_map.get(int(l), 'Unknown') for l in cluster_labels],
                                                index=group.index, dtype='object')
        return labels

    def _make_xgb(self, n_classes: int) -> xgb.XGBClassifier:
        # В xgboost 3.x GPU включается через device='cuda' + tree_method='hist' [web:64][web:56]
        params = dict(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            objective='multi:softprob',
            num_class=n_classes,
            eval_metric='mlogloss',
            tree_method='hist',
            device='cuda' if config.USE_GPU_CELLPOSE else 'cpu'
        )
        return xgb.XGBClassifier(**params)

    def train_ensemble(self, df: pd.DataFrame):
        logger.info("=== Обучение ансамбля классификаторов ===")

        X = self._prepare_features(df, fit=True)
        y = self._create_pseudo_labels(df)

        valid_mask = (y != 'Unknown')
        X = X[valid_mask]
        y = y[valid_mask].astype(str)

        if len(y) < 100:
            logger.warning("Недостаточно данных для обучения ансамбля")
            self.models = {}
            return

        y_enc = self.label_encoder.fit_transform(y)
        n_classes = len(self.label_encoder.classes_)

        logger.info(f"Обучение на {len(y_enc)} клетках с {X.shape[1]} признаками")
        logger.info(f"Классы: {list(self.label_encoder.classes_)}")
        logger.info(f"Распределение классов: {dict(pd.Series(y).value_counts())}")

        logger.info("  Обучение Random Forest...")
        self.models['rf'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
        self.models['rf'].fit(X, y_enc)
        self.feature_importance['rf'] = dict(zip(self.feature_cols, self.models['rf'].feature_importances_))

        logger.info("  Обучение XGBoost...")
        self.models['xgb'] = self._make_xgb(n_classes)
        try:
            self.models['xgb'].fit(X, y_enc)
        except xgb.core.XGBoostError as e:
            # Fallback, если wheel без CUDA или GPU недоступен
            logger.warning(f"XGBoost GPU недоступен/не поддерживается, переключаюсь на CPU. Детали: {e}")
            self.models['xgb'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                objective='multi:softprob',
                num_class=n_classes,
                eval_metric='mlogloss',
                tree_method='hist',
                device='cpu'
            )
            self.models['xgb'].fit(X, y_enc)

        logger.info("  Обучение LightGBM...")
        # LightGBM GPU может быть недоступен в конкретной сборке; оставим cpu как максимально совместимый вариант.
        # Если хотите попробовать GPU, замените device на 'gpu' и добавьте try/except по аналогии.
        self.models['lgb'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.RANDOM_STATE,
            device='cpu',
            n_jobs=-1
        )
        self.models['lgb'].fit(X, y_enc)

        logger.info("  Обучение Gradient Boosting...")
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            random_state=config.RANDOM_STATE
        )
        self.models['gb'].fit(X, y_enc)

        logger.info("Кросс-валидация...")
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y_enc, cv=cv, scoring='accuracy', n_jobs=-1)
            logger.info(f"  {name.upper()}: {scores.mean():.3f} (+/- {scores.std():.3f})")

        logger.info("Обучение ансамбля завершено")
        self._save_models()

    def predict_ensemble(self, df: pd.DataFrame) -> pd.Series:
        if not self.models:
            return pd.Series(['Unknown'] * len(df), index=df.index)

        X = self._prepare_features(df, fit=False)

        preds = {}
        for name, model in self.models.items():
            preds[name] = model.predict(X).astype(int)

        preds_df = pd.DataFrame(preds, index=df.index)
        voted = preds_df.mode(axis=1)[0].astype(int).to_numpy()

        phases = self.label_encoder.inverse_transform(voted)
        return pd.Series(phases, index=df.index)

    def _save_models(self):
        import joblib
        for name, model in self.models.items():
            joblib.dump(model, config.MODELS_DIR / f'{name}_model.pkl')

        joblib.dump(self.scaler, config.MODELS_DIR / 'scaler.pkl')
        joblib.dump(self.label_encoder, config.MODELS_DIR / 'label_encoder.pkl')
        joblib.dump(self.feature_cols, config.MODELS_DIR / 'feature_cols.pkl')
        joblib.dump(self.feature_medians, config.MODELS_DIR / 'feature_medians.pkl')


# ================= MAIN =================
def main():
    logger.info("=" * 70)
    logger.info("ENHANCED CELL CYCLE ANALYSIS PIPELINE")
    logger.info("=" * 70)

    logger.info("\n[ШАГ 1] Обработка изображений с расширенными признаками")
    processor = EnhancedImageProcessor()

    files = [
        *config.RAW_DATA_DIR.glob('*.jpg'),
        *config.RAW_DATA_DIR.glob('*.png'),
        *config.RAW_DATA_DIR.glob('*.tif'),
        *config.RAW_DATA_DIR.glob('*.tiff'),
    ]
    files = sorted(files)

    if not files:
        logger.error(f"В папке нет изображений: {config.RAW_DATA_DIR}")
        return

    all_data = []
    for fpath in tqdm(files, desc="Обработка файлов"):
        df_img = processor.process_single_image(fpath)
        if df_img is not None and not df_img.empty:
            all_data.append(df_img)

    if not all_data:
        logger.error("Не удалось извлечь данные ни из одного изображения.")
        return

    df_all = pd.concat(all_data, ignore_index=True)
    logger.info(f"Всего обработано клеток: {len(df_all)}")

    raw_data_file = config.RESULTS_DIR / 'all_cells_enhanced.csv'
    df_all.to_csv(raw_data_file, index=False)
    logger.info(f"Данные сохранены: {raw_data_file}")

    logger.info("\n[ШАГ 2] Обучение ансамбля классификаторов")
    ensemble = EnsembleClassifier()
    ensemble.train_ensemble(df_all)

    logger.info("\n[ШАГ 3] Классификация с использованием ансамбля")
    df_all['phase'] = ensemble.predict_ensemble(df_all)

    logger.info("\n[ШАГ 4] Агрегация результатов")
    results = df_all.groupby(['genotype', 'dose', 'phase']).size().unstack(fill_value=0)
    results_pct = results.div(results.sum(axis=1), axis=0) * 100

    logger.info("\nРезультаты классификации (%):")
    logger.info(results_pct.to_string())

    results_file = config.RESULTS_DIR / 'classification_results.csv'
    results_pct.to_csv(results_file)
    logger.info(f"\nРезультаты сохранены: {results_file}")

    final_file = config.RESULTS_DIR / 'all_cells_classified.csv'
    df_all.to_csv(final_file, index=False)
    logger.info(f"Классифицированные данные: {final_file}")

    logger.info("\n" + "=" * 70)
    logger.info("АНАЛИЗ ЗАВЕРШЕН")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
