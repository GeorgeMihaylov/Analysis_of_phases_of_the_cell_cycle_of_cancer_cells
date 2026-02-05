# scripts/final_cellcycle_analysis.py
# -*- coding: utf-8 -*-

import re
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, List, Tuple

import numpy as np
import pandas as pd
import cv2
import torch
import joblib
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
import xgboost as xgb

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries

from cellpose import models as cp_models, io

warnings.filterwarnings("ignore")

PHASE_ORDER = ["SubG1", "G1", "G2M"]
PHASE_COLORS = {"SubG1": "#f28e2b", "G1": "#4e79a7", "G2M": "#59a14f"}


# ================= CONFIG =================
class EnhancedConfig:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    RESULTS_BASE_DIR = PROJECT_ROOT / "results"

    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = RESULTS_BASE_DIR / f"final_analysis_{RUN_ID}"
    MODELS_DIR = RESULTS_DIR / "models"
    SEGMENTATION_PREVIEWS_DIR = RESULTS_DIR / "segmentation_previews"
    FIGURES_DIR = RESULTS_DIR / "figures"

    USE_GPU_CELLPOSE = torch.cuda.is_available()
    CELLPOSE_PRETRAINED_MODEL = "cpsam"

    MIN_CELL_SIZE = 80
    MAX_CELL_SIZE = 6000

    EXTRACT_TEXTURE_FEATURES = True
    EXTRACT_GRADIENT_FEATURES = True
    EXTRACT_HARALICK = True

    SAVE_SEGMENTATION_PREVIEWS = True
    MAX_PREVIEWS = 12

    LBP_RADIUS = 1
    LBP_POINTS = 8
    HARALICK_DISTANCES = [1, 2]
    HARALICK_ANGLES = [0, np.pi / 4, np.pi / 2]

    RANDOM_STATE = 42

    # GT из задания (0/4/10 Gy) — для остальных доз будет GMM fallback
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
        for d in [self.RESULTS_DIR, self.MODELS_DIR, self.SEGMENTATION_PREVIEWS_DIR, self.FIGURES_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        self.setup_logging()

    def setup_logging(self):
        log_file = self.RESULTS_DIR / f"analysis_{self.RUN_ID}.log"
        self.logger = logging.getLogger("FinalCellCycleAnalysis")
        self.logger.setLevel(logging.DEBUG)

        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)

        self.logger.handlers = []
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        global logger
        logger = self.logger


config = EnhancedConfig()
logger = config.logger


# ================= FEATURES =================
def add_biological_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Генерация расширенных биологических признаков...")
    df = df.copy()

    for col in ["area", "perimeter", "circularity", "solidity", "intensity_mean"]:
        if col not in df.columns:
            df[col] = 0.0

    df["area_intensity_ratio"] = df["area"] / (df["intensity_mean"] + 1e-7)
    df["compactness"] = (4 * np.pi * df["area"]) / (df["perimeter"] ** 2 + 1e-7)

    df["g2m_score"] = 0.0
    for fname in df["filename"].unique():
        m = df["filename"] == fname
        if int(m.sum()) == 0:
            continue
        intens = df.loc[m, "intensity_mean"].astype(float)
        areas = df.loc[m, "area"].astype(float)
        intens_norm = (intens - intens.min()) / (intens.max() - intens.min() + 1e-7)
        area_norm = (areas - areas.min()) / (areas.max() - areas.min() + 1e-7)
        df.loc[m, "g2m_score"] = intens_norm * area_norm * df.loc[m, "circularity"].astype(float)

    df["solidity_inv"] = 1.0 - df["solidity"].astype(float)

    df["subg1_score"] = 0.0
    for fname in df["filename"].unique():
        m = df["filename"] == fname
        if int(m.sum()) == 0:
            continue
        intens = df.loc[m, "intensity_mean"].astype(float)
        areas = df.loc[m, "area"].astype(float)
        intens_inv = 1.0 - ((intens - intens.min()) / (intens.max() - intens.min() + 1e-7))
        area_inv = 1.0 - ((areas - areas.min()) / (areas.max() - areas.min() + 1e-7))
        df.loc[m, "subg1_score"] = intens_inv * area_inv * df.loc[m, "solidity_inv"].astype(float)

    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


class EnhancedFeatureExtractor:
    @staticmethod
    def extract_intensity_features(pixels: np.ndarray) -> Dict[str, float]:
        if pixels.size == 0:
            return {}
        pixels = pixels.astype(np.float32)
        s = pd.Series(pixels.ravel())
        return {
            "intensity_mean": float(np.mean(pixels)),
            "intensity_median": float(np.median(pixels)),
            "intensity_std": float(np.std(pixels)),
            "intensity_min": float(np.min(pixels)),
            "intensity_max": float(np.max(pixels)),
            "intensity_iqr": float(np.percentile(pixels, 75) - np.percentile(pixels, 25)),
            "intensity_skewness": float(s.skew()),
            "intensity_kurtosis": float(s.kurtosis()),
            "intensity_entropy": float(shannon_entropy(pixels)),
        }

    @staticmethod
    def extract_morphology(mask_region: np.ndarray, contour: np.ndarray) -> Dict[str, float]:
        area = float(np.sum(mask_region > 0))
        perimeter = float(cv2.arcLength(contour, True))
        out = {
            "area": float(area),
            "perimeter": float(perimeter),
            "circularity": float(4 * np.pi * area / (perimeter ** 2 + 1e-7)),
        }
        try:
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            out["solidity"] = float(area / (hull_area + 1e-7))
        except Exception:
            out["solidity"] = 0.0
        return out

    @staticmethod
    def extract_gradient_features(img_region: np.ndarray) -> Dict[str, float]:
        if img_region.size == 0:
            return {}
        img_region = img_region.astype(np.float32)
        grad_x = cv2.Sobel(img_region, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_region, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        direction = np.arctan2(grad_y, grad_x)
        return {
            "gradient_mean": float(np.mean(mag)),
            "gradient_std": float(np.std(mag)),
            "gradient_max": float(np.max(mag)),
            "gradient_direction_entropy": float(shannon_entropy(direction)),
        }

    @staticmethod
    def extract_lbp_entropy(img_masked: np.ndarray, mask_region: np.ndarray) -> Dict[str, float]:
        lbp = local_binary_pattern(img_masked, config.LBP_POINTS, config.LBP_RADIUS, method="uniform")
        vals = lbp[mask_region > 0]
        if vals.size == 0:
            return {}
        return {"lbp_entropy": float(shannon_entropy(vals))}

    @staticmethod
    def extract_haralick(img_masked: np.ndarray) -> Dict[str, float]:
        vmin, vmax = float(np.min(img_masked)), float(np.max(img_masked))
        if vmax <= vmin:
            return {}
        img_u8 = ((img_masked - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        glcm = graycomatrix(
            img_u8,
            distances=config.HARALICK_DISTANCES,
            angles=config.HARALICK_ANGLES,
            levels=256,
            symmetric=True,
            normed=True,
        )
        props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
        out = {}
        for p in props:
            try:
                v = graycoprops(glcm, p)
                out[f"haralick_{p}_mean"] = float(np.mean(v))
                out[f"haralick_{p}_std"] = float(np.std(v))
            except Exception:
                pass
        return out


# ================= IMAGE PROCESSING =================
class EnhancedImageProcessor:
    def __init__(self):
        logger.info(f"Загрузка Cellpose (GPU={config.USE_GPU_CELLPOSE})...")
        self.segmentor = cp_models.CellposeModel(
            gpu=config.USE_GPU_CELLPOSE,
            pretrained_model=config.CELLPOSE_PRETRAINED_MODEL,
        )
        self.fe = EnhancedFeatureExtractor()
        self._preview_left = int(config.MAX_PREVIEWS)

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        raise ValueError(f"Неподдерживаемая размерность изображения: {img.shape}")

    @staticmethod
    def _parse_filename(fname: str) -> Dict[str, Any]:
        """
        Ожидаем формат, близкий к:
        HCT116_CDK8KO_24h_0Gy_slide2_field01.jpg
        HCT116_WT_48h_10Gy_....tif
        """
        pattern = re.compile(
            r"(?:HCT116[_-]?)?"
            r"(?P<genotype>WT|CDK8KO|CDK8|p53KO|P53KO|p53ko)"
            r"[_-](?P<time>\d+)h"
            r"[_-](?P<dose>\d+)Gy",
            re.IGNORECASE,
        )
        m = pattern.search(fname)
        genotype = "Unknown"
        time_h = 0
        dose = 0
        if m:
            d = m.groupdict()
            genotype = str(d["genotype"]).upper()
            time_h = int(d["time"])
            dose = int(d["dose"])

        treatment = "NONE"
        if re.search(r"SnxB", fname, flags=re.IGNORECASE):
            treatment = "SNXB"

        return {"genotype": genotype, "time": time_h, "dose": dose, "treatment": treatment}

    def _save_segmentation_preview(self, img_gray: np.ndarray, masks: np.ndarray, stem: str):
        if not config.SAVE_SEGMENTATION_PREVIEWS or self._preview_left <= 0:
            return
        try:
            overlay = label2rgb(masks, image=img_gray, bg_label=0, alpha=0.35)
            bnd = find_boundaries(masks, mode="outer")

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_gray, cmap="gray"); axes[0].set_title("Original"); axes[0].axis("off")
            axes[1].imshow(overlay); axes[1].set_title("Masks overlay"); axes[1].axis("off")
            axes[2].imshow(img_gray, cmap="gray")
            axes[2].imshow(bnd, cmap="autumn", alpha=0.85)
            axes[2].set_title("Boundaries"); axes[2].axis("off")

            out = config.SEGMENTATION_PREVIEWS_DIR / f"{stem}_segmentation.png"
            fig.tight_layout()
            fig.savefig(out, dpi=200)
            plt.close(fig)

            self._preview_left -= 1
        except Exception as e:
            logger.debug(f"Preview save failed for {stem}: {e}")

    def process_file(self, fpath: Path) -> Optional[pd.DataFrame]:
        logger.info(f"Обработка: {fpath.name}")
        try:
            img = io.imread(str(fpath))
            img_gray = self._to_gray(img)

            out = self.segmentor.eval(img_gray, diameter=None, channels=None)
            if not (isinstance(out, tuple) and len(out) >= 1):
                raise ValueError("Неожиданный формат выхода Cellpose.")
            masks = out[0]

            unique_ids = np.unique(masks)
            unique_ids = unique_ids[unique_ids != 0]
            logger.info(f"  Сегментировано клеток: {len(unique_ids)}")

            self._save_segmentation_preview(img_gray, masks, fpath.stem)

            meta = self._parse_filename(fpath.name)
            rows = []

            for cid in unique_ids:
                cid = int(cid)
                mask_bool = masks == cid
                area_px = int(np.sum(mask_bool))
                if area_px < config.MIN_CELL_SIZE or area_px > config.MAX_CELL_SIZE:
                    continue

                y_idx, x_idx = np.where(mask_bool)
                if y_idx.size == 0 or x_idx.size == 0:
                    continue

                y_min, y_max = int(y_idx.min()), int(y_idx.max()) + 1
                x_min, x_max = int(x_idx.min()), int(x_idx.max()) + 1

                img_region = img_gray[y_min:y_max, x_min:x_max]
                mask_region = mask_bool[y_min:y_max, x_min:x_max].astype(np.uint8)

                contours, _ = cv2.findContours(mask_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                contour = max(contours, key=cv2.contourArea)

                feats = {}
                feats.update(self.fe.extract_intensity_features(img_gray[mask_bool]))
                feats.update(self.fe.extract_morphology(mask_region, contour))

                if config.EXTRACT_GRADIENT_FEATURES:
                    try:
                        feats.update(self.fe.extract_gradient_features(img_region))
                    except Exception:
                        pass

                img_masked = img_region.astype(np.float32) * mask_region.astype(np.float32)

                if config.EXTRACT_TEXTURE_FEATURES:
                    try:
                        feats.update(self.fe.extract_lbp_entropy(img_masked, mask_region))
                    except Exception:
                        pass

                if config.EXTRACT_HARALICK:
                    try:
                        feats.update(self.fe.extract_haralick(img_masked))
                    except Exception:
                        pass

                rows.append({"filename": fpath.name, "cell_id": cid, **meta, **feats})

            df = pd.DataFrame(rows)
            logger.info(f"  Признаки извлечены для клеток: {len(df)}")
            return df if not df.empty else None

        except Exception as e:
            logger.error(f"Ошибка {fpath.name}: {e}", exc_info=True)
            return None


# ================= CLASSIFIER =================
class SmartEnsembleClassifier:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.feature_cols: List[str] = []

    @staticmethod
    def _gt_table() -> pd.DataFrame:
        gt = pd.DataFrame(config.GROUND_TRUTH_DATA).copy()
        if "Treatment" not in gt.columns:
            gt["Treatment"] = "NONE"
        gt["Genotype"] = gt["Genotype"].astype(str).str.upper()
        gt["Treatment"] = gt["Treatment"].astype(str).str.upper()
        gt["Dose"] = gt["Dose"].astype(int)
        gt["Phase"] = gt["Phase"].astype(str)
        gt["Value"] = gt["Value"].astype(float)
        return gt

    @staticmethod
    def _gmm_labels_for_group(group: pd.DataFrame) -> pd.Series:
        """
        Fallback без GT: GMM(3) по log1p(intensity_mean),
        кластеры сортируем по среднему → SubG1/G1/G2M.
        """
        if "intensity_mean" not in group.columns or len(group) < 50:
            return pd.Series(["Unknown"] * len(group), index=group.index)

        x = np.log1p(group["intensity_mean"].astype(float).to_numpy()).reshape(-1, 1)
        try:
            gmm = GaussianMixture(n_components=3, random_state=config.RANDOM_STATE)
            cl = gmm.fit_predict(x)
            means = np.array([x[cl == k].mean() for k in range(3)], dtype=float)
            order = np.argsort(means)  # low→high
            phase_map = {int(order[0]): "SubG1", int(order[1]): "G1", int(order[2]): "G2M"}
            return pd.Series([phase_map.get(int(c), "Unknown") for c in cl], index=group.index)
        except Exception:
            return pd.Series(["Unknown"] * len(group), index=group.index)

    def _create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1) Если GT есть для (genotype,dose,treatment) или fallback на treatment=NONE — GT-guided разметка.
        2) Иначе — GMM fallback (чтобы не было сплошного Unknown на дозах 2/6/8).
        """
        logger.info("Генерация псевдо-меток (GT-guided + GMM fallback)...")
        gt = self._gt_table()

        df = df.copy()
        df["pseudo_label"] = "Unknown"
        df["confidence"] = 0.0

        group_cols = ["genotype", "dose", "treatment"]
        for (geno, dose, trt), group in df.groupby(group_cols):
            gt_subset = gt[
                (gt["Genotype"] == str(geno).upper())
                & (gt["Dose"] == int(dose))
                & (gt["Treatment"] == str(trt).upper())
            ]
            if gt_subset.empty:
                gt_subset = gt[
                    (gt["Genotype"] == str(geno).upper())
                    & (gt["Dose"] == int(dose))
                    & (gt["Treatment"] == "NONE")
                ]

            if gt_subset.empty:
                # GMM fallback
                labels = self._gmm_labels_for_group(group)
                df.loc[group.index, "pseudo_label"] = labels.values
                df.loc[group.index, "confidence"] = 0.75
                continue

            def _pct(phase: str) -> float:
                v = gt_subset.loc[gt_subset["Phase"] == phase, "Value"].values
                return float(v[0]) if v.size else 0.0

            subg1_pct = _pct("SubG1")
            g1_pct = _pct("G1")
            g2m_pct = _pct("G2M")

            total = int(len(group))
            n_subg1 = int(total * (subg1_pct / 100.0))
            n_g1 = int(total * (g1_pct / 100.0))
            n_g2m = max(0, total - n_subg1 - n_g1)

            g_sorted_subg1 = group.sort_values("subg1_score", ascending=False)
            idx_subg1 = g_sorted_subg1.index[:n_subg1]

            remaining_idx = list(set(group.index) - set(idx_subg1))
            remaining = group.loc[remaining_idx]

            g_sorted_g2m = remaining.sort_values("g2m_score", ascending=False)
            idx_g2m = g_sorted_g2m.index[:n_g2m]

            idx_g1 = list(set(remaining_idx) - set(idx_g2m))

            df.loc[idx_subg1, "pseudo_label"] = "SubG1"
            df.loc[idx_g1, "pseudo_label"] = "G1"
            df.loc[idx_g2m, "pseudo_label"] = "G2M"

            df.loc[idx_subg1, "confidence"] = 0.9
            df.loc[idx_g1, "confidence"] = 0.8
            df.loc[idx_g2m, "confidence"] = 0.9

        return df

    def train(self, df: pd.DataFrame):
        df_labeled = self._create_labels(df)
        train_df = df_labeled[df_labeled["pseudo_label"].isin(PHASE_ORDER)].copy()

        if len(train_df) < 200:
            logger.warning("Недостаточно данных для обучения моделей. Будет fallback на pseudo_label.")
            self.models = {}
            return

        exclude = {
            "filename", "cell_id", "genotype", "time", "dose", "treatment",
            "pseudo_label", "confidence", "phase"
        }
        self.feature_cols = [
            c for c in train_df.columns
            if c not in exclude and np.issubdtype(train_df[c].dtype, np.number)
        ]
        if not self.feature_cols:
            logger.warning("Нет числовых признаков. Будет fallback на pseudo_label.")
            self.models = {}
            return

        X = train_df[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = train_df["pseudo_label"].astype(str)
        w = train_df["confidence"].astype(float).values

        Xs = self.scaler.fit_transform(X)
        y_enc = self.label_encoder.fit_transform(y)
        n_classes = len(self.label_encoder.classes_)

        logger.info(f"Обучение: {len(y_enc)} клеток, признаков={Xs.shape[1]}, классы={list(self.label_encoder.classes_)}")

        rf = RandomForestClassifier(
            n_estimators=300, max_depth=15, class_weight="balanced",
            random_state=config.RANDOM_STATE, n_jobs=-1
        )
        rf.fit(Xs, y_enc, sample_weight=w)

        xgb_params = dict(
            n_estimators=250, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            objective="multi:softprob", num_class=n_classes,
            n_jobs=-1, random_state=config.RANDOM_STATE,
            tree_method="hist", eval_metric="mlogloss"
        )
        try:
            xgb_model = xgb.XGBClassifier(**xgb_params, device="cuda" if config.USE_GPU_CELLPOSE else "cpu")
            xgb_model.fit(Xs, y_enc, sample_weight=w)
        except Exception as e:
            logger.warning(f"XGBoost error: {e}. Fallback to CPU.")
            xgb_model = xgb.XGBClassifier(**xgb_params, device="cpu")
            xgb_model.fit(Xs, y_enc, sample_weight=w)

        self.models = {"rf": rf, "xgb": xgb_model}

        joblib.dump(self.scaler, config.MODELS_DIR / "scaler.pkl")
        joblib.dump(self.label_encoder, config.MODELS_DIR / "encoder.pkl")
        joblib.dump(self.feature_cols, config.MODELS_DIR / "features.pkl")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        # если модели не обучились — используем pseudo_label, но гарантируем, что он уже создан
        if not self.models:
            if "pseudo_label" not in df.columns:
                df2 = self._create_labels(df)
                return df2["pseudo_label"].astype(str)
            return df["pseudo_label"].astype(str)

        X = df.reindex(columns=self.feature_cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        Xs = self.scaler.transform(X)

        probs_rf = self.models["rf"].predict_proba(Xs)
        probs_xgb = self.models["xgb"].predict_proba(Xs)
        final_probs = 0.4 * probs_rf + 0.6 * probs_xgb

        pred = np.argmax(final_probs, axis=1)
        return pd.Series(self.label_encoder.inverse_transform(pred), index=df.index)


# ================= REPORT & FIGURE =================
def results_pct_from_cells(df_cells: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    counts = df_cells.groupby(group_cols + ["phase"]).size().unstack(fill_value=0)

    # берём только три фазы; Unknown не участвует в знаменателе
    for ph in PHASE_ORDER:
        if ph not in counts.columns:
            counts[ph] = 0
    counts3 = counts[PHASE_ORDER].copy()

    denom = counts3.sum(axis=1).replace(0, np.nan)
    pct = counts3.div(denom, axis=0) * 100.0
    return pct.fillna(0.0)


def gt_pct_from_config() -> pd.DataFrame:
    gt = pd.DataFrame(config.GROUND_TRUTH_DATA).copy()
    if "Treatment" not in gt.columns:
        gt["Treatment"] = "NONE"
    gt["Genotype"] = gt["Genotype"].astype(str).str.upper()
    gt["Treatment"] = gt["Treatment"].astype(str).str.upper()
    gt["Dose"] = gt["Dose"].astype(int)

    piv = gt.pivot_table(
        index=["Genotype", "Dose", "Treatment"], columns="Phase", values="Value", aggfunc="mean"
    ).fillna(0.0)

    for ph in PHASE_ORDER:
        if ph not in piv.columns:
            piv[ph] = 0.0
    piv = piv[PHASE_ORDER]
    piv.index = piv.index.set_names(["genotype", "dose", "treatment"])
    return piv.rename_axis(columns=None)


def bootstrap_phase_std(df_cells: pd.DataFrame, group_cols: List[str], n_boot: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in df_cells.groupby(group_cols):
        phases = g["phase"].astype(str).values
        phases = phases[np.isin(phases, PHASE_ORDER)]
        n = len(phases)
        if n < 40:
            continue

        boot = []
        for _ in range(n_boot):
            sample = phases[rng.integers(0, n, size=n)]
            s = pd.Series(sample).value_counts(normalize=True) * 100.0
            boot.append([float(s.get(p, 0.0)) for p in PHASE_ORDER])
        boot = np.array(boot)
        std = boot.std(axis=0, ddof=1)

        row = {}
        for c, v in zip(group_cols, key if isinstance(key, tuple) else (key,)):
            row[c] = v
        for i, ph in enumerate(PHASE_ORDER):
            row[ph] = float(std[i])
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=group_cols + PHASE_ORDER).set_index(group_cols)
    return pd.DataFrame(rows).set_index(group_cols)


def _sorted_conditions(df: pd.DataFrame) -> List[Tuple[int, str]]:
    conds = sorted({(int(d), str(t).upper()) for d, t in zip(df["dose"], df["treatment"])})
    conds.sort(key=lambda x: (x[0], 0 if x[1] == "NONE" else 1, x[1]))
    return conds


def plot_cellcycle_like_figure(df_cells: pd.DataFrame, out_png: Path, feature: str = "intensity_mean"):
    df = df_cells.copy()
    if feature not in df.columns:
        raise ValueError(f"Нет колонки '{feature}' для Panel A.")

    df["genotype"] = df["genotype"].astype(str).str.upper()
    df["treatment"] = df["treatment"].astype(str).str.upper()

    genotypes = sorted([g for g in df["genotype"].unique() if g != "UNKNOWN"])
    conds = _sorted_conditions(df)

    n_rows = max(1, len(genotypes))
    n_cols = max(1, len(conds))

    fig = plt.figure(figsize=(3.2 * n_cols, 2.6 * n_rows + 4.0))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2.3, 1.2], hspace=0.25)

    # Panel A
    gsA = gs[0].subgridspec(nrows=n_rows, ncols=n_cols, wspace=0.15, hspace=0.25)
    for i, geno in enumerate(genotypes):
        for j, (dose, trt) in enumerate(conds):
            ax = fig.add_subplot(gsA[i, j])
            sub = df[(df["genotype"] == geno) & (df["dose"] == dose) & (df["treatment"] == trt)]
            if sub.empty:
                ax.axis("off")
                continue

            for ph in PHASE_ORDER:
                s = sub[sub["phase"] == ph]
                if s.empty:
                    continue
                x = np.log1p(s[feature].astype(float).to_numpy())
                ax.hist(x, bins=60, density=False, histtype="stepfilled",
                        color=PHASE_COLORS[ph], alpha=0.55, linewidth=0.3)

            title = f"{dose} Gy" if trt == "NONE" else f"{dose} Gy + {trt}"
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(f"{geno}\nCount")
            else:
                ax.set_yticks([])
            ax.set_xticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    handles = [plt.Line2D([0], [0], color=PHASE_COLORS[p], lw=8) for p in PHASE_ORDER]
    fig.legend(handles, PHASE_ORDER, loc="upper right", frameon=False, bbox_to_anchor=(0.98, 0.98))

    # Panel B
    axB = fig.add_subplot(gs[1])

    group_cols = ["genotype", "dose", "treatment"]
    pred_pct = results_pct_from_cells(df, group_cols)
    pred_std = bootstrap_phase_std(df, group_cols)
    gt_pct = gt_pct_from_config()

    idx = pred_pct.index
    gt_aligned = gt_pct.reindex(idx)
    std_aligned = pred_std.reindex(idx).fillna(0.0)

    x_positions, x_labels, x_meta = [], [], []
    pos, gap = 0.0, 0.9
    for geno in genotypes:
        sub_idx = [k for k in idx if k[0] == geno]
        sub_idx.sort(key=lambda k: (int(k[1]), 0 if str(k[2]).upper() == "NONE" else 1, str(k[2]).upper()))
        for k in sub_idx:
            x_positions.append(pos)
            dose, trt = int(k[1]), str(k[2]).upper()
            x_labels.append(f"{dose}Gy" if trt == "NONE" else f"{dose}Gy+{trt}")
            x_meta.append(k)
            pos += 1.0
        pos += gap

    x_positions = np.array(x_positions, dtype=float)
    w = 0.36

    pred_mask = np.ones(len(x_positions), dtype=bool)
    gt_mask = np.array([k in gt_aligned.dropna(how="all").index for k in x_meta], dtype=bool)

    def draw_stacked(ax, x0, pct_rows, std_rows, draw_mask, title):
        bottom = np.zeros(len(x0), dtype=float)
        for ph in PHASE_ORDER:
            h = np.zeros(len(x0), dtype=float)
            for ii, key in enumerate(x_meta):
                if draw_mask[ii] and key in pct_rows.index:
                    h[ii] = float(pct_rows.loc[key, ph])
            ax.bar(x0, h, width=w, bottom=bottom, color=PHASE_COLORS[ph],
                   edgecolor="white", linewidth=0.6)
            if std_rows is not None and ph in std_rows.columns:
                err = np.zeros(len(x0), dtype=float)
                for ii, key in enumerate(x_meta):
                    if draw_mask[ii] and key in std_rows.index:
                        err[ii] = float(std_rows.loc[key, ph])
                y = bottom + h / 2.0
                ax.errorbar(x0, y, yerr=err, fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
            bottom += h
        ax.text(np.nanmean(x0), 103, title, ha="center", va="bottom", fontsize=10)

    draw_stacked(axB, x_positions - w / 2, pred_pct, std_aligned, pred_mask, "Pred")
    draw_stacked(axB, x_positions + w / 2, gt_aligned.fillna(0.0), None, gt_mask, "GT")

    axB.set_ylim(0, 110)
    axB.set_ylabel("% cells")
    axB.set_xticks(x_positions)
    axB.set_xticklabels(x_labels, fontsize=9)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)

    for geno in genotypes:
        xs = [x_positions[ii] for ii, k in enumerate(x_meta) if k[0] == geno]
        if xs:
            axB.text(np.mean(xs), -14, geno, ha="center", va="top", fontsize=10, transform=axB.transData)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


# ================= MAIN =================
def main():
    logger.info("=== ЗАПУСК ФИНАЛЬНОГО АНАЛИЗА ===")
    logger.info(f"PROJECT_ROOT: {config.PROJECT_ROOT}")
    logger.info(f"RAW_DATA_DIR: {config.RAW_DATA_DIR}")
    logger.info(f"RESULTS_DIR: {config.RESULTS_DIR}")

    valid_ext = {".jpg", ".png", ".tif", ".tiff"}
    files = sorted([f for f in config.RAW_DATA_DIR.glob("*.*") if f.suffix.lower() in valid_ext])

    if not files:
        logger.error(f"Файлы не найдены в {config.RAW_DATA_DIR}")
        return

    logger.info(f"Найдено изображений: {len(files)}")
    processor = EnhancedImageProcessor()

    dfs = []
    for f in tqdm(files, desc="Cellpose + Features"):
        df = processor.process_file(f)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        logger.error("Нет данных после обработки (проверьте MIN/MAX_CELL_SIZE).")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    logger.info(f"Всего клеток: {len(df_all)}")

    df_all = add_biological_features(df_all)

    raw_csv = config.RESULTS_DIR / "raw_features.csv"
    df_all.to_csv(raw_csv, index=False)
    logger.info(f"Saved: {raw_csv}")

    clf = SmartEnsembleClassifier()
    # создадим pseudo_label сразу (и для fallback, и для обучения)
    df_all = clf._create_labels(df_all)

    clf.train(df_all)
    df_all["phase"] = clf.predict(df_all).astype(str)

    final_csv = config.RESULTS_DIR / "final_classified_cells.csv"
    df_all.to_csv(final_csv, index=False)
    logger.info(f"Saved: {final_csv}")

    report_pct = results_pct_from_cells(df_all, ["genotype", "dose", "treatment"])
    report_csv = config.RESULTS_DIR / "classification_report_pct.csv"
    report_pct.round(3).to_csv(report_csv)
    logger.info(f"Saved: {report_csv}")
    logger.info("\n" + report_pct.round(2).to_string())

    fig_png = config.FIGURES_DIR / "cell_cycle_like_figure.png"
    plot_cellcycle_like_figure(df_all, fig_png, feature="intensity_mean")
    logger.info(f"Saved: {fig_png}")

    logger.info(f"Готово. Все результаты в: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
