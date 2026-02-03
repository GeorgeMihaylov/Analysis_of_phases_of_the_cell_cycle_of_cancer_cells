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
from cellpose import models as cp_models, io
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.signal import find_peaks


# ================= CONFIGURATION =================
class Config:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
    RESULTS_DIR = PROJECT_ROOT / 'results' / 'final_report_v3_fixed'  # Новая папка

    # Отключаем GPU для Cellpose, если были ошибки, но оставляем для Torch
    USE_GPU_CELLPOSE = torch.cuda.is_available()
    MIN_CELL_SIZE = 100

    # Расширенный паттерн для имен файлов
    FILENAME_PATTERN = re.compile(
        r"(?:HCT116[_-]?)?(?P<genotype>WT|CDK8KO|CDK8|p53KO|p53KO_CDK8KO)[_-](?P<time>\d+)h[_-](?P<dose>\d+)Gy",
        re.IGNORECASE
    )

    # Эталонные данные (Ground Truth)
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
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Data Dir: {self.RAW_DATA_DIR}")


config = Config()


# ================= 1. PARSING & PROCESSING =================

class MetadataParser:
    @staticmethod
    def parse(filename):
        match = config.FILENAME_PATTERN.search(filename)
        if match:
            data = match.groupdict()
            data['dose'] = int(data['dose'])
            geno = data['genotype'].upper()
            if 'CDK8' in geno and 'P53' not in geno and 'KO' in geno:
                data['genotype'] = 'CDK8KO'
            elif 'WT' in geno:
                data['genotype'] = 'WT'
            elif 'P53KO_CDK8KO' in geno:
                data['genotype'] = 'p53KO_CDK8KO'
            elif 'P53KO' in geno:
                data['genotype'] = 'p53KO'
            return data
        return {'genotype': 'Unknown', 'time': 'Unknown', 'dose': 0}


class ImageProcessor:
    def __init__(self):
        print("Загрузка Cellpose...")
        self.segmentor = cp_models.CellposeModel(gpu=config.USE_GPU_CELLPOSE, model_type='cyto')

    def process_directory(self):
        files = sorted(list(config.RAW_DATA_DIR.glob('*.jpg')) + list(config.RAW_DATA_DIR.glob('*.png')))
        if not files:
            print("ОШИБКА: Файлы не найдены!")
            return pd.DataFrame()

        print(f"Обработка {len(files)} файлов...")
        all_data = []

        for fpath in tqdm(files):
            try:
                meta = MetadataParser.parse(fpath.name)
                img = io.imread(str(fpath))

                # Сегментация
                masks, _, _ = self.segmentor.eval(img, diameter=None, channels=[0, 0])

                unique_cells = np.unique(masks)[1:]
                for cell_id in unique_cells:
                    mask_bool = (masks == cell_id)
                    area = np.sum(mask_bool)
                    if area < config.MIN_CELL_SIZE: continue

                    # Метрики
                    intensity = np.sum(img[mask_bool])
                    perimeter = cv2.arcLength(
                        cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0],
                        True)
                    circ = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

                    all_data.append({
                        **meta,
                        'filename': fpath.name,
                        'cell_id': cell_id,
                        'area': area,
                        'total_intensity': intensity,
                        'circularity': circ
                    })
            except Exception as e:
                print(f"Error {fpath.name}: {e}")
                continue

        return pd.DataFrame(all_data)


# ================= 2. SMART CLASSIFICATION (ANCHOR-BASED) =================

def classify_cells_smart(df):
    print("\n--- Смарт-классификация (Anchor-based) ---")
    if df.empty: return df

    # 1. Логарифмирование
    # Важно: используем natural log (np.log), тогда удвоение ДНК = +0.693
    df['log_int'] = np.log(df['total_intensity'])

    # 2. Поиск пика G1 на контроле (0 Gy)
    control_data = df[df['dose'] == 0]['log_int'].values

    # Строим гистограмму, чтобы найти моду (пик)
    hist, bin_edges = np.histogram(control_data, bins=100)
    peak_idx = np.argmax(hist)
    g1_peak_val = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2

    print(f"Найден пик G1 (log intensity): {g1_peak_val:.4f}")

    # 3. Расчет теоретических центров
    # G1 = peak
    # G2 = peak + ln(2) (так как ДНК удваивается)
    # S = где-то посередине (peak + ln(1.5))

    g1_center = g1_peak_val
    g2_center = g1_center + np.log(2.0)  # ~0.69
    s_center = (g1_center + g2_center) / 2

    print(f"Теоретические центры: G1={g1_center:.2f}, S={s_center:.2f}, G2={g2_center:.2f}")

    # 4. Инициализация GMM с подсказками
    # Мы говорим модели: "Начни искать кластеры ВОТ ЗДЕСЬ"
    X = df['log_int'].values.reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=3,
        covariance_type='full',
        means_init=np.array([[g1_center], [s_center], [g2_center]]),
        random_state=42
    )

    gmm.fit(X)

    # Проверяем, не перепутал ли GMM кластеры (сортируем по центрам)
    means = gmm.means_.flatten()
    sorted_idx = np.argsort(means)
    mapping = {sorted_idx[0]: 'G1', sorted_idx[1]: 'S', sorted_idx[2]: 'G2M'}

    # Предсказание
    labels = gmm.predict(X)
    df['phase'] = [mapping[l] for l in labels]

    # 5. Уточнение SubG1 (левее пика G1)
    # Все, что меньше (G1_peak - 2*sigma) или просто фиксированный отступ
    # По опыту: G1_peak - 0.5 в лог шкале отсекает осколки
    df.loc[df['log_int'] < (g1_peak_val - 0.4), 'phase'] = 'SubG1'

    # Уточнение Mitosis
    g2m_mask = (df['phase'] == 'G2M')
    df.loc[g2m_mask & (df['circularity'] > 0.88), 'phase'] = 'Mitosis'

    print("Распределение фаз:\n", df['phase'].value_counts())
    return df


# ================= 3. VISUALIZATION & VALIDATION =================

def generate_visualizations(df):
    print("\n--- Генерация графиков ---")
    sns.set_style("whitegrid")

    # A. Pseudo-Cytometry
    g = sns.FacetGrid(df, row="genotype", col="dose", hue="phase",
                      palette={"G1": "blue", "S": "purple", "G2M": "green", "SubG1": "orange", "Mitosis": "cyan"},
                      height=3, aspect=1.5, sharex=True)  # sharex=True для сравнения сдвигов
    g.map(sns.histplot, "log_int", bins=60, kde=True, element="step", alpha=0.6)
    g.add_legend()
    g.fig.suptitle("Corrected Pseudo-Flow Cytometry", y=1.02)
    plt.savefig(config.RESULTS_DIR / 'pseudo_cytometry_corrected.png', dpi=300, bbox_inches='tight')
    plt.close()

    # B. Dose Response
    df['is_G2_block'] = df['phase'].isin(['G2M', 'Mitosis'])
    g2_summary = df.groupby(['genotype', 'dose'])['is_G2_block'].mean().reset_index()
    g2_summary['is_G2_block'] *= 100

    plt.figure(figsize=(8, 6))
    sns.lineplot(data=g2_summary, x='dose', y='is_G2_block', hue='genotype', marker='o', linewidth=3)
    plt.title("G2/M Arrest (Corrected)")
    plt.ylabel("% Cells in G2/M")
    plt.savefig(config.RESULTS_DIR / 'dose_response_corrected.png', dpi=300)
    plt.close()

    # C. Stacked Bars
    props = df.groupby(['genotype', 'dose'])['phase'].value_counts(normalize=True).unstack(fill_value=0) * 100
    for col in ['SubG1', 'G1', 'S', 'G2M', 'Mitosis']:
        if col not in props.columns: props[col] = 0
    props = props[['SubG1', 'G1', 'S', 'G2M', 'Mitosis']]

    for geno in df['genotype'].unique():
        if geno not in props.index.get_level_values(0): continue
        subset = props.loc[geno]
        subset.plot(kind='bar', stacked=True, color=['orange', 'blue', 'purple', 'green', 'cyan'], figsize=(10, 6))
        plt.title(f"Distribution: {geno}")
        plt.tight_layout()
        plt.savefig(config.RESULTS_DIR / f'bars_{geno}.png')
        plt.close()


def validate(my_df):
    print("\n--- Валидация ---")
    my_summary = my_df.groupby(['genotype', 'dose'])['phase'].value_counts(normalize=True).reset_index(name='my_val')
    my_summary['my_val'] *= 100

    gt_df = pd.DataFrame(config.GROUND_TRUTH_DATA)
    merged = pd.merge(gt_df, my_summary, left_on=['Genotype', 'Dose', 'Phase'], right_on=['genotype', 'dose', 'phase'])

    if not merged.empty:
        mae = np.mean(np.abs(merged['Value'] - merged['my_val']))
        print(f"NEW MAE: {mae:.2f}% (Было 28%)")
        merged.to_csv(config.RESULTS_DIR / 'validation_v3.csv', index=False)

        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=merged, x='Value', y='my_val', hue='phase', s=100)
        plt.plot([0, 100], [0, 100], 'r--')
        plt.title(f"Accuracy (MAE={mae:.1f}%)")
        plt.savefig(config.RESULTS_DIR / 'accuracy_plot.png')


# ================= MAIN =================
def main():
    print("=== FINAL CELL ANALYSIS v3 (FIXED) ===")

    # 1. Processing
    processor = ImageProcessor()
    df = processor.process_directory()

    if df.empty: return

    # 2. Classification
    df = classify_cells_smart(df)

    # 3. Reports
    df.to_csv(config.RESULTS_DIR / 'final_data.csv', index=False)
    generate_visualizations(df)

    try:
        validate(df)
    except Exception as e:
        print(f"Validation skipped: {e}")

    print(f"\nDone! Results in: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
