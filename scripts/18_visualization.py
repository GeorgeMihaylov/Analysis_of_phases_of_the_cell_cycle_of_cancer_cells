import sys
import os
import re
import json
import logging
import numpy as np
import pandas as pd
import cv2
import torch  # <--- Добавлен пропущенный импорт
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from glob import glob
from cellpose import models as cp_models, io
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()


# ================= КОНФИГУРАЦИЯ =================
class VisConfig:
    # Пути (автоматически ищем корень проекта)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'

    # Укажите здесь конкретную папку с результатами, если нужно
    # Если None, скрипт возьмет самую свежую папку final_analysis_*
    TARGET_RESULT_DIR = None

    # Цветовая схема для фаз
    PHASE_COLORS = {
        'G1': (0, 0, 255),  # Синий
        'S': (255, 255, 0),  # Желтый
        'G2M': (0, 255, 0),  # Зеленый
        'SubG1': (255, 0, 0),  # Красный
        'Mitosis': (255, 0, 255)  # Маджента
    }

    # Matplotlib цвета (нормализованные 0-1)
    MPL_COLORS = {
        'G1': 'tab:blue',
        'G2M': 'tab:green',
        'SubG1': 'tab:red',
        'S': 'tab:orange',
        'Unknown': 'gray'
    }

    # Ground Truth Data
    GT_DATA = [
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


config = VisConfig()


# ================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =================
def find_latest_results():
    if config.TARGET_RESULT_DIR:
        return Path(config.TARGET_RESULT_DIR)

    # Ищем в папке results
    results_path = config.PROJECT_ROOT / 'results'
    candidates = sorted(list(results_path.glob('final_analysis_*')))

    if not candidates:
        raise FileNotFoundError("Папки с результатами не найдены!")

    latest = candidates[-1]
    logger.info(f"Выбрана папка результатов: {latest}")
    return latest


def load_data(result_dir):
    csv_path = result_dir / 'final_classified_cells.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Файл {csv_path} не найден. Сначала запустите анализ.")

    df = pd.read_csv(csv_path)
    return df


# ================= ВИЗУАЛИЗАЦИЯ 1: СРАВНЕНИЕ С GT =================
def plot_comparison_bar(df, output_dir):
    logger.info("Построение сравнительных графиков...")

    # Агрегация предсказанных данных
    pred_agg = df.groupby(['genotype', 'dose', 'phase']).size().unstack(fill_value=0)
    pred_agg = pred_agg.div(pred_agg.sum(axis=1), axis=0) * 100
    pred_agg = pred_agg.reset_index().melt(id_vars=['genotype', 'dose'], var_name='Phase', value_name='Value')
    pred_agg['Source'] = 'Predicted'

    # Подготовка GT данных
    gt_df = pd.DataFrame(config.GT_DATA)
    gt_df.columns = [c.lower() for c in gt_df.columns]
    gt_df.rename(columns={'value': 'Value', 'phase': 'Phase'}, inplace=True)
    gt_df['Source'] = 'Ground Truth'

    # Объединение
    combined = pd.concat([pred_agg, gt_df], ignore_index=True)

    # Фильтр только общих доз
    combined = combined[combined['dose'].isin([0, 4, 10])]

    # Построение графика
    g = sns.catplot(
        data=combined, kind="bar",
        x="dose", y="Value", hue="Source", col="genotype", row="Phase",
        palette="muted", height=3, aspect=1.5,
        sharex=False
    )
    g.set_axis_labels("Dose (Gy)", "Cells (%)")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Comparison: Predicted vs Flow Cytometry (Ground Truth)')

    save_path = output_dir / 'comparison_bar_chart.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Сохранен график сравнения: {save_path}")


# ================= ВИЗУАЛИЗАЦИЯ 2: ВИЗУАЛЬНАЯ СЕГМЕНТАЦИЯ =================
def visualize_segmentation_overlay(df, result_dir):
    """
    Берет несколько примеров изображений, прогоняет через Cellpose (чтобы получить контуры)
    и раскрашивает клетки в соответствии с предсказанной фазой из CSV.
    """
    logger.info("Генерация визуальных карт сегментации (Overlay)...")

    vis_dir = result_dir / 'segmentation_overlays'
    vis_dir.mkdir(exist_ok=True)

    # Загружаем модель Cellpose
    use_gpu = torch.cuda.is_available()
    model = cp_models.CellposeModel(gpu=use_gpu, pretrained_model='cpsam')
    logger.info(f"Cellpose загружен (GPU={use_gpu})")

    # Выбираем по 1 примеру для каждой комбинации Генотип/Доза
    unique_conditions = df[['genotype', 'dose']].drop_duplicates()

    for _, row in unique_conditions.iterrows():
        geno, dose = row['genotype'], row['dose']

        # Находим файл
        sample_files = df[(df['genotype'] == geno) & (df['dose'] == dose)]['filename'].unique()
        if len(sample_files) == 0: continue

        target_file = sample_files[0]
        img_path = config.RAW_DATA_DIR / target_file

        if not img_path.exists():
            logger.warning(f"Файл {target_file} не найден в {config.RAW_DATA_DIR}")
            continue

        logger.info(f"Обработка {target_file}...")

        # 1. Загрузка и сегментация
        img = io.imread(str(img_path))
        if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Получаем маски заново
        masks, _, _ = model.eval(img, diameter=None, channels=None)

        # 2. Подготовка цветной маски
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Данные по этому файлу из CSV
        file_data = df[df['filename'] == target_file].set_index('cell_id')

        unique_cells = np.unique(masks)[1:]

        for cell_id in unique_cells:
            if cell_id not in file_data.index: continue

            phase = file_data.loc[cell_id, 'phase']
            color = config.PHASE_COLORS.get(phase, (128, 128, 128))  # RGB

            # Маска для одной клетки
            cell_mask = (masks == cell_id).astype(np.uint8)

            # Контуры
            contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Рисуем
            bgr_color = (color[2], color[1], color[0])
            cv2.drawContours(img_rgb, contours, -1, bgr_color, 1)

        # Добавляем легенду на изображение
        legend_y = 30
        for phase, color in config.PHASE_COLORS.items():
            bgr_color = (color[2], color[1], color[0])
            cv2.rectangle(img_rgb, (10, legend_y - 15), (30, legend_y + 5), bgr_color, -1)
            cv2.putText(img_rgb, phase, (40, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            legend_y += 30

        # Сохранение
        save_name = vis_dir / f"overlay_{geno}_{dose}Gy_{target_file}"
        cv2.imwrite(str(save_name), img_rgb)

    logger.info(f"Визуализации сохранены в {vis_dir}")


# ================= ВИЗУАЛИЗАЦИЯ 3: РАСПРЕДЕЛЕНИЕ ПРИЗНАКОВ =================
def plot_feature_violins(df, output_dir):
    logger.info("Построение распределения признаков (Violin Plots)...")

    features_to_plot = ['intensity_mean', 'area', 'g2m_score', 'subg1_score']
    plot_df = df.copy()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i, feature in enumerate(features_to_plot):
        if feature not in df.columns: continue

        q_low = plot_df[feature].quantile(0.01)
        q_high = plot_df[feature].quantile(0.99)
        feat_df = plot_df[(plot_df[feature] >= q_low) & (plot_df[feature] <= q_high)]

        # Исправленный вызов Seaborn
        sns.violinplot(
            data=feat_df, x="phase", y=feature, hue="phase",
            order=['SubG1', 'G1', 'G2M'],
            palette=config.MPL_COLORS, ax=axes[i], legend=False
        )
        axes[i].set_title(f"Distribution of {feature} by Phase")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'features_by_phase.png', dpi=300)
    plt.close()


# ================= MAIN =================
def main():
    try:
        # 1. Находим и загружаем данные
        res_dir = find_latest_results()
        df = load_data(res_dir)

        logger.info(f"Загружено {len(df)} клеток.")

        # 2. Графики сравнения с GT
        plot_comparison_bar(df, res_dir)

        # 3. Violin plots
        plot_feature_violins(df, res_dir)

        # 4. Визуализация на изображениях
        visualize_segmentation_overlay(df, res_dir)

        logger.info("=== Визуализация завершена успешно! ===")
        print(f"\nГрафики лежат в папке: {res_dir}")
        print(f"Примеры с наложением масок: {res_dir}/segmentation_overlays")

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)


if __name__ == "__main__":
    main()
