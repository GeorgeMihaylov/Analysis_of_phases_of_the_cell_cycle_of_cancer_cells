# -*- coding: utf-8 -*-
"""
Анализ результатов Kelly dataset - загружает сохраненные данные и создает визуализации
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Константы
PHASES_3 = ["SubG1", "G1", "G2M"]
PHASE_COLORS = {"SubG1": "#f28e2b", "G1": "#4e79a7", "G2M": "#59a14f"}


def load_and_analyze_results(results_dir: Path):
    """Загружает и анализирует результаты"""
    print(f"Загружаем результаты из: {results_dir}")

    # Загружаем данные
    manifest_path = results_dir / "manifest_cells.csv"
    predictions_path = results_dir / "cell_predictions.csv"
    aggregated_path = results_dir / "aggregated_predictions.csv"
    ground_truth_path = results_dir / "ground_truth.csv"

    if not all(p.exists() for p in [manifest_path, predictions_path, aggregated_path, ground_truth_path]):
        print("Ошибка: не все файлы результатов найдены")
        return

    manifest = pd.read_csv(manifest_path)
    predictions = pd.read_csv(predictions_path)
    aggregated = pd.read_csv(aggregated_path)
    ground_truth = pd.read_csv(ground_truth_path)

    print(f"Загружено {len(manifest)} клеток")
    print(f"Условия: {len(aggregated)} уникальных комбинаций время-концентрация")

    # Создаем директорию для графиков
    figures_dir = results_dir / "analysis_figures"
    figures_dir.mkdir(exist_ok=True)

    # 1. Распределение клеток по условиям
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # По времени
    time_counts = manifest.groupby("time").size()
    axes[0].bar(time_counts.index, time_counts.values)
    axes[0].set_xlabel("Время (часы)")
    axes[0].set_ylabel("Количество клеток")
    axes[0].set_title("Распределение клеток по времени")
    axes[0].grid(True, alpha=0.3)

    # По концентрации
    conc_counts = manifest.groupby("concentration").size()
    axes[1].bar(conc_counts.index, conc_counts.values)
    axes[1].set_xlabel("Концентрация (µM)")
    axes[1].set_ylabel("Количество клеток")
    axes[1].set_title("Распределение клеток по концентрации")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "cell_distribution.png", dpi=150)
    plt.close()

    # 2. Распределение фаз по предсказаниям
    phase_counts = predictions["phase_pred"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(range(len(phase_counts)), phase_counts.values)
    ax.set_xticks(range(len(phase_counts)))
    ax.set_xticklabels(phase_counts.index)
    ax.set_xlabel("Фаза клеточного цикла")
    ax.set_ylabel("Количество клеток")
    ax.set_title("Распределение предсказанных фаз")

    # Добавляем проценты
    total = phase_counts.sum()
    for i, (phase, count) in enumerate(phase_counts.items()):
        percentage = 100 * count / total
        ax.text(i, count + total * 0.01, f"{percentage:.1f}%",
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(figures_dir / "phase_distribution.png", dpi=150)
    plt.close()

    # 3. Сравнение предсказаний с GT (scatter plots)
    # Переименовываем колонки GT для совместимости
    gt_renamed = ground_truth.rename(columns={
        "SubG1": "SubG1_gt",
        "G1": "G1_gt",
        "G2M": "G2M_gt"
    })

    # Объединяем
    merged = pd.merge(
        aggregated,
        gt_renamed,
        on=["time", "concentration", "treatment"],
        how="inner"
    )

    if not merged.empty:
        # Scatter plot для каждой фазы
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metrics = {}

        for idx, phase in enumerate(PHASES_3):
            ax = axes[idx]
            x = merged[f"{phase}_gt"].values
            y = merged[f"{phase}_pred"].values

            ax.scatter(x, y, alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
            ax.plot([0, 100], [0, 100], 'r--', alpha=0.5, label='Идеальное соответствие')
            ax.set_xlabel(f"Цитометрия {phase} (%)")
            ax.set_ylabel(f"Предсказано {phase} (%)")
            ax.set_title(phase)
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Вычисляем метрики
            if len(x) > 1:
                # Корреляция
                r = np.corrcoef(x, y)[0, 1]
                # MAE
                mae = np.mean(np.abs(x - y))
                # RMSE
                rmse = np.sqrt(np.mean((x - y) ** 2))

                metrics[phase] = {
                    "correlation": r,
                    "mae": mae,
                    "rmse": rmse
                }

                # Добавляем текст с метриками
                text = f"r = {r:.3f}\nMAE = {mae:.1f}%\nRMSE = {rmse:.1f}%"
                ax.text(0.05, 0.95, text, transform=ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.savefig(figures_dir / "predictions_vs_gt_scatter.png", dpi=150)
        plt.close()

        # 4. Графики по времени и концентрации
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))

        times = sorted(merged["time"].unique())

        for idx, phase in enumerate(PHASES_3):
            ax = axes[idx]

            for time_val in times:
                time_data = merged[merged["time"] == time_val]
                if len(time_data) > 0:
                    # Сортируем по концентрации
                    time_data = time_data.sort_values("concentration")

                    concs = time_data["concentration"].values
                    preds = time_data[f"{phase}_pred"].values
                    gts = time_data[f"{phase}_gt"].values

                    # Предсказания
                    ax.plot(concs, preds, 'o-', linewidth=2, markersize=8,
                            label=f"{time_val}h - предсказано", alpha=0.8)
                    # Цитометрия
                    ax.plot(concs, gts, 's--', linewidth=1.5, markersize=6,
                            label=f"{time_val}h - цитометрия", alpha=0.8)

            ax.set_ylabel(f"{phase} (%)", fontsize=12)
            ax.set_title(f"Фаза {phase} - зависимость от концентрации", fontsize=14)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)

        axes[-1].set_xlabel("Концентрация ауранофина (µM)", fontsize=12)
        plt.tight_layout()
        plt.savefig(figures_dir / "concentration_response.png", dpi=150)
        plt.close()

        # 5. Тепловая карта ошибок
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, phase in enumerate(PHASES_3):
            ax = axes[idx]

            # Создаем матрицу для тепловой карты
            times = sorted(merged["time"].unique())
            concs = sorted(merged["concentration"].unique())

            error_matrix = np.zeros((len(times), len(concs)))

            for i, time_val in enumerate(times):
                for j, conc_val in enumerate(concs):
                    mask = (merged["time"] == time_val) & (merged["concentration"] == conc_val)
                    if mask.any():
                        pred = merged.loc[mask, f"{phase}_pred"].values[0]
                        gt = merged.loc[mask, f"{phase}_gt"].values[0]
                        error_matrix[i, j] = abs(pred - gt)

            im = ax.imshow(error_matrix, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(len(concs)))
            ax.set_xticklabels([f"{c}µM" for c in concs])
            ax.set_yticks(range(len(times)))
            ax.set_yticklabels([f"{t}ч" for t in times])
            ax.set_xlabel("Концентрация")
            ax.set_ylabel("Время")
            ax.set_title(f"Абсолютная ошибка - {phase}")

            # Добавляем значения в ячейки
            for i in range(len(times)):
                for j in range(len(concs)):
                    text = ax.text(j, i, f"{error_matrix[i, j]:.1f}",
                                   ha="center", va="center", color="black",
                                   fontweight='bold' if error_matrix[i, j] > 15 else 'normal')

            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(figures_dir / "error_heatmap.png", dpi=150)
        plt.close()

        # Сохраняем метрики
        metrics_df = pd.DataFrame({
            phase: [metrics[phase]["correlation"], metrics[phase]["mae"], metrics[phase]["rmse"]]
            for phase in PHASES_3
        }, index=["Correlation", "MAE (%)", "RMSE (%)"])

        metrics_df.to_csv(figures_dir / "metrics.csv")

        # Средние метрики
        avg_mae = np.mean([metrics[p]["mae"] for p in PHASES_3])
        avg_rmse = np.mean([metrics[p]["rmse"] for p in PHASES_3])

        print("\n=== МЕТРИКИ ПРЕДСКАЗАНИЙ ===")
        print(metrics_df)
        print(f"\nСредняя MAE: {avg_mae:.2f}%")
        print(f"Средняя RMSE: {avg_rmse:.2f}%")

        # 6. Дополнительный анализ: SubG1 как маркер апоптоза
        print("\n=== АНАЛИЗ SubG1 (АПОПТОЗ) ===")
        subg1_data = merged[["time", "concentration", "SubG1_pred", "SubG1_gt"]].copy()
        subg1_data["error"] = abs(subg1_data["SubG1_pred"] - subg1_data["SubG1_gt"])

        print("SubG1 по условиям:")
        print(subg1_data.sort_values(["time", "concentration"]).to_string(index=False))

        # График SubG1
        fig, ax = plt.subplots(figsize=(10, 6))

        for time_val in sorted(subg1_data["time"].unique()):
            time_subset = subg1_data[subg1_data["time"] == time_val].sort_values("concentration")
            ax.plot(time_subset["concentration"], time_subset["SubG1_pred"],
                    'o-', label=f"{time_val}h - предсказано", linewidth=2, markersize=8)
            ax.plot(time_subset["concentration"], time_subset["SubG1_gt"],
                    's--', label=f"{time_val}h - цитометрия", linewidth=1.5, markersize=6)

        ax.set_xlabel("Концентрация ауранофина (µM)", fontsize=12)
        ax.set_ylabel("SubG1 (%)", fontsize=12)
        ax.set_title("SubG1 (апоптоз) - зависимость от концентрации", fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(figures_dir / "subg1_analysis.png", dpi=150)
        plt.close()

    else:
        print("Нет совпадающих условий для сравнения с цитометрией")

    # 7. Сводный отчет
    report = f"""
    ОТЧЕТ ОБ АНАЛИЗЕ KELLY DATASET
    ================================

    ОБЩАЯ ИНФОРМАЦИЯ:
    - Всего клеток: {len(manifest)}
    - Условий: {len(aggregated)}
    - Файлов изображений: {manifest['filename'].nunique()}

    РАСПРЕДЕЛЕНИЕ КЛЕТОК:
    {manifest.groupby('time').size().to_string()}

    ПРЕДСКАЗАННЫЕ ФАЗЫ:
    {predictions['phase_pred'].value_counts().to_string()}

    КЛЮЧЕВЫЕ НАБЛЮДЕНИЯ:
    1. Обратите внимание на низкое количество клеток для 6ч, 2µM (всего 1 клетка)
    2. Это может объяснить низкую точность для этого условия
    3. SubG1 (маркер апоптоза) увеличивается с концентрацией и временем

    РЕКОМЕНДАЦИИ:
    1. Повторить эксперимент для 6ч, 2µM для получения большего количества клеток
    2. Рассмотреть возможность исключения этого условия из анализа
    3. Проверить качество сегментации для изображений с малым количеством клеток
    """

    with open(figures_dir / "summary_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print("Анализ завершен!")
    print(f"Все графики сохранены в: {figures_dir}")
    print("=" * 60)

    return merged, metrics


if __name__ == "__main__":
    # Укажите путь к результатам
    results_dir = Path(
        r"D:\PycharmProjects\Analysis_of_phases_of_the_cell_cycle_of_cancer_cells\results\kelly_simple_20260206_143727")

    if results_dir.exists():
        merged, metrics = load_and_analyze_results(results_dir)
    else:
        print(f"Директория не найдена: {results_dir}")
        print("Доступные директории:")
        base_dir = Path(r"D:\PycharmProjects\Analysis_of_phases_of_the_cell_cycle_of_cancer_cells\results")
        for d in base_dir.iterdir():
            if d.is_dir() and "kelly" in d.name.lower():
                print(f"  - {d.name}")