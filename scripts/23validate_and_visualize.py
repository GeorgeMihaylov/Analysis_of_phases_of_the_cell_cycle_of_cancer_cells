# scripts/05validate_and_visualize.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def plot_validation_results(df_results, output_path):
    # 1. Агрегация предсказаний
    # Группируем по (Genotype, Dose) и считаем % каждой фазы
    preds = df_results.groupby(['Genotype', 'Dose'])['pred_phase'].value_counts(normalize=True).unstack().fillna(
        0) * 100
    preds = preds[['SubG1', 'G1', 'G2M']]  # Порядок

    # 2. Подготовка Ground Truth данных (из вашего задания)
    gt_data = []
    # WT
    gt_data.append({'Genotype': 'WT', 'Dose': 0, 'SubG1': 6.9, 'G1': 61.4, 'G2M': 20.4})
    gt_data.append({'Genotype': 'WT', 'Dose': 4, 'SubG1': 22.9, 'G1': 26.4, 'G2M': 45.9})
    gt_data.append({'Genotype': 'WT', 'Dose': 10, 'SubG1': 36.3, 'G1': 20.1, 'G2M': 36.1})
    # CDK8KO
    gt_data.append({'Genotype': 'CDK8KO', 'Dose': 0, 'SubG1': 9.0, 'G1': 63.4, 'G2M': 16.4})
    gt_data.append({'Genotype': 'CDK8KO', 'Dose': 4, 'SubG1': 35.4, 'G1': 21.8, 'G2M': 33.2})
    gt_data.append({'Genotype': 'CDK8KO', 'Dose': 10, 'SubG1': 48.1, 'G1': 23.2, 'G2M': 19.4})

    df_gt = pd.DataFrame(gt_data).set_index(['Genotype', 'Dose'])

    # 3. Построение графика
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # Цветовая палитра
    colors = {'SubG1': '#d62728', 'G1': '#1f77b4', 'G2M': '#2ca02c'}

    # Функция для отрисовки стеков
    def plot_stacked(ax, df, title):
        bottom = np.zeros(len(df))
        doses = df.index.get_level_values('Dose').astype(str) + ' Gy'
        genotypes = df.index.get_level_values('Genotype')
        labels = [f"{g}\n{d}" for g, d in zip(genotypes, doses)]

        for phase in ['SubG1', 'G1', 'G2M']:
            values = df[phase].values
            ax.bar(np.arange(len(df)), values, bottom=bottom, label=phase, color=colors[phase], alpha=0.9, width=0.6)
            bottom += values

        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_title(title)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    # График Ground Truth
    plot_stacked(axes[0], df_gt, "Flow Cytometry (Ground Truth)")
    axes[0].set_ylabel("% Cells")

    # График Predictions
    # Выравниваем индексы предсказаний под GT
    preds_aligned = preds.reindex(df_gt.index).fillna(0)
    plot_stacked(axes[1], preds_aligned, "Microscopy Predictions (Your Model)")

    # Легенда
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Comparison chart saved to {output_path}")

# Пример вызова
# plot_validation_results(final_df, 'results/validation_comparison.png')
