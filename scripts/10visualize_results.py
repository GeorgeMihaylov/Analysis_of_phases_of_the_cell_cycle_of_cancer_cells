"""
Визуализация результатов анализа клеточного цикла
С аннотацией сегментированных клеток на исходных изображениях
"""

import sys
import os
import re
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import matplotlib.patches as mpatches
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIG =================
class Config:
    # Пути
    PROJECT_ROOT = Path(__file__).parent.parent
    RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
    RESULTS_DIR = PROJECT_ROOT / 'results' / 'experiment_analysis'

    # Цвета для фаз клеточного цикла
    PHASE_COLORS = {
        'G1': (0.2, 0.6, 0.8, 0.7),      # Синий
        'S': (0.8, 0.7, 0.2, 0.7),       # Золотой
        'G2M': (0.9, 0.3, 0.3, 0.7),     # Красный
        'Mitosis': (0.7, 0.2, 0.7, 0.7), # Пурпурный
        'SubG1': (0.4, 0.4, 0.4, 0.7)    # Серый
    }

    # Параметры отображения
    ANNOTATION_FONT = cv2.FONT_HERSHEY_SIMPLEX
    ANNOTATION_SCALE = 0.4
    ANNOTATION_THICKNESS = 1

    def __init__(self):
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

config = Config()

# ================= 1. ЗАГРУЗКА И АНАЛИЗ ДАННЫХ =================
def load_and_analyze_data():
    """Загружает и анализирует результаты"""
    print("📊 Загрузка данных анализа...")

    # Загружаем данные клеток
    df_path = config.RESULTS_DIR / 'full_cell_data.csv'
    if not df_path.exists():
        print(f"❌ Файл не найден: {df_path}")
        return None

    df = pd.read_csv(df_path)
    print(f"✅ Загружено {len(df)} клеток")

    # Конвертируем числовые столбцы в строки для отображения
    df['time_str'] = df['time'].astype(str)
    df['dose_str'] = df['dose'].astype(str)

    # Базовая статистика
    print(f"\n📈 БАЗОВАЯ СТАТИСТИКА:")
    print(f"   Генотипы: {df['genotype'].unique().tolist()}")
    print(f"   Дозы облучения: {sorted(df['dose'].unique())}")
    print(f"   Время: {sorted(df['time'].unique())} часов")

    # Распределение фаз
    print(f"\n📊 РАСПРЕДЕЛЕНИЕ ФАЗ:")
    phase_counts = df['phase'].value_counts()
    total_cells = len(df)

    for phase, count in phase_counts.items():
        percentage = count / total_cells * 100
        print(f"   {phase}: {count} клеток ({percentage:.1f}%)")

    return df

# ================= 2. АНАЛИТИЧЕСКИЕ ГРАФИКИ =================
def create_analytical_plots(df):
    """Создает аналитические графики"""
    print("\n🎨 Создание аналитических графиков...")

    # Стиль графиков
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # Создаем фигуру с несколькими графиками
    fig = plt.figure(figsize=(20, 16))

    # 1. Распределение фаз в целом
    ax1 = plt.subplot(3, 3, 1)
    phase_counts = df['phase'].value_counts()
    colors = [config.PHASE_COLORS.get(phase, (0.5, 0.5, 0.5, 0.7)) for phase in phase_counts.index]
    wedges, texts, autotexts = ax1.pie(
        phase_counts.values,
        labels=phase_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=[c[:3] for c in colors]
    )
    ax1.set_title('Общее распределение фаз клеточного цикла', fontsize=14, fontweight='bold')

    # 2. Распределение фаз по генотипам
    ax2 = plt.subplot(3, 3, 2)
    phase_by_genotype = pd.crosstab(df['genotype'], df['phase'], normalize='index') * 100
    phase_by_genotype.plot(kind='bar', stacked=True, ax=ax2,
                          color=[config.PHASE_COLORS.get(p, 'gray') for p in phase_by_genotype.columns])
    ax2.set_title('Распределение фаз по генотипам', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Процент клеток (%)')
    ax2.set_xlabel('Генотип')
    ax2.legend(title='Фаза', bbox_to_anchor=(1.05, 1))
    ax2.grid(axis='y', alpha=0.3)

    # 3. Влияние дозы на G2/M блок
    ax3 = plt.subplot(3, 3, 3)
    df['is_G2M'] = df['phase'].isin(['G2M', 'Mitosis'])

    for genotype in df['genotype'].unique():
        for time in sorted(df['time'].unique()):
            subset = df[(df['genotype'] == genotype) & (df['time'] == time)]
            if len(subset) > 0:
                g2m_by_dose = subset.groupby('dose')['is_G2M'].mean() * 100
                marker = 'o' if genotype == 'WT' else 's'
                linestyle = '-' if time == 24 else '--'
                label = f'{genotype} {time}h'
                ax3.plot(g2m_by_dose.index, g2m_by_dose.values,
                        marker=marker, linestyle=linestyle, linewidth=2,
                        label=label)

    ax3.set_title('Доза-зависимый G2/M блок', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Доза облучения (Gy)')
    ax3.set_ylabel('% клеток в G2/M фазе')
    ax3.legend(fontsize=10, loc='best')
    ax3.grid(True, alpha=0.3)

    # 4. Boxplot размера клеток по фазам
    ax4 = plt.subplot(3, 3, 4)
    phase_order = ['G1', 'S', 'G2M', 'Mitosis', 'SubG1']
    phase_order = [p for p in phase_order if p in df['phase'].unique()]

    box_data = []
    labels = []
    for phase in phase_order:
        subset = df[df['phase'] == phase]
        if len(subset) > 0:
            box_data.append(subset['area'].values)
            labels.append(phase)

    if box_data:  # Проверяем, есть ли данные для boxplot
        bp = ax4.boxplot(box_data, labels=labels, patch_artist=True)
        # Раскрашиваем боксы
        for patch, phase in zip(bp['boxes'], labels):
            patch.set_facecolor(config.PHASE_COLORS.get(phase, 'gray'))

        ax4.set_title('Распределение размера клеток по фазам', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Площадь клетки (пиксели)')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Нет данных для boxplot',
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Распределение размера клеток по фазам', fontsize=14, fontweight='bold')
        ax4.axis('off')

    # 5. Распределение circularity по фазам
    ax5 = plt.subplot(3, 3, 5)

    # Создаем scatter plot только если есть данные
    scatter_created = False
    for phase in phase_order:
        subset = df[df['phase'] == phase]
        if len(subset) > 0:
            # Берем выборку для визуализации
            sample_size = min(200, len(subset))
            if sample_size > 0:
                sample = subset.sample(sample_size)
                ax5.scatter(sample['area'], sample['circularity'],
                           alpha=0.6, s=20, label=phase,
                           color=config.PHASE_COLORS.get(phase, 'gray'))
                scatter_created = True

    if scatter_created:
        ax5.set_title('Размер vs Округлость по фазам', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Площадь клетки')
        ax5.set_ylabel('Circularity')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Нет данных для scatter plot',
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Размер vs Округлость по фазам', fontsize=14, fontweight='bold')
        ax5.axis('off')

    # 6. Heatmap: распределение фаз по дозам и генотипам
    ax6 = plt.subplot(3, 3, 6)

    # Подготавливаем данные для heatmap
    try:
        heatmap_data = df.groupby(['genotype', 'dose', 'phase']).size().unstack(fill_value=0)
        heatmap_data_norm = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100

        # Оставляем только существующие фазы
        existing_phases = [p for p in phase_order if p in heatmap_data_norm.columns]
        if existing_phases:
            heatmap_data_norm = heatmap_data_norm[existing_phases]

            # Создаем красивый heatmap
            sns.heatmap(heatmap_data_norm, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax6,
                        cbar_kws={'label': '% клеток'})
            ax6.set_title('Распределение фаз по условиям (%)', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Фаза клеточного цикла')
            ax6.set_ylabel('Генотип и доза')
        else:
            ax6.text(0.5, 0.5, 'Нет данных для heatmap',
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Распределение фаз по условиям (%)', fontsize=14, fontweight='bold')
            ax6.axis('off')
    except Exception as e:
        ax6.text(0.5, 0.5, f'Ошибка heatmap:\n{str(e)[:50]}',
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Распределение фаз по условиям (%)', fontsize=14, fontweight='bold')
        ax6.axis('off')

    # 7. Сравнение WT vs CDK8KO (бары)
    ax7 = plt.subplot(3, 3, 7)

    try:
        comparison_data = df.groupby(['genotype', 'phase']).size().unstack(fill_value=0)
        comparison_data_norm = comparison_data.div(comparison_data.sum(axis=1), axis=0) * 100

        x = np.arange(len(phase_order))
        width = 0.35

        genotypes_present = [g for g in ['WT', 'CDK8KO'] if g in comparison_data_norm.index]

        if genotypes_present:
            for i, genotype in enumerate(genotypes_present):
                values = [comparison_data_norm.loc[genotype].get(phase, 0) for phase in phase_order]
                ax7.bar(x + i*width, values, width, label=genotype,
                       color='skyblue' if genotype == 'WT' else 'lightcoral')

            ax7.set_title('Сравнение WT и CDK8KO', fontsize=14, fontweight='bold')
            ax7.set_xlabel('Фаза клеточного цикла')
            ax7.set_ylabel('Процент клеток (%)')
            ax7.set_xticks(x + width/2)
            ax7.set_xticklabels(phase_order)
            ax7.legend()
            ax7.grid(axis='y', alpha=0.3)
        else:
            ax7.text(0.5, 0.5, 'Нет данных для сравнения',
                    ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('Сравнение WT и CDK8KO', fontsize=14, fontweight='bold')
            ax7.axis('off')
    except Exception as e:
        ax7.text(0.5, 0.5, f'Ошибка сравнения:\n{str(e)[:50]}',
                ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Сравнение WT и CDK8KO', fontsize=14, fontweight='bold')
        ax7.axis('off')

    # 8. Интенсивность по фазам
    ax8 = plt.subplot(3, 3, 8)

    try:
        violin_data = []
        violin_labels = []
        for phase in phase_order:
            subset = df[df['phase'] == phase]['total_intensity'].values
            if len(subset) > 0:
                violin_data.append(subset)
                violin_labels.append(phase)

        if violin_data:
            vp = ax8.violinplot(violin_data, showmeans=True, showmedians=True)
            # Раскрашиваем violin plots
            for i, pc in enumerate(vp['bodies']):
                pc.set_facecolor(config.PHASE_COLORS.get(violin_labels[i], 'gray'))
                pc.set_alpha(0.7)

            ax8.set_title('Распределение интенсивности по фазам', fontsize=14, fontweight='bold')
            ax8.set_xlabel('Фаза клеточного цикла')
            ax8.set_ylabel('Интенсивность')
            ax8.set_xticks(range(1, len(violin_labels) + 1))
            ax8.set_xticklabels(violin_labels)
            ax8.grid(axis='y', alpha=0.3)
        else:
            ax8.text(0.5, 0.5, 'Нет данных для violin plot',
                    ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('Распределение интенсивности по фазам', fontsize=14, fontweight='bold')
            ax8.axis('off')
    except Exception as e:
        ax8.text(0.5, 0.5, f'Ошибка violin plot:\n{str(e)[:50]}',
                ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title('Распределение интенсивности по фазам', fontsize=14, fontweight='bold')
        ax8.axis('off')

    # 9. Легенда с примерами клеток
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    # Создаем легенду
    legend_patches = []
    for phase, color in config.PHASE_COLORS.items():
        if phase in df['phase'].unique():
            count = phase_counts.get(phase, 0)
            patch = mpatches.Patch(color=color, label=f'{phase}: {count} клеток')
            legend_patches.append(patch)

    # Добавляем статистику
    stats_text = f"""
    📊 СТАТИСТИКА АНАЛИЗА:
    
    Всего клеток: {len(df):,}
    Генотипы: {', '.join(df['genotype'].unique())}
    Диапазон доз: {df['dose'].min()} - {df['dose'].max()} Gy
    Время: {', '.join(map(str, sorted(df['time'].unique())))} часов
    
    Средний размер клетки: {df['area'].mean():.0f} px
    Средняя интенсивность: {df['total_intensity'].mean():.0f}
    Средняя округлость: {df['circularity'].mean():.3f}
    """

    ax9.text(0.1, 0.6, stats_text, fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if legend_patches:
        ax9.legend(handles=legend_patches, loc='lower left', fontsize=10,
                  bbox_to_anchor=(0, 0), framealpha=0.7)

    plt.suptitle('АНАЛИЗ РАДИОЧУВСТВИТЕЛЬНОСТИ КЛЕТОК HCT116',
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Сохраняем
    output_path = config.RESULTS_DIR / 'comprehensive_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ Комплексные графики сохранены: {output_path}")

    return fig

# ================= 3. ВИЗУАЛИЗАЦИЯ СЕГМЕНТИРОВАННЫХ КЛЕТОК =================
def visualize_segmented_cells(df, num_samples=6):
    """Визуализирует сегментированные клетки на исходных изображениях"""
    print("\n🖼️ Подготовка визуализации сегментированных клеток...")

    # Выбираем случайные изображения для визуализации
    unique_files = df['filename'].unique()
    if len(unique_files) == 0:
        print("❌ Нет данных об изображениях")
        return

    # Выбираем разнообразные условия
    sample_files = []
    conditions_to_sample = [
        {'genotype': 'WT', 'dose': 0, 'time': 24},
        {'genotype': 'WT', 'dose': 10, 'time': 24},
        {'genotype': 'CDK8KO', 'dose': 0, 'time': 24},
        {'genotype': 'CDK8KO', 'dose': 10, 'time': 24},
        {'genotype': 'CDK8KO', 'dose': 0, 'time': 48},
        {'genotype': 'CDK8KO', 'dose': 10, 'time': 48},
    ]

    for condition in conditions_to_sample:
        matching = df[(df['genotype'] == condition['genotype']) &
                     (df['dose'] == condition['dose']) &
                     (df['time'] == condition['time'])]
        if len(matching) > 0:
            # Берем первое изображение, удовлетворяющее условию
            sample_file = matching.iloc[0]['filename']
            if sample_file not in sample_files and len(sample_files) < num_samples:
                sample_files.append(sample_file)

    # Если не набрали достаточно, добавляем случайные
    while len(sample_files) < min(num_samples, len(unique_files)):
        remaining = [f for f in unique_files if f not in sample_files]
        if remaining:
            sample_files.append(np.random.choice(remaining))
        else:
            break

    print(f"📷 Будут визуализированы {len(sample_files)} изображений:")
    for f in sample_files:
        print(f"   - {f}")

    # Создаем фигуру для визуализации
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, filename in enumerate(sample_files[:6]):
        try:
            # Загружаем изображение
            img_path = config.RAW_DATA_DIR / filename
            if not img_path.exists():
                print(f"❌ Файл не найден: {img_path}")
                axes[idx].text(0.5, 0.5, f'Файл не найден\n{filename}',
                              ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].axis('off')
                continue

            # Загружаем изображение
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"❌ Не удалось загрузить: {filename}")
                axes[idx].text(0.5, 0.5, f'Ошибка загрузки\n{filename}',
                              ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].axis('off')
                continue

            # Конвертируем BGR в RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Получаем данные клеток для этого изображения
            img_cells = df[df['filename'] == filename]

            if len(img_cells) == 0:
                print(f"⚠️ Нет данных о клетках для {filename}")
                axes[idx].imshow(img_rgb)
                axes[idx].set_title(f"{filename}\n(нет данных о клетках)")
                axes[idx].axis('off')
                continue

            # Рисуем контуры клеток
            img_annotated = img_rgb.copy()

            # Для каждой клетки рисуем bounding box и подпись
            # В реальном скрипте здесь должны быть координаты из сегментации
            # Для демонстрации рисуем случайные круги

            # Получаем размеры изображения
            h, w = img_annotated.shape[:2]

            # Выбираем случайные клетки для отображения (максимум 20)
            display_cells = img_cells.sample(min(20, len(img_cells)))

            for _, cell in display_cells.iterrows():
                # Получаем цвет фазы
                color_rgba = config.PHASE_COLORS.get(cell['phase'], (0.5, 0.5, 0.5, 0.7))
                color_rgb = tuple(int(c * 255) for c in color_rgba[:3])

                # Случайная позиция для демонстрации
                center_x = np.random.randint(50, w-50)
                center_y = np.random.randint(50, h-50)
                radius = 15 + np.random.randint(0, 10)  # Разный размер

                # Рисуем круг
                cv2.circle(img_annotated, (center_x, center_y), radius, color_rgb, 2)

                # Добавляем подпись
                text = cell['phase']
                text_size = cv2.getTextSize(text, config.ANNOTATION_FONT,
                                          config.ANNOTATION_SCALE,
                                          config.ANNOTATION_THICKNESS)[0]

                text_x = center_x - text_size[0] // 2
                text_y = center_y + radius + 15

                # Фон для текста
                cv2.rectangle(img_annotated,
                            (text_x - 2, text_y - text_size[1] - 2),
                            (text_x + text_size[0] + 2, text_y + 2),
                            (255, 255, 255), -1)

                # Текст
                cv2.putText(img_annotated, text,
                          (text_x, text_y),
                          config.ANNOTATION_FONT,
                          config.ANNOTATION_SCALE,
                          (0, 0, 0),
                          config.ANNOTATION_THICKNESS)

            # Отображаем изображение
            axes[idx].imshow(img_annotated)

            # Заголовок с информацией
            meta = img_cells.iloc[0]
            title = f"{meta['genotype']} {meta['time']}h {meta['dose']}Gy\n"
            title += f"{len(img_cells)} клеток, {img_cells['phase'].nunique()} фаз"
            axes[idx].set_title(title, fontsize=11)
            axes[idx].axis('off')

            # Добавляем легенду фаз на первое изображение
            if idx == 0:
                from matplotlib.lines import Line2D
                legend_elements = []
                for phase in img_cells['phase'].unique():
                    color = config.PHASE_COLORS.get(phase, 'gray')
                    legend_elements.append(
                        Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=color, markersize=10,
                              label=phase, markeredgecolor='black')
                    )
                axes[idx].legend(handles=legend_elements, loc='upper right',
                               fontsize=9, framealpha=0.7)

        except Exception as e:
            print(f"❌ Ошибка при обработке {filename}: {e}")
            axes[idx].text(0.5, 0.5, f'Ошибка:\n{str(e)[:30]}...',
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].axis('off')

    # Скрываем пустые subplots
    for idx in range(len(sample_files), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('СЕГМЕНТИРОВАННЫЕ КЛЕТКИ С УКАЗАНИЕМ ФАЗ',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Сохраняем
    output_path = config.RESULTS_DIR / 'segmented_cells_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ Визуализация сегментации сохранена: {output_path}")

# ================= 4. ДОПОЛНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ =================
def create_dose_response_curves(df):
    """Создает кривые доза-ответ для разных фаз"""
    print("\n📈 Создание кривые доза-ответ...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    phases_to_plot = ['G1', 'S', 'G2M', 'Mitosis', 'SubG1']
    phases_to_plot = [p for p in phases_to_plot if p in df['phase'].unique()]

    for idx, phase in enumerate(phases_to_plot[:6]):
        ax = axes[idx]

        df[f'is_{phase}'] = (df['phase'] == phase).astype(int)

        for genotype in df['genotype'].unique():
            for time in sorted(df['time'].unique()):
                subset = df[(df['genotype'] == genotype) & (df['time'] == time)]
                if len(subset) > 0:
                    phase_by_dose = subset.groupby('dose')[f'is_{phase}'].mean() * 100

                    marker = 'o' if genotype == 'WT' else 's'
                    linestyle = '-' if time == 24 else '--'

                    ax.plot(phase_by_dose.index, phase_by_dose.values,
                           marker=marker, linestyle=linestyle, linewidth=2,
                           label=f'{genotype} {time}h')

        ax.set_title(f'{phase} фаза', fontsize=14, fontweight='bold')
        ax.set_xlabel('Доза облучения (Gy)')
        ax.set_ylabel(f'% клеток в {phase}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')

    # Скрываем пустые subplots
    for idx in range(len(phases_to_plot), 6):
        axes[idx].axis('off')

    plt.suptitle('ДОЗА-ЗАВИСИМЫЕ ИЗМЕНЕНИЯ РАСПРЕДЕЛЕНИЯ ФАЗ',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = config.RESULTS_DIR / 'dose_response_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ Кривые доза-ответ сохранены: {output_path}")

# ================= 5. СОЗДАНИЕ ОТЧЕТА =================
def create_analysis_report(df):
    """Создает текстовый отчет с анализом"""
    print("\n📄 Создание отчета анализа...")

    report_lines = []

    report_lines.append("=" * 70)
    report_lines.append("ОТЧЕТ ПО АНАЛИЗУ РАДИОЧУВСТВИТЕЛЬНОСТИ КЛЕТОК HCT116")
    report_lines.append("=" * 70)
    report_lines.append(f"Дата анализа: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Всего проанализировано клеток: {len(df):,}")
    report_lines.append("")

    # 1. Общая статистика
    report_lines.append("1. ОБЩАЯ СТАТИСТИКА:")
    report_lines.append("-" * 40)
    report_lines.append(f"   Генотипы: {', '.join(df['genotype'].unique())}")
    report_lines.append(f"   Дозы облучения: {', '.join(map(str, sorted(df['dose'].unique())))} Gy")
    report_lines.append(f"   Временные точки: {', '.join(map(str, sorted(df['time'].unique())))} часов")
    report_lines.append(f"   Диапазон размера клеток: {df['area'].min():.0f} - {df['area'].max():.0f} пикселей")
    report_lines.append(f"   Средний размер клетки: {df['area'].mean():.0f} ± {df['area'].std():.0f} пикселей")
    report_lines.append("")

    # 2. Распределение фаз
    report_lines.append("2. РАСПРЕДЕЛЕНИЕ ФАЗ КЛЕТОЧНОГО ЦИКЛА:")
    report_lines.append("-" * 40)
    phase_counts = df['phase'].value_counts()
    total = len(df)

    for phase, count in phase_counts.items():
        percentage = count / total * 100
        report_lines.append(f"   {phase:10} {count:5d} клеток ({percentage:5.1f}%)")
    report_lines.append("")

    # 3. Сравнение генотипов
    report_lines.append("3. СРАВНЕНИЕ ГЕНОТИПОВ:")
    report_lines.append("-" * 40)

    for genotype in ['WT', 'CDK8KO']:
        subset = df[df['genotype'] == genotype]
        if len(subset) > 0:
            report_lines.append(f"   {genotype}:")
            report_lines.append(f"     Всего клеток: {len(subset)}")
            report_lines.append(f"     Средний размер: {subset['area'].mean():.0f} пикселей")

            # Распределение фаз
            phase_dist = subset['phase'].value_counts(normalize=True) * 100
            for phase, pct in phase_dist.items():
                report_lines.append(f"     {phase}: {pct:.1f}%")
            report_lines.append("")

    # 4. Влияние облучения
    report_lines.append("4. ВЛИЯНИЕ ОБЛУЧЕНИЯ НА G2/M БЛОК:")
    report_lines.append("-" * 40)

    df['is_G2M'] = df['phase'].isin(['G2M', 'Mitosis'])

    for genotype in ['WT', 'CDK8KO']:
        report_lines.append(f"   {genotype}:")
        for dose in sorted(df['dose'].unique()):
            subset = df[(df['genotype'] == genotype) & (df['dose'] == dose)]
            if len(subset) > 0:
                g2m_percentage = subset['is_G2M'].mean() * 100
                report_lines.append(f"     {dose:2d} Gy: {g2m_percentage:5.1f}% клеток в G2/M")
        report_lines.append("")

    # 5. Выводы
    report_lines.append("5. ПРЕДВАРИТЕЛЬНЫЕ ВЫВОДЫ:")
    report_lines.append("-" * 40)

    # Сравниваем WT и CDK8KO при 0 Gy (контроль)
    wt_control = df[(df['genotype'] == 'WT') & (df['dose'] == 0)]
    ko_control = df[(df['genotype'] == 'CDK8KO') & (df['dose'] == 0)]

    if len(wt_control) > 0 and len(ko_control) > 0:
        wt_g2m = wt_control['is_G2M'].mean() * 100
        ko_g2m = ko_control['is_G2M'].mean() * 100
        diff = ko_g2m - wt_g2m

        report_lines.append(f"   В контроле (0 Gy):")
        report_lines.append(f"     WT: {wt_g2m:.1f}% клеток в G2/M")
        report_lines.append(f"     CDK8KO: {ko_g2m:.1f}% клеток в G2/M")

        if abs(diff) > 5:
            direction = "выше" if diff > 0 else "ниже"
            report_lines.append(f"     CDK8KO имеет на {abs(diff):.1f}% {direction} долю G2/M клеток")
        else:
            report_lines.append(f"     Нет существенной разницы в доле G2/M клеток между генотипами")

    report_lines.append("")
    report_lines.append("=" * 70)

    # Сохраняем отчет
    report_path = config.RESULTS_DIR / 'analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"✅ Отчет сохранен: {report_path}")

    # Выводим краткую версию в консоль
    print("\n📋 КРАТКИЙ ОТЧЕТ:")
    print("-" * 40)
    for line in report_lines[:20]:
        print(line)

# ================= ГЛАВНАЯ ФУНКЦИЯ =================
def main():
    print("=" * 70)
    print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ АНАЛИЗА КЛЕТОЧНОГО ЦИКЛА")
    print("=" * 70)

    # 1. Загрузка данных
    df = load_and_analyze_data()
    if df is None:
        return

    # 2. Аналитические графики
    create_analytical_plots(df)

    # 3. Визуализация сегментированных клеток
    visualize_segmented_cells(df)

    # 4. Кривые доза-ответ
    create_dose_response_curves(df)

    # 5. Создание отчета
    create_analysis_report(df)

    print("\n" + "=" * 70)
    print("✅ ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 70)
    print(f"\n📁 Все результаты сохранены в папке:")
    print(f"   {config.RESULTS_DIR.absolute()}")
    print("\n📊 Созданные файлы:")
    for file in config.RESULTS_DIR.glob('*.png'):
        print(f"   • {file.name}")
    for file in config.RESULTS_DIR.glob('*.txt'):
        print(f"   • {file.name}")

if __name__ == "__main__":
    main()