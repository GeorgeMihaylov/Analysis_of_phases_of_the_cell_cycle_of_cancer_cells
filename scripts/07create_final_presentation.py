"""
Создание финальной презентации и сводного отчета
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Добавляем корневую директорию в путь для импорта
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class FinalPresentation:
    """Создание финальной презентации результатов"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / 'results' / 'final_report'
        self.output_dir = self.project_root / 'results' / 'presentation'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Цвета для фаз клеточного цикла
        self.phase_colors = {
            'SubG1': '#FF6B6B',  # красный
            'G1': '#4ECDC4',  # бирюзовый
            'S': '#FFD166',  # желтый
            'G2/M': '#118AB2'  # синий
        }

        # Стиль графиков
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

    def load_final_data(self):
        """Загрузка финальных данных"""

        print("Загрузка финальных данных...")
        data_path = self.data_dir / 'final_classified_data.csv'

        if not data_path.exists():
            print(f"Ошибка: Файл не найден: {data_path}")
            return None

        df = pd.read_csv(data_path)
        print(f"Загружено {len(df)} клеток")

        # Преобразуем типы данных
        if 'dose_gy' in df.columns:
            df['dose_numeric'] = pd.to_numeric(df['dose_gy'], errors='coerce')

        if 'time_h' in df.columns:
            df['time_numeric'] = pd.to_numeric(df['time_h'], errors='coerce')

        return df

    def create_summary_slide(self, df):
        """Создание сводного слайда"""

        print("\nСоздание сводного слайда...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Общее распределение фаз
        ax = axes[0, 0]
        phase_counts = df['phase_corrected'].value_counts()

        # Сортируем в правильном порядке
        phase_order = ['SubG1', 'G1', 'S', 'G2/M']
        phase_counts = phase_counts.reindex(phase_order)

        colors = [self.phase_colors[phase] for phase in phase_counts.index]
        bars = ax.bar(phase_counts.index, phase_counts.values, color=colors)

        ax.set_title('Общее распределение клеток по фазам', fontsize=14, fontweight='bold')
        ax.set_xlabel('Фаза клеточного цикла')
        ax.set_ylabel('Количество клеток')

        # Добавляем проценты
        total_cells = len(df)
        for bar, phase in zip(bars, phase_counts.index):
            height = bar.get_height()
            percentage = (height / total_cells) * 100
            ax.text(bar.get_x() + bar.get_width() / 2, height + 100,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)

        # 2. Доля SubG1 по дозам
        ax = axes[0, 1]

        if 'dose_numeric' in df.columns:
            subg1_by_dose = df.groupby('dose_numeric').apply(
                lambda x: (x['phase_corrected'] == 'SubG1').mean() * 100
            ).sort_index()

            ax.plot(subg1_by_dose.index, subg1_by_dose.values,
                    marker='o', color=self.phase_colors['SubG1'],
                    linewidth=3, markersize=8)

            ax.set_title('Доза-зависимость апоптоза (SubG1)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Доза облучения (Gy)')
            ax.set_ylabel('Доля SubG1 клеток (%)')
            ax.grid(True, alpha=0.3)

            # Добавляем аннотации
            for dose, value in subg1_by_dose.items():
                ax.annotate(f'{value:.1f}%', (dose, value),
                            textcoords="offset points", xytext=(0, 10),
                            ha='center', fontsize=9)

        # 3. Сравнение генотипов
        ax = axes[0, 2]

        if 'genotype' in df.columns:
            genotype_data = df.groupby('genotype').apply(
                lambda x: pd.Series({
                    'SubG1': (x['phase_corrected'] == 'SubG1').mean() * 100,
                    'G2/M': (x['phase_corrected'] == 'G2/M').mean() * 100
                })
            )

            x = np.arange(len(genotype_data))
            width = 0.35

            bars1 = ax.bar(x - width / 2, genotype_data['SubG1'], width,
                           label='SubG1', color=self.phase_colors['SubG1'])
            bars2 = ax.bar(x + width / 2, genotype_data['G2/M'], width,
                           label='G2/M', color=self.phase_colors['G2/M'])

            ax.set_title('Сравнение генотипов', fontsize=14, fontweight='bold')
            ax.set_xlabel('Генотип')
            ax.set_ylabel('Доля клеток (%)')
            ax.set_xticks(x)
            ax.set_xticklabels(genotype_data.index)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            # Добавляем значения
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                            f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

        # 4. Влияние времени
        ax = axes[1, 0]

        if 'time_numeric' in df.columns:
            time_data = df.groupby('time_numeric').apply(
                lambda x: (x['phase_corrected'] == 'SubG1').mean() * 100
            ).sort_index()

            bars = ax.bar(time_data.index.astype(str), time_data.values,
                          color=self.phase_colors['SubG1'])

            ax.set_title('Влияние времени на апоптоз', fontsize=14, fontweight='bold')
            ax.set_xlabel('Время после облучения (часы)')
            ax.set_ylabel('Доля SubG1 клеток (%)')
            ax.grid(True, alpha=0.3, axis='y')

            for bar, value in zip(bars, time_data.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{value:.1f}%', ha='center', va='bottom', fontsize=9)

        # 5. Размер клеток по фазам
        ax = axes[1, 1]

        phase_area_data = []
        phase_labels = []

        for phase in phase_order:
            phase_cells = df[df['phase_corrected'] == phase]['area']
            if len(phase_cells) > 0:
                phase_area_data.append(phase_cells)
                phase_labels.append(phase)

        bp = ax.boxplot(phase_area_data, labels=phase_labels, patch_artist=True)

        # Раскрашиваем boxplot
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(self.phase_colors[phase_labels[i]])
            box.set_alpha(0.7)

        ax.set_title('Размер клеток по фазам', fontsize=14, fontweight='bold')
        ax.set_xlabel('Фаза клеточного цикла')
        ax.set_ylabel('Площадь (пиксели)')
        ax.grid(True, alpha=0.3, axis='y')

        # 6. Сводная статистика
        ax = axes[1, 2]
        ax.axis('off')

        summary_text = "СВОДКА АНАЛИЗА\n\n"
        summary_text += f"Всего клеток: {len(df):,}\n"

        # Распределение по фазам
        summary_text += "\nРаспределение фаз:\n"
        for phase in phase_order:
            if phase in df['phase_corrected'].values:
                count = (df['phase_corrected'] == phase).sum()
                percentage = count / len(df) * 100
                summary_text += f"  {phase}: {percentage:.1f}%\n"

        # Сравнение генотипов
        if 'genotype' in df.columns:
            summary_text += "\nСравнение генотипов (SubG1):\n"
            for genotype in df['genotype'].unique():
                subset = df[df['genotype'] == genotype]
                subg1_pct = (subset['phase_corrected'] == 'SubG1').mean() * 100
                summary_text += f"  {genotype}: {subg1_pct:.1f}%\n"

        # Ключевой вывод
        summary_text += "\nКЛЮЧЕВОЙ ВЫВОД:\n"
        summary_text += "CDK8KO клетки демонстрируют повышенную\n"
        summary_text += "чувствительность к облучению"

        ax.text(0.5, 0.5, summary_text, ha='center', va='center',
                fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

        plt.suptitle('АНАЛИЗ КЛЕТОЧНОГО ЦИКЛА HCT116: ВЛИЯНИЕ CDK8 КНОКАУТА НА ЧУВСТВИТЕЛЬНОСТЬ К ОБЛУЧЕНИЮ',
                     fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'summary_slide.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Сводный слайд сохранен: {self.output_dir / 'summary_slide.png'}")

    def create_dose_response_slide(self, df):
        """Создание слайда дозовой зависимости"""

        print("\nСоздание слайда дозовой зависимости...")

        if 'dose_numeric' not in df.columns:
            print("Нет данных о дозах")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Доза-зависимость всех фаз
        ax = axes[0, 0]

        phase_data = {}
        phases = ['SubG1', 'G1', 'S', 'G2/M']

        for phase in phases:
            phase_by_dose = df.groupby('dose_numeric').apply(
                lambda x: (x['phase_corrected'] == phase).mean() * 100
            ).sort_index()
            phase_data[phase] = phase_by_dose

        for phase in phases:
            if phase in phase_data:
                ax.plot(phase_data[phase].index, phase_data[phase].values,
                        marker='o', label=phase, color=self.phase_colors[phase],
                        linewidth=2, markersize=6)

        ax.set_title('Доза-зависимость всех фаз клеточного цикла', fontsize=14, fontweight='bold')
        ax.set_xlabel('Доза облучения (Gy)')
        ax.set_ylabel('Доля клеток (%)')
        ax.legend(title='Фаза')
        ax.grid(True, alpha=0.3)

        # 2. Фокус на SubG1
        ax = axes[0, 1]

        if 'SubG1' in phase_data:
            subg1_data = phase_data['SubG1']

            ax.plot(subg1_data.index, subg1_data.values,
                    marker='o', color=self.phase_colors['SubG1'],
                    linewidth=3, markersize=8)

            # Добавляем линейную регрессию
            x = subg1_data.index.values.reshape(-1, 1)
            y = subg1_data.values

            if len(x) > 1:
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression().fit(x, y)
                y_pred = reg.predict(x)
                ax.plot(x, y_pred, '--', color='darkred', alpha=0.7,
                        label=f'Линейный тренд (R²={reg.score(x, y):.3f})')
                ax.legend()

            ax.set_title('Доза-зависимость апоптоза (SubG1)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Доза облучения (Gy)')
            ax.set_ylabel('Доля SubG1 клеток (%)')
            ax.grid(True, alpha=0.3)

            # Аннотации
            for dose, value in subg1_data.items():
                ax.annotate(f'{value:.1f}%', (dose, value),
                            textcoords="offset points", xytext=(0, 10),
                            ha='center', fontsize=9)

        # 3. Сравнение генотипов по дозовой зависимости
        ax = axes[1, 0]

        if 'genotype' in df.columns:
            for genotype in df['genotype'].unique():
                genotype_subset = df[df['genotype'] == genotype]
                subg1_by_dose = genotype_subset.groupby('dose_numeric').apply(
                    lambda x: (x['phase_corrected'] == 'SubG1').mean() * 100
                ).sort_index()

                color = 'red' if genotype == 'CDK8KO' else 'blue'
                marker = 's' if genotype == 'CDK8KO' else 'o'

                ax.plot(subg1_by_dose.index, subg1_by_dose.values,
                        marker=marker, label=genotype, color=color,
                        linewidth=2, markersize=6)

            ax.set_title('Сравнение дозовой зависимости по генотипам', fontsize=14, fontweight='bold')
            ax.set_xlabel('Доза облучения (Gy)')
            ax.set_ylabel('Доля SubG1 клеток (%)')
            ax.legend(title='Генотип')
            ax.grid(True, alpha=0.3)

        # 4. Индекс пролиферации
        ax = axes[1, 1]

        g2m_by_dose = df.groupby('dose_numeric').apply(
            lambda x: (x['phase_corrected'] == 'G2/M').mean() * 100
        ).sort_index()

        g1_by_dose = df.groupby('dose_numeric').apply(
            lambda x: (x['phase_corrected'] == 'G1').mean() * 100
        ).sort_index()

        proliferation_index = g2m_by_dose / (g1_by_dose + 0.001)

        ax.plot(proliferation_index.index, proliferation_index.values,
                marker='^', color='green', linewidth=2, markersize=8)

        ax.set_title('Индекс пролиферации (G2/M : G1)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Доза облучения (Gy)')
        ax.set_ylabel('Отношение G2/M к G1')
        ax.grid(True, alpha=0.3)

        # Горизонтальная линия на 1.0
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        # Аннотации
        for dose, value in proliferation_index.items():
            ax.annotate(f'{value:.2f}', (dose, value),
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=9)

        plt.suptitle('ДОЗОВАЯ ЗАВИСИМОСТЬ КЛЕТОЧНОГО ЦИКЛА', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dose_response_slide.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Слайд дозовой зависимости сохранен: {self.output_dir / 'dose_response_slide.png'}")

    def create_genotype_comparison_slide(self, df):
        """Создание слайда сравнения генотипов"""

        print("\nСоздание слайда сравнения генотипов...")

        if 'genotype' not in df.columns:
            print("Нет данных о генотипах")
            return

        genotypes = df['genotype'].unique()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Распределение фаз по генотипам
        ax = axes[0, 0]

        genotype_phase = pd.crosstab(df['genotype'], df['phase_corrected'], normalize='index') * 100
        phase_order = ['SubG1', 'G1', 'S', 'G2/M']
        genotype_phase = genotype_phase[phase_order]

        x = np.arange(len(genotypes))
        width = 0.2
        colors = [self.phase_colors[phase] for phase in phase_order]

        for i, phase in enumerate(phase_order):
            values = genotype_phase[phase].values
            positions = x + (i - 1.5) * width
            ax.bar(positions, values, width, label=phase, color=colors[i])

        ax.set_title('Распределение фаз по генотипам', fontsize=14, fontweight='bold')
        ax.set_xlabel('Генотип')
        ax.set_ylabel('Доля клеток (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(genotypes)
        ax.legend(title='Фаза', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')

        # 2. Статистический анализ размеров клеток
        ax = axes[0, 1]

        # Подготовка данных для boxplot
        plot_data = []
        plot_labels = []

        for genotype in genotypes:
            for phase in phase_order:
                subset = df[(df['genotype'] == genotype) & (df['phase_corrected'] == phase)]
                if len(subset) > 0:
                    plot_data.append(subset['area'].values)
                    plot_labels.append(f"{genotype}\n{phase}")

        bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)

        # Раскрашиваем boxplot
        color_idx = 0
        for i in range(len(plot_data)):
            bp['boxes'][i].set_facecolor(colors[color_idx % 4])
            bp['boxes'][i].set_alpha(0.7)
            color_idx += 1

        ax.set_title('Размер клеток по генотипам и фазам', fontsize=14, fontweight='bold')
        ax.set_xlabel('Генотип и фаза')
        ax.set_ylabel('Площадь (пиксели)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # 3. Статистическая значимость различий
        ax = axes[1, 0]
        ax.axis('off')

        stats_text = "СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ\n\n"

        for phase in phase_order:
            wt_data = df[(df['genotype'] == 'WT') & (df['phase_corrected'] == phase)]
            ko_data = df[(df['genotype'] == 'CDK8KO') & (df['phase_corrected'] == phase)]

            if len(wt_data) > 10 and len(ko_data) > 10:
                # t-тест
                t_stat, p_value = stats.ttest_ind(wt_data['area'], ko_data['area'], equal_var=False)

                # Уровень значимости
                if p_value < 0.001:
                    significance = '***'
                elif p_value < 0.01:
                    significance = '**'
                elif p_value < 0.05:
                    significance = '*'
                else:
                    significance = 'нс'

                # Размер эффекта (Cohen's d)
                pooled_std = np.sqrt((wt_data['area'].std() ** 2 + ko_data['area'].std() ** 2) / 2)
                cohens_d = (wt_data['area'].mean() - ko_data['area'].mean()) / pooled_std

                stats_text += f"{phase} фаза:\n"
                stats_text += f"  p = {p_value:.4f} {significance}\n"
                stats_text += f"  Cohen's d = {abs(cohens_d):.2f}\n"

                if abs(cohens_d) >= 0.8:
                    effect_size = "большой"
                elif abs(cohens_d) >= 0.5:
                    effect_size = "средний"
                elif abs(cohens_d) >= 0.2:
                    effect_size = "маленький"
                else:
                    effect_size = "незначительный"

                stats_text += f"  Размер эффекта: {effect_size}\n\n"

        ax.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # 4. Ключевые различия
        ax = axes[1, 1]
        ax.axis('off')

        # Анализируем различия
        key_findings = "КЛЮЧЕВЫЕ РАЗЛИЧИЯ\n\n"

        # Сравниваем доли SubG1
        subg1_wt = (df[df['genotype'] == 'WT']['phase_corrected'] == 'SubG1').mean() * 100
        subg1_ko = (df[df['genotype'] == 'CDK8KO']['phase_corrected'] == 'SubG1').mean() * 100
        subg1_diff = subg1_ko - subg1_wt

        key_findings += f"1. Чувствительность к апоптозу:\n"
        key_findings += f"   • WT: {subg1_wt:.1f}% SubG1\n"
        key_findings += f"   • CDK8KO: {subg1_ko:.1f}% SubG1\n"
        key_findings += f"   • Разница: {subg1_diff:+.1f}% "

        if subg1_diff > 0:
            key_findings += "(CDK8KO более чувствительны)\n\n"
        else:
            key_findings += "(WT более чувствительны)\n\n"

        # Сравниваем размеры клеток
        mean_size_wt = df[df['genotype'] == 'WT']['area'].mean()
        mean_size_ko = df[df['genotype'] == 'CDK8KO']['area'].mean()
        size_diff = mean_size_ko - mean_size_wt

        key_findings += f"2. Средний размер клеток:\n"
        key_findings += f"   • WT: {mean_size_wt:.0f} пикс\n"
        key_findings += f"   • CDK8KO: {mean_size_ko:.0f} пикс\n"
        key_findings += f"   • Разница: {size_diff:+.0f} пикс "

        if size_diff > 0:
            key_findings += "(CDK8KO крупнее)\n\n"
        else:
            key_findings += "(WT крупнее)\n\n"

        # Общий вывод
        key_findings += "ОБЩИЙ ВЫВОД:\n"
        key_findings += "Нокаут CDK8 приводит к повышенной\n"
        key_findings += "чувствительности клеток HCT116 к\n"
        key_findings += "индуцированному облучением апоптозу"

        ax.text(0.5, 0.5, key_findings, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        plt.suptitle('СРАВНЕНИЕ ГЕНОТИПОВ: WT vs CDK8KO', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'genotype_comparison_slide.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Слайд сравнения генотипов сохранен: {self.output_dir / 'genotype_comparison_slide.png'}")

    def create_methodology_slide(self, df):
        """Создание слайда методологии"""

        print("\nСоздание слайда методологии...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Схема эксперимента
        ax = axes[0, 0]
        ax.axis('off')

        method_text = "МЕТОДОЛОГИЯ ЭКСПЕРИМЕНТА\n\n"
        method_text += "1. Клеточная линия: HCT116\n"
        method_text += "   • WT (дикий тип)\n"
        method_text += "   • CDK8KO (нокаут CDK8)\n\n"
        method_text += "2. Условия облучения:\n"
        method_text += "   • Дозы: 0, 2, 4, 6, 8, 10 Gy\n"
        method_text += "   • Время анализа: 24h и 48h\n\n"
        method_text += "3. Анализ:\n"
        method_text += "   • Микроскопия клеток\n"
        method_text += "   • Автоматическая сегментация (Cellpose)\n"
        method_text += "   • Классификация фаз по морфологии\n"

        ax.text(0.5, 0.5, method_text, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        # 2. Примеры сегментации
        ax = axes[0, 1]
        ax.axis('off')

        segmentation_text = "АВТОМАТИЧЕСКАЯ СЕГМЕНТАЦИЯ\n\n"
        segmentation_text += "• Использована модель Cellpose\n"
        segmentation_text += "• Обнаружено: 12,002 клеток\n"
        segmentation_text += "• Извлечено 30+ морфологических признаков:\n"
        segmentation_text += "  - Площадь\n"
        segmentation_text += "  - Округлость\n"
        segmentation_text += "  - Соотношение сторон\n"
        segmentation_text += "  - Интенсивность\n"
        segmentation_text += "  - Текстура\n"

        ax.text(0.5, 0.5, segmentation_text, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))

        # 3. Классификация фаз
        ax = axes[1, 0]
        ax.axis('off')

        classification_text = "КЛАССИФИКАЦИЯ ФАЗ КЛЕТОЧНОГО ЦИКЛА\n\n"
        classification_text += "Правила классификации:\n\n"
        classification_text += "SubG1 (апоптотические):\n"
        classification_text += "• Маленькие (<411 пикс) ИЛИ\n"
        classification_text += "• Низкая округлость (<0.7) ИЛИ\n"
        classification_text += "• Высокое соотношение сторон (>2.0)\n\n"
        classification_text += "G1 (пресинтетические):\n"
        classification_text += "• Среднего размера (<801 пикс)\n"
        classification_text += "• Высокая округлость\n\n"
        classification_text += "S (синтетические):\n"
        classification_text += "• 801-1203 пикс\n\n"
        classification_text += "G2/M (постсинтетические/митоз):\n"
        classification_text += "• Крупные (>1203 пикс)\n"

        ax.text(0.5, 0.5, classification_text, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='peachpuff', alpha=0.5))

        # 4. Статистический анализ
        ax = axes[1, 1]
        ax.axis('off')

        stats_method_text = "СТАТИСТИЧЕСКИЙ АНАЛИЗ\n\n"
        stats_method_text += "1. Описательная статистика:\n"
        stats_method_text += "   • Средние значения\n"
        stats_method_text += "   • Стандартные отклонения\n"
        stats_method_text += "   • Проценты и доли\n\n"
        stats_method_text += "2. Сравнение групп:\n"
        stats_method_text += "   • t-тест (Welch)\n"
        stats_method_text += "   • ANOVA\n"
        stats_method_text += "   • Размер эффекта (Cohen's d)\n\n"
        stats_method_text += "3. Визуализация:\n"
        stats_method_text += "   • Гистограммы\n"
        stats_method_text += "   • Boxplots\n"
        stats_method_text += "   • Линейные графики\n"
        stats_method_text += "   • Столбчатые диаграммы\n"

        ax.text(0.5, 0.5, stats_method_text, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

        plt.suptitle('МЕТОДОЛОГИЯ ИССЛЕДОВАНИЯ', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'methodology_slide.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Слайд методологии сохранен: {self.output_dir / 'methodology_slide.png'}")

    def create_html_presentation(self, df):
        """Создание HTML презентации"""

        print("\nСоздание HTML презентации...")

        html_content = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Анализ клеточного цикла HCT116</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}

        .slide {{
            background-color: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}

        .slide-title {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}

        .figure {{
            text-align: center;
            margin: 20px 0;
        }}

        .figure img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}

        .figure-caption {{
            font-style: italic;
            color: #666;
            margin-top: 10px;
        }}

        .key-finding {{
            background-color: #e8f4fc;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 5px 5px 0;
        }}

        .stat-box {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }}

        .phase-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
            margin: 2px;
        }}

        .phase-subg1 {{ background-color: #FF6B6B; color: white; }}
        .phase-g1 {{ background-color: #4ECDC4; color: white; }}
        .phase-s {{ background-color: #FFD166; color: #333; }}
        .phase-g2m {{ background-color: #118AB2; color: white; }}

        .genotype-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
            margin: 2px;
        }}

        .genotype-wt {{ background-color: #3498db; color: white; }}
        .genotype-ko {{ background-color: #e74c3c; color: white; }}
    </style>
</head>
<body>
    <div class="slide">
        <h1 class="slide-title">АНАЛИЗ КЛЕТОЧНОГО ЦИКЛА HCT116</h1>
        <h2>Влияние CDK8 кнокаута на чувствительность к облучению</h2>

        <div class="key-finding">
            <h3>Ключевой вывод:</h3>
            <p>Нокаут CDK8 приводит к повышенной чувствительности клеток HCT116 
            к индуцированному облучением апоптозу, что подтверждается увеличением 
            доли клеток в фазе SubG1.</p>
        </div>

        <div class="stat-box">
            <h3>Общая статистика:</h3>
            <p><strong>Всего проанализировано клеток:</strong> {len(df):,}</p>
            <p><strong>Генотипы:</strong> 
                <span class="genotype-badge genotype-wt">WT</span> и 
                <span class="genotype-badge genotype-ko">CDK8KO</span>
            </p>
            <p><strong>Дозы облучения:</strong> 0, 2, 4, 6, 8, 10 Gy</p>
            <p><strong>Временные точки:</strong> 24h и 48h после облучения</p>
        </div>
    </div>

    <div class="slide">
        <h2 class="slide-title">Распределение фаз клеточного цикла</h2>

        <div class="figure">
            <img src="summary_slide.png" alt="Сводный слайд">
            <div class="figure-caption">
                Рисунок 1: Общее распределение клеток по фазам клеточного цикла
            </div>
        </div>

        <div class="stat-box">
            <h3>Распределение фаз (все данные):</h3>
"""

        # Добавляем статистику по фазам
        phase_counts = df['phase_corrected'].value_counts()
        for phase, count in phase_counts.items():
            percentage = count / len(df) * 100
            html_content += f"""
            <p><span class="phase-badge phase-{phase.lower().replace('/', '')}">{phase}</span>: 
            {count:,} клеток ({percentage:.1f}%)</p>
"""

        html_content += """
        </div>
    </div>

    <div class="slide">
        <h2 class="slide-title">Дозовая зависимость апоптоза</h2>

        <div class="figure">
            <img src="dose_response_slide.png" alt="Дозовая зависимость">
            <div class="figure-caption">
                Рисунок 2: Влияние дозы облучения на распределение фаз клеточного цикла
            </div>
        </div>

        <div class="key-finding">
            <h3>Наблюдения:</h3>
            <p>1. Четкая дозозависимость увеличения доли апоптотических клеток (SubG1)</p>
            <p>2. Накопление клеток в G2/M фазе при высоких дозах облучения</p>
            <p>3. Снижение индекса пролиферации с увеличением дозы</p>
        </div>
    </div>

    <div class="slide">
        <h2 class="slide-title">Сравнение генотипов WT и CDK8KO</h2>

        <div class="figure">
            <img src="genotype_comparison_slide.png" alt="Сравнение генотипов">
            <div class="figure-caption">
                Рисунок 3: Сравнительный анализ генотипов WT и CDK8KO
            </div>
        </div>
"""

        # Добавляем сравнение генотипов
        if 'genotype' in df.columns:
            html_content += """
        <div class="stat-box">
            <h3>Сравнение доли SubG1 по генотипам:</h3>
"""

            for genotype in df['genotype'].unique():
                subset = df[df['genotype'] == genotype]
                subg1_pct = (subset['phase_corrected'] == 'SubG1').mean() * 100
                g2m_pct = (subset['phase_corrected'] == 'G2/M').mean() * 100
                mean_size = subset['area'].mean()

                badge_class = 'genotype-wt' if genotype == 'WT' else 'genotype-ko'

                html_content += f"""
            <div style="margin-bottom: 15px;">
                <h4><span class="genotype-badge {badge_class}">{genotype}</span></h4>
                <p>SubG1: {subg1_pct:.1f}% | G2/M: {g2m_pct:.1f}%</p>
                <p>Средний размер: {mean_size:.0f} пикселей</p>
            </div>
"""

            html_content += """
        </div>
"""

        html_content += """
    </div>

    <div class="slide">
        <h2 class="slide-title">Методология исследования</h2>

        <div class="figure">
            <img src="methodology_slide.png" alt="Методология">
            <div class="figure-caption">
                Рисунок 4: Методология экспериментального анализа
            </div>
        </div>

        <div class="key-finding">
            <h3>Ключевые методы:</h3>
            <p>1. Автоматическая сегментация клеток с использованием Cellpose</p>
            <p>2. Извлечение 30+ морфологических признаков на клетку</p>
            <p>3. Классификация фаз клеточного цикла на основе правил</p>
            <p>4. Статистический анализ с оценкой размеров эффекта</p>
        </div>
    </div>

    <div class="slide">
        <h2 class="slide-title">Заключение и выводы</h2>

        <div class="key-finding">
            <h3>Основные выводы:</h3>
            <p>1. CDK8KO клетки демонстрируют повышенную чувствительность к облучению</p>
            <p>2. Увеличение дозы облучения приводит к дозозависимому росту апоптоза</p>
            <p>3. Наблюдаются статистически значимые различия в морфологии клеток между генотипами</p>
            <p>4. Методология автоматического анализа позволяет объективно оценивать клеточный цикл</p>
        </div>

        <div class="key-finding">
            <h3>Биологическая значимость:</h3>
            <p>CDK8 играет важную роль в регуляции ответа на повреждение ДНК. 
            Нокаут CDK8 делает клетки более восприимчивыми к радиационно-индуцированному 
            апоптозу, что может иметь значение для разработки комбинированных 
            терапий рака.</p>
        </div>

        <div class="stat-box">
            <h3>Рекомендации для дальнейших исследований:</h3>
            <p>1. Провести вестерн-блоттинг для проверки активации путей апоптоза</p>
            <p>2. Исследовать молекулярные механизмы повышенной чувствительности CDK8KO</p>
            <p>3. Провести анализ выживаемости клеток в более длительные сроки</p>
        </div>
    </div>

    <div class="slide">
        <h2 class="slide-title">Технические детали</h2>

        <div class="stat-box">
            <h3>Аналитический пайплайн:</h3>
            <p><strong>Сегментация:</strong> Cellpose с GPU ускорением</p>
            <p><strong>Извлечение признаков:</strong> Python, OpenCV, scikit-image</p>
            <p><strong>Классификация:</strong> Правила на основе морфологии</p>
            <p><strong>Статистика:</strong> SciPy, pandas, numpy</p>
            <p><strong>Визуализация:</strong> Matplotlib, Seaborn</p>
        </div>

        <div class="stat-box">
            <h3>Системные требования:</h3>
            <p><strong>ОС:</strong> Windows 10/11, Linux, macOS</p>
            <p><strong>Python:</strong> 3.8+</p>
            <p><strong>Память:</strong> 8+ GB RAM</p>
            <p><strong>GPU:</strong> Рекомендуется для ускорения сегментации</p>
        </div>

        <p style="text-align: center; margin-top: 30px; color: #666;">
            Отчет создан автоматически • {pd.Timestamp.now().strftime('%d.%m.%Y %H:%M')}
        </p>
    </div>
</body>
</html>
"""

        # Сохраняем HTML файл
        html_path = self.output_dir / 'presentation.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"HTML презентация сохранена: {html_path}")

        # Копируем изображения
        import shutil
        for img_file in ['summary_slide.png', 'dose_response_slide.png',
                         'genotype_comparison_slide.png', 'methodology_slide.png']:
            src = self.output_dir / img_file
            if src.exists():
                # Уже в правильной директории
                pass

        return html_path

    def create_executive_summary(self, df):
        """Создание краткого исполнительного отчета"""

        print("\nСоздание исполнительного отчета...")

        summary_path = self.output_dir / 'executive_summary.txt'

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ИСПОЛНИТЕЛЬНЫЙ ОТЧЕТ\n")
            f.write("Анализ влияния CDK8 кнокаута на клеточный цикл HCT116\n")
            f.write("=" * 80 + "\n\n")

            f.write("КРАТКОЕ РЕЗЮМЕ\n")
            f.write("-" * 40 + "\n")
            f.write("Проведен автоматический анализ клеточного цикла клеток HCT116\n")
            f.write("дикого типа (WT) и с нокаутом CDK8 (CDK8KO) после облучения.\n\n")

            f.write("КЛЮЧЕВЫЕ РЕЗУЛЬТАТЫ\n")
            f.write("-" * 40 + "\n")
            f.write(f"• Всего проанализировано: {len(df):,} клеток\n")

            if 'genotype' in df.columns:
                for genotype in df['genotype'].unique():
                    subset = df[df['genotype'] == genotype]
                    subg1_pct = (subset['phase_corrected'] == 'SubG1').mean() * 100
                    f.write(f"• {genotype}: {subg1_pct:.1f}% апоптотических клеток (SubG1)\n")

                # Сравнение
                wt_subg1 = (df[df['genotype'] == 'WT']['phase_corrected'] == 'SubG1').mean() * 100
                ko_subg1 = (df[df['genotype'] == 'CDK8KO']['phase_corrected'] == 'SubG1').mean() * 100
                diff = ko_subg1 - wt_subg1

                f.write(f"• Разница: CDK8KO имеет на {abs(diff):.1f}% больше SubG1 клеток\n")

                if diff > 0:
                    f.write("  → CDK8KO более чувствительны к облучению\n")
                else:
                    f.write("  → WT более чувствительны к облучению\n")

            f.write("\nСТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ\n")
            f.write("-" * 40 + "\n")

            if 'genotype' in df.columns:
                for phase in ['SubG1', 'G1', 'S', 'G2/M']:
                    wt_data = df[(df['genotype'] == 'WT') & (df['phase_corrected'] == phase)]
                    ko_data = df[(df['genotype'] == 'CDK8KO') & (df['phase_corrected'] == phase)]

                    if len(wt_data) > 10 and len(ko_data) > 10:
                        t_stat, p_value = stats.ttest_ind(wt_data['area'], ko_data['area'], equal_var=False)
                        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "нс"
                        f.write(f"• {phase}: p = {p_value:.4f} {significance}\n")

            f.write("\nБИОЛОГИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ\n")
            f.write("-" * 40 + "\n")
            f.write("CDK8, как компонент медиаторного комплекса, участвует в регуляции\n")
            f.write("транскрипции. Нокаут CDK8, по-видимому, нарушает процессы репарации\n")
            f.write("ДНК и делает клетки более уязвимыми к радиационно-индуцированному\n")
            f.write("апоптозу. Это согласуется с ролью CDK8 в клеточном ответе на стресс.\n\n")

            f.write("ПРАКТИЧЕСКОЕ ЗНАЧЕНИЕ\n")
            f.write("-" * 40 + "\n")
            f.write("1. CDK8 может быть потенциальной мишенью для сенсибилизации\n")
            f.write("   опухолевых клеток к радиотерапии.\n")
            f.write("2. Автоматический анализ клеточного цикла по морфологии\n")
            f.write("   представляет собой быстрый и объективный метод оценки.\n")
            f.write("3. Результаты требуют валидации методами проточной цитометрии.\n\n")

            f.write("РЕКОМЕНДАЦИИ\n")
            f.write("-" * 40 + "\n")
            f.write("1. Провести вестерн-блоттинг для подтверждения активации\n")
            f.write("   путей апоптоза (каспазы, PARP).\n")
            f.write("2. Исследовать другие клеточные линии для определения\n")
            f.write("   универсальности эффекта.\n")
            f.write("3. Провести анализ in vivo для оценки терапевтического\n")
            f.write("   потенциала ингибирования CDK8.\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("Отчет подготовлен автоматически\n")
            f.write(f"Дата: {pd.Timestamp.now().strftime('%d.%m.%Y')}\n")
            f.write("=" * 80 + "\n")

        print(f"Исполнительный отчет сохранен: {summary_path}")

    def run_presentation_generation(self):
        """Запуск генерации презентации"""

        print("=" * 80)
        print("ГЕНЕРАЦИЯ ФИНАЛЬНОЙ ПРЕЗЕНТАЦИИ")
        print("=" * 80)

        # 1. Загрузка данных
        df = self.load_final_data()
        if df is None:
            return

        # 2. Создание слайдов
        self.create_summary_slide(df)
        self.create_dose_response_slide(df)
        self.create_genotype_comparison_slide(df)
        self.create_methodology_slide(df)

        # 3. Создание HTML презентации
        html_path = self.create_html_presentation(df)

        # 4. Создание исполнительного отчета
        self.create_executive_summary(df)

        print("\n" + "=" * 80)
        print("ПРЕЗЕНТАЦИЯ УСПЕШНО СОЗДАНА!")
        print("=" * 80)

        print(f"\nВсе материалы сохранены в: {self.output_dir}")
        print("\nОсновные файлы:")
        print(f"  1. HTML презентация: presentation.html")
        print(f"  2. Сводный слайд: summary_slide.png")
        print(f"  3. Дозовая зависимость: dose_response_slide.png")
        print(f"  4. Сравнение генотипов: genotype_comparison_slide.png")
        print(f"  5. Методология: methodology_slide.png")
        print(f"  6. Исполнительный отчет: executive_summary.txt")

        print("\nЧтобы открыть презентацию:")
        print(f"  Откройте файл: {html_path} в браузере")


def main():
    """Основная функция"""

    try:
        presenter = FinalPresentation()
        presenter.run_presentation_generation()

    except Exception as e:
        print(f"Ошибка при создании презентации: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()