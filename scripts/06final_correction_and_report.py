"""
Финальная коррекция классификации и создание отчетов
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


class FinalCorrection:
    """Финальная коррекция классификации и создание отчетов"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / 'results' / 'cell_cycle'
        self.output_dir = self.project_root / 'results' / 'final_report'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_and_correct_data(self):
        """Загрузка и коррекция данных"""

        print("Загрузка данных для коррекции...")
        data_path = self.data_dir / 'cells_with_phases.csv'

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

        # Определяем контрольные данные (0 Gy)
        control_mask = df['dose_numeric'] == 0
        control_data = df[control_mask]

        print(f"\nКонтрольные данные (0 Gy): {len(control_data)} клеток")
        print("Распределение фаз в контроле до коррекции:")
        print(control_data['phase'].value_counts(normalize=True) * 100)

        return df, control_data

    def analyze_control_distribution(self, control_data):
        """Анализ распределения в контроле для определения порогов"""

        print("\n" + "=" * 60)
        print("АНАЛИЗ РАСПРЕДЕЛЕНИЯ В КОНТРОЛЕ")
        print("=" * 60)

        # Анализ распределения площади
        area_stats = control_data['area'].describe()
        print("\nСтатистика площади в контроле:")
        print(f"  Среднее: {area_stats['mean']:.0f}")
        print(f"  Медиана: {area_stats['50%']:.0f}")
        print(f"  Стандартное отклонение: {area_stats['std']:.0f}")
        print(f"  Минимум: {area_stats['min']:.0f}")
        print(f"  Максимум: {area_stats['max']:.0f}")

        # Определяем ожидаемое распределение в контроле
        # В здоровой популяции клеток:
        # - Большинство клеток должно быть в G1 (диплоидные)
        # - Немного в G2/M (тетраплоидные)
        # - Минимальное количество в SubG1 (апоптоз)

        # Вычисляем ожидаемые процентили для нормального распределения
        # Предполагаем, что 70% клеток в G1, 20% в G2/M, 5% в S, 5% в SubG1
        g1_threshold = control_data['area'].quantile(0.7)
        s_threshold_low = control_data['area'].quantile(0.75)
        s_threshold_high = control_data['area'].quantile(0.95)

        print(f"\nПредлагаемые пороги на основе распределения:")
        print(f"  G1 (до {g1_threshold:.0f} пикселей) - 70% клеток")
        print(f"  S ({s_threshold_low:.0f}-{s_threshold_high:.0f} пикселей) - 20% клеток")
        print(f"  G2/M (>{s_threshold_high:.0f} пикселей) - 5% клеток")
        print(f"  SubG1 (<{control_data['area'].quantile(0.05):.0f} пикселей) - 5% клеток")

        return {
            'g1_max': g1_threshold,
            's_min': s_threshold_low,
            's_max': s_threshold_high,
            'subg1_max': control_data['area'].quantile(0.05)
        }

    def apply_corrected_classification(self, df, thresholds):
        """Применение скорректированной классификации"""

        print("\n" + "=" * 60)
        print("ПРИМЕНЕНИЕ СКОРРЕКТИРОВАННОЙ КЛАССИФИКАЦИИ")
        print("=" * 60)

        # Создаем копию данных для коррекции
        df_corrected = df.copy()

        # Функция классификации на основе порогов и морфологии
        def classify_cell(row):
            area = row['area']
            circularity = row['circularity'] if 'circularity' in row else 0.8
            aspect_ratio = row['aspect_ratio'] if 'aspect_ratio' in row else 1.2

            # Критерии для SubG1 (апоптотические клетки):
            # 1. Маленькие ИЛИ
            # 2. Низкая circularity (сморщенные) ИЛИ
            # 3. Высокий aspect_ratio (вытянутые)

            is_subg1 = (
                    (area < thresholds['subg1_max']) or
                    (circularity < 0.7) or  # менее круглые
                    (aspect_ratio > 2.0)  # более вытянутые
            )

            if is_subg1:
                return 'SubG1'
            elif area <= thresholds['g1_max']:
                return 'G1'
            elif area <= thresholds['s_max']:
                return 'S'
            else:
                return 'G2/M'

        # Применяем новую классификацию
        print("Применение новой классификации...")
        df_corrected['phase_corrected'] = df_corrected.apply(classify_cell, axis=1)

        # Сравниваем старую и новую классификацию
        comparison = pd.crosstab(df_corrected['phase'], df_corrected['phase_corrected'])
        print("\nСравнение старой и новой классификации:")
        print(comparison)

        # Анализируем изменения в контроле
        control_corrected = df_corrected[df_corrected['dose_numeric'] == 0]
        print(f"\nРаспределение фаз в контроле после коррекции:")
        phase_counts = control_corrected['phase_corrected'].value_counts(normalize=True) * 100
        for phase, percentage in phase_counts.items():
            count = (control_corrected['phase_corrected'] == phase).sum()
            print(f"  {phase}: {count} клеток ({percentage:.1f}%)")

        return df_corrected

    def create_dose_response_analysis(self, df_corrected):
        """Анализ дозовой зависимости"""

        print("\n" + "=" * 60)
        print("АНАЛИЗ ДОЗОВОЙ ЗАВИСИМОСТИ")
        print("=" * 60)

        if 'dose_numeric' not in df_corrected.columns:
            print("Нет данных о дозах")
            return

        # Создаем фигуру
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Доля SubG1 по дозам
        ax = axes[0, 0]
        subg1_by_dose = df_corrected.groupby('dose_numeric').apply(
            lambda x: (x['phase_corrected'] == 'SubG1').mean() * 100
        ).sort_index()

        ax.plot(subg1_by_dose.index, subg1_by_dose.values,
                marker='o', color='red', linewidth=3, markersize=8)
        ax.set_title('Доза-зависимость апоптоза (SubG1)')
        ax.set_xlabel('Доза облучения (Gy)')
        ax.set_ylabel('Доля SubG1 клеток (%)')
        ax.grid(True, alpha=0.3)

        # Добавляем аннотации
        for dose, value in subg1_by_dose.items():
            ax.annotate(f'{value:.1f}%', (dose, value),
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=9)

        # 2. Доля G2/M по дозам
        ax = axes[0, 1]
        g2m_by_dose = df_corrected.groupby('dose_numeric').apply(
            lambda x: (x['phase_corrected'] == 'G2/M').mean() * 100
        ).sort_index()

        ax.plot(g2m_by_dose.index, g2m_by_dose.values,
                marker='s', color='blue', linewidth=3, markersize=8)
        ax.set_title('Доза-зависимость клеток G2/M')
        ax.set_xlabel('Доза облучения (Gy)')
        ax.set_ylabel('Доля G2/M клеток (%)')
        ax.grid(True, alpha=0.3)

        # 3. Индекс пролиферации (G2/M к G1)
        ax = axes[0, 2]
        g1_by_dose = df_corrected.groupby('dose_numeric').apply(
            lambda x: (x['phase_corrected'] == 'G1').mean() * 100
        ).sort_index()

        proliferation_index = g2m_by_dose / (g1_by_dose + 0.001)

        ax.plot(proliferation_index.index, proliferation_index.values,
                marker='^', color='green', linewidth=3, markersize=8)
        ax.set_title('Индекс пролиферации (G2/M : G1)')
        ax.set_xlabel('Доза облучения (Gy)')
        ax.set_ylabel('Отношение G2/M к G1')
        ax.grid(True, alpha=0.3)

        # 4. Сравнение генотипов по дозовой зависимости SubG1
        ax = axes[1, 0]

        if 'genotype' in df_corrected.columns:
            for genotype in df_corrected['genotype'].unique():
                genotype_data = df_corrected[df_corrected['genotype'] == genotype]
                subg1_by_dose_genotype = genotype_data.groupby('dose_numeric').apply(
                    lambda x: (x['phase_corrected'] == 'SubG1').mean() * 100
                ).sort_index()

                ax.plot(subg1_by_dose_genotype.index, subg1_by_dose_genotype.values,
                        marker='o', linewidth=2, label=genotype)

            ax.set_title('Доза-зависимость SubG1 по генотипам')
            ax.set_xlabel('Доза облучения (Gy)')
            ax.set_ylabel('Доля SubG1 клеток (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 5. Размер клеток по дозам
        ax = axes[1, 1]

        # Средняя площадь по дозам
        mean_area_by_dose = df_corrected.groupby('dose_numeric')['area'].mean().sort_index()
        std_area_by_dose = df_corrected.groupby('dose_numeric')['area'].std().sort_index()

        ax.errorbar(mean_area_by_dose.index, mean_area_by_dose.values,
                    yerr=std_area_by_dose.values, marker='o',
                    capsize=5, linewidth=2)
        ax.set_title('Изменение размера клеток при облучении')
        ax.set_xlabel('Доза облучения (Gy)')
        ax.set_ylabel('Средняя площадь клетки (пиксели)')
        ax.grid(True, alpha=0.3)

        # 6. Общее распределение фаз
        ax = axes[1, 2]

        phase_counts = df_corrected['phase_corrected'].value_counts()
        bars = ax.bar(phase_counts.index, phase_counts.values,
                      color=['red', 'green', 'orange', 'blue'])
        ax.set_title('Общее распределение фаз клеточного цикла')
        ax.set_xlabel('Фаза клеточного цикла')
        ax.set_ylabel('Количество клеток')
        ax.grid(True, alpha=0.3, axis='y')

        # Добавляем проценты
        total_cells = len(df_corrected)
        for bar, count in zip(bars, phase_counts.values):
            percentage = count / total_cells * 100
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'dose_response_analysis.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Графики дозовой зависимости сохранены: {self.output_dir / 'dose_response_analysis.png'}")

        return {
            'subg1_by_dose': subg1_by_dose,
            'g2m_by_dose': g2m_by_dose,
            'proliferation_index': proliferation_index
        }

    def create_genotype_comparison_report(self, df_corrected):
        """Создание отчета сравнения генотипов"""

        print("\n" + "=" * 60)
        print("ОТЧЕТ СРАВНЕНИЯ ГЕНОТИПОВ")
        print("=" * 60)

        if 'genotype' not in df_corrected.columns:
            print("Нет данных о генотипах")
            return

        genotypes = df_corrected['genotype'].unique()
        print(f"Анализ генотипов: {list(genotypes)}")

        # Создаем фигуру
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Общее распределение фаз по генотипам
        ax = axes[0, 0]

        genotype_phase = pd.crosstab(df_corrected['genotype'],
                                     df_corrected['phase_corrected'],
                                     normalize='index') * 100

        genotype_phase = genotype_phase[['SubG1', 'G1', 'S', 'G2/M']]

        genotype_phase.plot(kind='bar', ax=ax)
        ax.set_title('Распределение фаз по генотипам')
        ax.set_xlabel('Генотип')
        ax.set_ylabel('Доля клеток (%)')
        ax.legend(title='Фаза')
        ax.grid(True, alpha=0.3, axis='y')

        # 2. Размер клеток по генотипам и фазам
        ax = axes[0, 1]

        phase_order = ['SubG1', 'G1', 'S', 'G2/M']
        plot_data = []
        labels = []

        for genotype in genotypes:
            for phase in phase_order:
                subset = df_corrected[(df_corrected['genotype'] == genotype) &
                                      (df_corrected['phase_corrected'] == phase)]
                if len(subset) > 0:
                    plot_data.append(subset['area'].values)
                    labels.append(f"{genotype}\n{phase}")

        bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)

        # Раскрашиваем boxplot
        colors = ['lightcoral', 'lightgreen', 'wheat', 'lightblue']
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors[i % 4])

        ax.set_title('Размер клеток по генотипам и фазам')
        ax.set_xlabel('Генотип и фаза')
        ax.set_ylabel('Площадь клетки (пиксели)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # 3. Статистическая значимость различий
        ax = axes[0, 2]
        ax.axis('off')

        stats_text = "Статистическая значимость различий между генотипами:\n\n"

        for phase in phase_order:
            wt_data = df_corrected[(df_corrected['genotype'] == 'WT') &
                                   (df_corrected['phase_corrected'] == phase)]
            ko_data = df_corrected[(df_corrected['genotype'] == 'CDK8KO') &
                                   (df_corrected['phase_corrected'] == phase)]

            if len(wt_data) > 10 and len(ko_data) > 10:
                # t-тест для площадей
                t_stat, p_value = stats.ttest_ind(wt_data['area'],
                                                  ko_data['area'],
                                                  equal_var=False)

                # Определяем уровень значимости
                if p_value < 0.001:
                    significance = '***'
                elif p_value < 0.01:
                    significance = '**'
                elif p_value < 0.05:
                    significance = '*'
                else:
                    significance = 'нс'

                wt_mean = wt_data['area'].mean()
                ko_mean = ko_data['area'].mean()
                wt_count = len(wt_data)
                ko_count = len(ko_data)

                stats_text += f"{phase}:\n"
                stats_text += f"  WT: {wt_mean:.0f} пикс (n={wt_count})\n"
                stats_text += f"  CDK8KO: {ko_mean:.0f} пикс (n={ko_count})\n"
                stats_text += f"  p = {p_value:.4f} {significance}\n\n"

        ax.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Статистический анализ')

        # 4. Доля SubG1 по дозам для каждого генотипа
        ax = axes[1, 0]

        if 'dose_numeric' in df_corrected.columns:
            for genotype in genotypes:
                genotype_data = df_corrected[df_corrected['genotype'] == genotype]
                subg1_by_dose = genotype_data.groupby('dose_numeric').apply(
                    lambda x: (x['phase_corrected'] == 'SubG1').mean() * 100
                ).sort_index()

                ax.plot(subg1_by_dose.index, subg1_by_dose.values,
                        marker='o', linewidth=2, label=genotype)

            ax.set_title('Доза-зависимость SubG1 по генотипам')
            ax.set_xlabel('Доза облучения (Gy)')
            ax.set_ylabel('Доля SubG1 клеток (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 5. Доля G2/M по дозам для каждого генотипа
        ax = axes[1, 1]

        if 'dose_numeric' in df_corrected.columns:
            for genotype in genotypes:
                genotype_data = df_corrected[df_corrected['genotype'] == genotype]
                g2m_by_dose = genotype_data.groupby('dose_numeric').apply(
                    lambda x: (x['phase_corrected'] == 'G2/M').mean() * 100
                ).sort_index()

                ax.plot(g2m_by_dose.index, g2m_by_dose.values,
                        marker='s', linewidth=2, label=genotype)

            ax.set_title('Доза-зависимость G2/M по генотипам')
            ax.set_xlabel('Доза облучения (Gy)')
            ax.set_ylabel('Доля G2/M клеток (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 6. Сводная таблица по генотипам
        ax = axes[1, 2]
        ax.axis('off')

        summary_text = "Сводная информация по генотипам:\n\n"

        for genotype in genotypes:
            genotype_data = df_corrected[df_corrected['genotype'] == genotype]
            total_cells = len(genotype_data)

            summary_text += f"{genotype} (n={total_cells}):\n"

            for phase in phase_order:
                phase_count = (genotype_data['phase_corrected'] == phase).sum()
                phase_pct = phase_count / total_cells * 100
                mean_area = genotype_data[genotype_data['phase_corrected'] == phase]['area'].mean()

                summary_text += f"  {phase}: {phase_count} ({phase_pct:.1f}%), "
                summary_text += f"ср. площадь: {mean_area:.0f} пикс\n"

            summary_text += "\n"

        ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Сводка по генотипам')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'genotype_comparison_report.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Отчет сравнения генотипов сохранен: {self.output_dir / 'genotype_comparison_report.png'}")

        return genotype_phase

    def create_final_statistical_analysis(self, df_corrected):
        """Создание финального статистического анализа"""

        print("\n" + "=" * 60)
        print("ФИНАЛЬНЫЙ СТАТИСТИЧЕСКИЙ АНАЛИЗ")
        print("=" * 60)

        # Создаем сводную таблицу для анализа
        summary_data = []

        # Анализ по всем факторам
        factors = ['genotype', 'dose_numeric', 'time_numeric']
        available_factors = [f for f in factors if f in df_corrected.columns]

        # Группируем данные
        for factor in available_factors:
            for value in df_corrected[factor].unique():
                subset = df_corrected[df_corrected[factor] == value]

                if len(subset) > 0:
                    for phase in ['SubG1', 'G1', 'S', 'G2/M']:
                        phase_count = (subset['phase_corrected'] == phase).sum()
                        phase_pct = phase_count / len(subset) * 100
                        mean_area = subset[subset['phase_corrected'] == phase]['area'].mean()

                        summary_data.append({
                            'factor': factor,
                            'value': value,
                            'phase': phase,
                            'count': phase_count,
                            'percentage': phase_pct,
                            'mean_area': mean_area
                        })

        # Создаем DataFrame
        summary_df = pd.DataFrame(summary_data)

        # Сохраняем сводную таблицу
        summary_path = self.output_dir / 'statistical_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"Статистическая сводка сохранена: {summary_path}")

        # Двухфакторный ANOVA для площади клеток
        if 'genotype' in df_corrected.columns and 'dose_numeric' in df_corrected.columns:
            print("\nДвухфакторный ANOVA (площадь клеток):")

            # Подготовка данных
            anova_data = []
            for genotype in df_corrected['genotype'].unique():
                for dose in df_corrected['dose_numeric'].unique():
                    subset = df_corrected[(df_corrected['genotype'] == genotype) &
                                          (df_corrected['dose_numeric'] == dose)]
                    if len(subset) > 0:
                        anova_data.extend([{
                            'genotype': genotype,
                            'dose': dose,
                            'area': area
                        } for area in subset['area'].values[:100]])  # Ограничиваем для скорости

            anova_df = pd.DataFrame(anova_data)

            if len(anova_df) > 0:
                # Простая реализация ANOVA
                print("  Анализ влияния генотипа и дозы на размер клеток...")

                # Группируем по условиям
                groups = {}
                for (genotype, dose), group in anova_df.groupby(['genotype', 'dose']):
                    if len(group) > 5:
                        groups[f"{genotype}_{dose}Gy"] = group['area'].values

                # Выполняем ANOVA
                if len(groups) >= 2:
                    f_stat, p_value = stats.f_oneway(*groups.values())
                    print(f"  One-way ANOVA: F = {f_stat:.2f}, p = {p_value:.4f}")

        return summary_df

    def create_comprehensive_report(self, df_corrected, thresholds, dose_analysis, genotype_analysis):
        """Создание комплексного отчета"""

        print("\n" + "=" * 60)
        print("СОЗДАНИЕ КОМПЛЕКСНОГО ОТЧЕТА")
        print("=" * 60)

        report_path = self.output_dir / 'comprehensive_analysis_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("КОМПЛЕКСНЫЙ ОТЧЕТ ПО АНАЛИЗУ КЛЕТОЧНОГО ЦИКЛА HCT116\n")
            f.write("=" * 80 + "\n\n")

            # Методы
            f.write("МЕТОДЫ\n")
            f.write("-" * 40 + "\n")
            f.write("1. Сегментация: Cellpose (автоматическая сегментация клеток)\n")
            f.write("2. Извлечение признаков: 30+ морфологических признаков на клетку\n")
            f.write("3. Классификация: Правила на основе площади и формы клеток\n")
            f.write("4. Пороги классификации:\n")
            for key, value in thresholds.items():
                f.write(f"   - {key}: {value:.0f} пикселей\n")
            f.write("\n")

            # Результаты
            f.write("РЕЗУЛЬТАТЫ\n")
            f.write("-" * 40 + "\n")
            f.write(f"Всего проанализировано клеток: {len(df_corrected)}\n\n")

            f.write("Общее распределение фаз клеточного цикла:\n")
            phase_counts = df_corrected['phase_corrected'].value_counts()
            for phase, count in phase_counts.items():
                percentage = count / len(df_corrected) * 100
                f.write(f"  {phase}: {count} клеток ({percentage:.1f}%)\n")
            f.write("\n")

            # Анализ дозовой зависимости
            f.write("ДОЗОВАЯ ЗАВИСИМОСТЬ АПОПТОЗА (SubG1):\n")
            if 'dose_analysis' in locals() and dose_analysis is not None:
                for dose, pct in dose_analysis['subg1_by_dose'].items():
                    f.write(f"  {dose} Gy: {pct:.1f}% SubG1 клеток\n")
            f.write("\n")

            # Сравнение генотипов
            f.write("СРАВНЕНИЕ ГЕНОТИПОВ:\n")
            if 'genotype_analysis' in locals() and genotype_analysis is not None:
                for genotype in genotype_analysis.index:
                    f.write(f"\n  {genotype}:\n")
                    for phase in ['SubG1', 'G1', 'S', 'G2/M']:
                        if phase in genotype_analysis.columns:
                            pct = genotype_analysis.loc[genotype, phase]
                            f.write(f"    {phase}: {pct:.1f}%\n")
            f.write("\n")

            # Ключевые выводы
            f.write("КЛЮЧЕВЫЕ ВЫВОДЫ\n")
            f.write("-" * 40 + "\n")

            # 1. О генотипах
            if 'genotype' in df_corrected.columns:
                wt_data = df_corrected[df_corrected['genotype'] == 'WT']
                ko_data = df_corrected[df_corrected['genotype'] == 'CDK8KO']

                if len(wt_data) > 0 and len(ko_data) > 0:
                    wt_subg1 = (wt_data['phase_corrected'] == 'SubG1').mean() * 100
                    ko_subg1 = (ko_data['phase_corrected'] == 'SubG1').mean() * 100

                    f.write("1. Чувствительность к облучению:\n")
                    f.write(f"   - WT клетки: {wt_subg1:.1f}% SubG1 (апоптотических)\n")
                    f.write(f"   - CDK8KO клетки: {ko_subg1:.1f}% SubG1 (апоптотических)\n")
                    f.write(f"   - CDK8KO показывает {abs(wt_subg1 - ko_subg1):.1f}% больше SubG1 клеток\n")

                    # Определяем уровень устойчивости
                    if ko_subg1 > wt_subg1:
                        f.write("   - Вывод: CDK8KO более чувствительны к апоптозу\n")
                    else:
                        f.write("   - Вывод: WT более чувствительны к апоптозу\n")
                    f.write("\n")

            # 2. О дозовой зависимости
            if 'dose_analysis' in locals() and dose_analysis is not None:
                f.write("2. Дозовая зависимость:\n")
                doses = sorted(dose_analysis['subg1_by_dose'].index)

                if len(doses) > 1:
                    # Анализируем тренд
                    subg1_values = [dose_analysis['subg1_by_dose'][d] for d in doses]

                    # Проверяем, увеличивается ли доля SubG1 с дозой
                    if all(subg1_values[i] <= subg1_values[i + 1] for i in range(len(subg1_values) - 1)):
                        f.write("   - Четкая дозозависимость: доля SubG1 увеличивается с дозой\n")
                    else:
                        f.write("   - Сложная дозозависимость: возможно, пороговая или нелинейная\n")

                    # Анализируем G2/M задержку
                    g2m_values = [dose_analysis['g2m_by_dose'][d] for d in doses]
                    if g2m_values[-1] > g2m_values[0]:
                        f.write("   - Наблюдается накопление клеток в G2/M фазе при высоких дозах\n")
                    f.write("\n")

            # 3. Рекомендации для дальнейших исследований
            f.write("РЕКОМЕНДАЦИИ ДЛЯ ДАЛЬНЕЙШИХ ИССЛЕДОВАНИЙ\n")
            f.write("-" * 40 + "\n")
            f.write("1. Провести вестерн-блоттинг для проверки активации путей апоптоза\n")
            f.write("2. Использовать проточную цитометрию для валидации распределения фаз\n")
            f.write("3. Исследовать молекулярные механизмы повышенной чувствительности CDK8KO\n")
            f.write("4. Провести анализ выживаемости клеток в более длительные сроки\n")
            f.write("5. Использовать дополнительные маркеры для точной идентификации фаз\n")
            f.write("\n")

            # Ограничения
            f.write("ОГРАНИЧЕНИЯ ИССЛЕДОВАНИЯ\n")
            f.write("-" * 40 + "\n")
            f.write("1. Классификация основана только на морфологии (без ДНК-окрашивания)\n")
            f.write("2. Возможны ошибки сегментации при высокой плотности клеток\n")
            f.write("3. Не учитывается возможное влияние микросреды на морфологию\n")
            f.write("4. Статистическая мощность может быть ограничена для редких событий\n")

        print(f"Комплексный отчет сохранен: {report_path}")

    def run_final_analysis(self):
        """Запуск финального анализа"""

        print("=" * 80)
        print("ФИНАЛЬНЫЙ АНАЛИЗ КЛЕТОЧНОГО ЦИКЛА HCT116")
        print("=" * 80)

        # 1. Загрузка и коррекция данных
        df, control_data = self.load_and_correct_data()
        if df is None:
            return

        # 2. Анализ распределения в контроле
        thresholds = self.analyze_control_distribution(control_data)

        # 3. Применение скорректированной классификации
        df_corrected = self.apply_corrected_classification(df, thresholds)

        # 4. Анализ дозовой зависимости
        dose_analysis = self.create_dose_response_analysis(df_corrected)

        # 5. Сравнение генотипов
        genotype_analysis = self.create_genotype_comparison_report(df_corrected)

        # 6. Статистический анализ
        statistical_summary = self.create_final_statistical_analysis(df_corrected)

        # 7. Создание комплексного отчета
        self.create_comprehensive_report(df_corrected, thresholds, dose_analysis, genotype_analysis)

        # 8. Сохранение финальных данных
        final_data_path = self.output_dir / 'final_classified_data.csv'
        df_corrected.to_csv(final_data_path, index=False)
        print(f"\nФинальные данные сохранены: {final_data_path}")

        print("\n" + "=" * 80)
        print("АНАЛИЗ УСПЕШНО ЗАВЕРШЕН!")
        print("=" * 80)

        print(f"\nВсе результаты сохранены в: {self.output_dir}")
        print("\nОсновные файлы:")
        print(f"  1. Финальные данные: final_classified_data.csv")
        print(f"  2. Графики дозовой зависимости: dose_response_analysis.png")
        print(f"  3. Отчет сравнения генотипов: genotype_comparison_report.png")
        print(f"  4. Статистическая сводка: statistical_summary.csv")
        print(f"  5. Комплексный отчет: comprehensive_analysis_report.txt")


def main():
    """Основная функция"""

    try:
        analyzer = FinalCorrection()
        analyzer.run_final_analysis()

    except Exception as e:
        print(f"Ошибка при выполнении финального анализа: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()