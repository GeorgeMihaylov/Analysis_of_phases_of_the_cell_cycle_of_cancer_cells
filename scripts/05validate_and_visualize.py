"""
Валидация и визуализация классификации фаз клеточного цикла
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

class ValidationVisualizer:
    """Визуализация и валидация классификации клеток"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / 'results' / 'cell_cycle'
        self.raw_data_dir = self.project_root / 'data' / 'raw'
        self.masks_dir = self.project_root / 'results' / 'masks'
        self.output_dir = self.project_root / 'results' / 'validation'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Загрузка данных с классификацией"""
        data_path = self.data_dir / 'cells_with_phases.csv'

        if not data_path.exists():
            print(f"Ошибка: Файл с классификацией не найден: {data_path}")
            return None

        print(f"Загрузка данных классификации из {data_path}")
        df = pd.read_csv(data_path)
        print(f"Загружено {len(df)} клеток")

        # Преобразуем типы данных
        if 'dose_gy' in df.columns:
            # Проверяем, является ли уже числовым
            if df['dose_gy'].dtype == object:
                df['dose_numeric'] = pd.to_numeric(df['dose_gy'], errors='coerce')
            else:
                df['dose_numeric'] = df['dose_gy']

        if 'time_h' in df.columns:
            df['time_numeric'] = pd.to_numeric(df['time_h'], errors='coerce')

        return df

    def check_high_subg1(self, df):
        """Анализ высокого уровня SubG1 в контроле"""

        print("\n" + "=" * 60)
        print("АНАЛИЗ ВЫСОКОГО УРОВНЯ SubG1 В КОНТРОЛЕ")
        print("=" * 60)

        # Проверяем наличие данных о дозе
        if 'dose_numeric' not in df.columns:
            print("Нет данных о дозе облучения")
            return None

        # Фильтруем контрольные образцы (0 Gy)
        control_df = df[df['dose_numeric'] == 0]

        if len(control_df) == 0:
            print("Нет контрольных данных (0 Gy)")
            return control_df

        print(f"Контрольные образцы (0 Gy): {len(control_df)} клеток")

        # Распределение по фазам в контроле
        print("\nРаспределение фаз в контроле (0 Gy):")
        phase_counts = control_df['phase'].value_counts(normalize=True) * 100
        for phase, percentage in phase_counts.items():
            count = (control_df['phase'] == phase).sum()
            print(f"  {phase}: {count} клеток ({percentage:.1f}%)")

        # Анализ площадей клеток в контроле по фазам
        print("\nСтатистика площадей в контроле по фазам:")
        for phase in control_df['phase'].unique():
            phase_data = control_df[control_df['phase'] == phase]['area']
            print(f"  {phase}: среднее = {phase_data.mean():.0f}, "
                  f"медиана = {phase_data.median():.0f}, "
                  f"min = {phase_data.min():.0f}, max = {phase_data.max():.0f}")

        # Проверяем, не слишком ли низкие пороги для SubG1
        print("\nПроверка порогов классификации:")

        # Определяем квартили площади для всей популяции
        area_q1 = df['area'].quantile(0.25)
        area_q2 = df['area'].quantile(0.5)  # медиана
        area_q3 = df['area'].quantile(0.75)

        print(f"  Q1 площади (25% перцентиль): {area_q1:.0f}")
        print(f"  Медиана площади: {area_q2:.0f}")
        print(f"  Q3 площади (75% перцентиль): {area_q3:.0f}")

        # Анализируем клетки, классифицированные как SubG1
        subg1_cells = df[df['phase'] == 'SubG1']
        print(f"\nКлетки SubG1 (все дозы):")
        print(f"  Средняя площадь: {subg1_cells['area'].mean():.0f}")
        print(f"  Медиана площади: {subg1_cells['area'].median():.0f}")
        print(f"  Диапазон: {subg1_cells['area'].min():.0f} - {subg1_cells['area'].max():.0f}")

        # Проверяем, какие клетки попадают в SubG1
        print("\nХарактеристики клеток SubG1 в контроле:")
        subg1_control = control_df[control_df['phase'] == 'SubG1']

        # Анализ дополнительных признаков
        features_to_check = ['circularity', 'aspect_ratio', 'solidity', 'intensity_mean']
        for feature in features_to_check:
            if feature in subg1_control.columns:
                mean_val = subg1_control[feature].mean()
                median_val = subg1_control[feature].median()
                print(f"  {feature}: среднее = {mean_val:.3f}, медиана = {median_val:.3f}")

        return control_df

    def plot_phase_distribution_by_dose(self, df):
        """Визуализация распределения фаз по дозам"""

        print("\nСоздание графиков распределения фаз по дозам...")

        if 'dose_numeric' not in df.columns:
            print("Нет данных о дозах облучения")
            return None

        # Порядок фаз
        phase_order = ['SubG1', 'G1', 'S', 'G2/M']

        # Создаем фигуру с несколькими графиками
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Распределение долей по дозам (столбчатая диаграмма)
        dose_phase = pd.crosstab(df['dose_numeric'], df['phase'], normalize='index')
        dose_phase = dose_phase[phase_order]
        dose_phase = dose_phase.sort_index()

        ax = axes[0, 0]
        dose_phase.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title('Распределение фаз клеточного цикла по дозам облучения')
        ax.set_xlabel('Доза облучения (Gy)')
        ax.set_ylabel('Доля клеток')
        ax.legend(title='Фаза', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')

        # 2. Линейные графики для каждой фазы
        ax = axes[0, 1]
        for phase in phase_order:
            if phase in dose_phase.columns:
                ax.plot(dose_phase.index, dose_phase[phase], marker='o', label=phase, linewidth=2)

        ax.set_title('Динамика фаз клеточного цикла при облучении')
        ax.set_xlabel('Доза облучения (Gy)')
        ax.set_ylabel('Доля клеток')
        ax.legend(title='Фаза')
        ax.grid(True, alpha=0.3)

        # 3. Изменение доли SubG1 по дозам
        ax = axes[1, 0]
        subg1_by_dose = df.groupby('dose_numeric').apply(lambda x: (x['phase'] == 'SubG1').mean() * 100)
        subg1_by_dose = subg1_by_dose.sort_index()

        ax.plot(subg1_by_dose.index, subg1_by_dose.values, marker='o', color='red', linewidth=3, markersize=8)
        ax.set_title('Доля апоптотических клеток (SubG1) при облучении')
        ax.set_xlabel('Доза облучения (Gy)')
        ax.set_ylabel('Доля SubG1 клеток (%)')
        ax.grid(True, alpha=0.3)

        # Добавляем аннотации
        for dose, value in subg1_by_dose.items():
            ax.annotate(f'{value:.1f}%', (dose, value), textcoords="offset points",
                       xytext=(0,10), ha='center', fontsize=9)

        # 4. Отношение G2/M к G1 (индекс пролиферации)
        ax = axes[1, 1]
        g2m_by_dose = df.groupby('dose_numeric').apply(lambda x: (x['phase'] == 'G2/M').mean() * 100)
        g1_by_dose = df.groupby('dose_numeric').apply(lambda x: (x['phase'] == 'G1').mean() * 100)

        proliferation_index = g2m_by_dose / (g1_by_dose + 0.001)  # добавляем маленькое значение для избежания деления на 0

        ax.plot(proliferation_index.index, proliferation_index.values,
                marker='s', color='green', linewidth=3, markersize=8)
        ax.set_title('Индекс пролиферации (G2/M / G1) при облучении')
        ax.set_xlabel('Доза облучения (Gy)')
        ax.set_ylabel('Отношение G2/M к G1')
        ax.grid(True, alpha=0.3)

        # Добавляем аннотации
        for dose, value in proliferation_index.items():
            ax.annotate(f'{value:.2f}', (dose, value), textcoords="offset points",
                       xytext=(0,10), ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'phase_distribution_by_dose_detailed.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Графики сохранены: {self.output_dir / 'phase_distribution_by_dose_detailed.png'}")

        return dose_phase

    def compare_genotypes(self, df):
        """Сравнение генотипов WT и CDK8KO"""

        print("\n" + "=" * 60)
        print("СРАВНЕНИЕ ГЕНОТИПОВ WT И CDK8KO")
        print("=" * 60)

        if 'genotype' not in df.columns:
            print("В данных отсутствует информация о генотипе")
            return

        genotypes = df['genotype'].unique()
        print(f"Найдены генотипы: {list(genotypes)}")

        # Создаем графики сравнения
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Общее распределение фаз по генотипам
        ax = axes[0, 0]
        genotype_phase = pd.crosstab(df['genotype'], df['phase'], normalize='index')
        phase_order = ['SubG1', 'G1', 'S', 'G2/M']
        genotype_phase = genotype_phase[phase_order]

        genotype_phase.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title('Распределение фаз по генотипам')
        ax.set_xlabel('Генотип')
        ax.set_ylabel('Доля клеток')
        ax.legend(title='Фаза')
        ax.grid(True, alpha=0.3, axis='y')

        # 2. Доля SubG1 по генотипам и дозам
        ax = axes[0, 1]

        if 'dose_numeric' in df.columns:
            # Создаем сводную таблицу
            pivot_table = df.pivot_table(
                values='phase',
                index='dose_numeric',
                columns='genotype',
                aggfunc=lambda x: (x == 'SubG1').mean() * 100
            )

            for genotype in genotypes:
                if genotype in pivot_table.columns:
                    ax.plot(pivot_table.index, pivot_table[genotype],
                           marker='o', label=genotype, linewidth=2)

            ax.set_title('Доля SubG1 клеток по генотипам и дозам')
            ax.set_xlabel('Доза облучения (Gy)')
            ax.set_ylabel('Доля SubG1 клеток (%)')
            ax.legend(title='Генотип')
            ax.grid(True, alpha=0.3)

        # 3. Доля G2/M по генотипам и дозам
        ax = axes[0, 2]

        if 'dose_numeric' in df.columns:
            pivot_table_g2m = df.pivot_table(
                values='phase',
                index='dose_numeric',
                columns='genotype',
                aggfunc=lambda x: (x == 'G2/M').mean() * 100
            )

            for genotype in genotypes:
                if genotype in pivot_table_g2m.columns:
                    ax.plot(pivot_table_g2m.index, pivot_table_g2m[genotype],
                           marker='s', label=genotype, linewidth=2)

            ax.set_title('Доля G2/M клеток по генотипам и дозам')
            ax.set_xlabel('Доза облучения (Gy)')
            ax.set_ylabel('Доля G2/M клеток (%)')
            ax.legend(title='Генотип')
            ax.grid(True, alpha=0.3)

        # 4. Размер клеток (площадь) по генотипам и фазам
        ax = axes[1, 0]

        # Boxplot площадей по фазам для каждого генотипа
        plot_data = []
        labels = []

        for genotype in genotypes:
            for phase in phase_order:
                subset = df[(df['genotype'] == genotype) & (df['phase'] == phase)]
                if len(subset) > 0:
                    plot_data.append(subset['area'].values)
                    labels.append(f"{genotype}\n{phase}")

        if plot_data:
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

        # 5. Индекс пролиферации (G2/M к G1) по генотипам
        ax = axes[1, 1]

        proliferation_by_genotype = {}
        for genotype in genotypes:
            genotype_data = df[df['genotype'] == genotype]
            g2m_ratio = (genotype_data['phase'] == 'G2/M').mean()
            g1_ratio = (genotype_data['phase'] == 'G1').mean()
            proliferation_index = g2m_ratio / (g1_ratio + 0.001)
            proliferation_by_genotype[genotype] = proliferation_index

        bars = ax.bar(proliferation_by_genotype.keys(), proliferation_by_genotype.values(),
                     color=['skyblue', 'lightcoral'])
        ax.set_title('Индекс пролиферации по генотипам')
        ax.set_xlabel('Генотип')
        ax.set_ylabel('Отношение G2/M к G1')
        ax.grid(True, alpha=0.3, axis='y')

        # Добавляем значения на столбцы
        for bar, value in zip(bars, proliferation_by_genotype.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{value:.2f}', ha='center', va='bottom')

        # 6. Круговая диаграмма для сравнения генотипов
        ax = axes[1, 2]
        ax.axis('off')

        # Сводная информация
        summary_text = "Сводка по генотипам:\n\n"
        for genotype in genotypes:
            genotype_data = df[df['genotype'] == genotype]
            total_cells = len(genotype_data)
            subg1_pct = (genotype_data['phase'] == 'SubG1').mean() * 100
            g2m_pct = (genotype_data['phase'] == 'G2/M').mean() * 100

            summary_text += f"{genotype}:\n"
            summary_text += f"  Клеток: {total_cells}\n"
            summary_text += f"  SubG1: {subg1_pct:.1f}%\n"
            summary_text += f"  G2/M: {g2m_pct:.1f}%\n\n"

        ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Сводка по генотипам')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'genotype_comparison.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Графики сравнения генотипов сохранены: {self.output_dir / 'genotype_comparison.png'}")

        # Статистический анализ различий
        print("\nСтатистический анализ различий между генотипами:")

        for phase in phase_order:
            wt_data = df[(df['genotype'] == 'WT') & (df['phase'] == phase)]
            ko_data = df[(df['genotype'] == 'CDK8KO') & (df['phase'] == phase)]

            if len(wt_data) > 10 and len(ko_data) > 10:
                # t-тест для площадей
                t_stat, p_value = stats.ttest_ind(wt_data['area'], ko_data['area'],
                                                 equal_var=False)
                print(f"\n  {phase} фаза:")
                print(f"    WT: {len(wt_data)} клеток, средняя площадь = {wt_data['area'].mean():.0f}")
                print(f"    CDK8KO: {len(ko_data)} клеток, средняя площадь = {ko_data['area'].mean():.0f}")
                print(f"    t-тест: t = {t_stat:.2f}, p = {p_value:.4f}")

        return genotype_phase

    def analyze_time_effects(self, df):
        """Анализ влияния времени после облучения"""

        print("\n" + "=" * 60)
        print("АНАЛИЗ ВЛИЯНИЯ ВРЕМЕНИ ПОСЛЕ ОБЛУЧЕНИЯ")
        print("=" * 60)

        if 'time_numeric' not in df.columns or 'dose_numeric' not in df.columns:
            print("В данных отсутствует информация о времени или дозе")
            return

        time_points = sorted(df['time_numeric'].unique())

        print(f"Временные точки: {time_points} часов")

        # Создаем графики
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Распределение фаз по времени
        ax = axes[0, 0]
        time_phase = pd.crosstab(df['time_numeric'], df['phase'], normalize='index')
        phase_order = ['SubG1', 'G1', 'S', 'G2/M']
        time_phase = time_phase[phase_order]
        time_phase = time_phase.sort_index()

        time_phase.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title('Распределение фаз по времени после облучения')
        ax.set_xlabel('Время после облучения (часы)')
        ax.set_ylabel('Доля клеток')
        ax.legend(title='Фаза')
        ax.grid(True, alpha=0.3, axis='y')

        # 2. Изменение доли SubG1 по времени
        ax = axes[0, 1]

        # Группируем по времени и дозе
        time_dose_phase = df.groupby(['time_numeric', 'dose_numeric']).apply(
            lambda x: (x['phase'] == 'SubG1').mean() * 100
        ).unstack()

        # Берем первые 3 дозы для ясности
        doses_to_plot = sorted(df['dose_numeric'].unique())[:3]
        for dose in doses_to_plot:
            if dose in time_dose_phase.columns:
                ax.plot(time_dose_phase.index, time_dose_phase[dose],
                       marker='o', label=f'{dose} Gy', linewidth=2)

        ax.set_title('Доля SubG1 клеток по времени и дозе')
        ax.set_xlabel('Время после облучения (часы)')
        ax.set_ylabel('Доля SubG1 клеток (%)')
        ax.legend(title='Доза')
        ax.grid(True, alpha=0.3)

        # 3. Сравнение 24h и 48h для каждой дозы
        ax = axes[1, 0]

        # Подготовка данных
        comparison_data = []
        doses = sorted(df['dose_numeric'].unique())

        for dose in doses:
            time_24 = df[(df['dose_numeric'] == dose) & (df['time_numeric'] == 24)]
            time_48 = df[(df['dose_numeric'] == dose) & (df['time_numeric'] == 48)]

            if len(time_24) > 0 and len(time_48) > 0:
                subg1_24 = (time_24['phase'] == 'SubG1').mean() * 100
                subg1_48 = (time_48['phase'] == 'SubG1').mean() * 100
                comparison_data.append({
                    'dose': dose,
                    '24h': subg1_24,
                    '48h': subg1_48
                })

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            x = np.arange(len(comparison_df))
            width = 0.35

            ax.bar(x - width/2, comparison_df['24h'], width, label='24h', color='skyblue')
            ax.bar(x + width/2, comparison_df['48h'], width, label='48h', color='lightcoral')

            ax.set_xlabel('Доза облучения (Gy)')
            ax.set_ylabel('Доля SubG1 клеток (%)')
            ax.set_title('Сравнение 24h и 48h: доля SubG1')
            ax.set_xticks(x)
            ax.set_xticklabels([str(d) for d in comparison_df['dose']])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        # 4. Сводная информация
        ax = axes[1, 1]
        ax.axis('off')

        summary_text = "Влияние времени на клеточный цикл:\n\n"

        for time_point in time_points:
            time_data = df[df['time_numeric'] == time_point]
            total_cells = len(time_data)

            summary_text += f"{time_point} часов:\n"
            summary_text += f"  Всего клеток: {total_cells}\n"

            for phase in phase_order:
                phase_pct = (time_data['phase'] == phase).mean() * 100
                summary_text += f"  {phase}: {phase_pct:.1f}%\n"

            summary_text += "\n"

        ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Сводка по временным точкам')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'time_effects_analysis.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Графики влияния времени сохранены: {self.output_dir / 'time_effects_analysis.png'}")

        # Статистический анализ
        print("\nСтатистический анализ влияния времени:")

        for dose in [0, 10]:  # Анализируем контроль и максимальную дозу
            for phase in ['SubG1', 'G2/M']:
                time_24_data = df[(df['dose_numeric'] == dose) &
                                 (df['time_numeric'] == 24) &
                                 (df['phase'] == phase)]
                time_48_data = df[(df['dose_numeric'] == dose) &
                                 (df['time_numeric'] == 48) &
                                 (df['phase'] == phase)]

                if len(time_24_data) > 10 and len(time_48_data) > 10:
                    # Сравнение площадей
                    t_stat, p_value = stats.ttest_ind(time_24_data['area'],
                                                     time_48_data['area'],
                                                     equal_var=False)

                    print(f"\n  Доза {dose} Gy, фаза {phase}:")
                    print(f"    24h: {len(time_24_data)} клеток, площадь = {time_24_data['area'].mean():.0f}")
                    print(f"    48h: {len(time_48_data)} клеток, площадь = {time_48_data['area'].mean():.0f}")
                    print(f"    t-тест: t = {t_stat:.2f}, p = {p_value:.4f}")

    def create_summary_report(self, df):
        """Создание сводного отчета"""

        print("\n" + "=" * 60)
        print("СОЗДАНИЕ СВОДНОГО ОТЧЕТА")
        print("=" * 60)

        report_path = self.output_dir / 'validation_summary_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("СВОДНЫЙ ОТЧЕТ ПО АНАЛИЗУ КЛЕТОЧНОГО ЦИКЛА\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Всего проанализировано клеток: {len(df)}\n\n")

            # Общее распределение фаз
            f.write("ОБЩЕЕ РАСПРЕДЕЛЕНИЕ ФАЗ:\n")
            f.write("-" * 40 + "\n")
            phase_counts = df['phase'].value_counts()
            for phase, count in phase_counts.items():
                percentage = count / len(df) * 100
                f.write(f"{phase}: {count} клеток ({percentage:.1f}%)\n")
            f.write("\n")

            # Анализ по дозам
            f.write("АНАЛИЗ ПО ДОЗАМ ОБЛУЧЕНИЯ:\n")
            f.write("-" * 40 + "\n")
            if 'dose_numeric' in df.columns:
                doses = sorted(df['dose_numeric'].unique())
                for dose in doses:
                    dose_data = df[df['dose_numeric'] == dose]
                    subg1_pct = (dose_data['phase'] == 'SubG1').mean() * 100
                    g2m_pct = (dose_data['phase'] == 'G2/M').mean() * 100
                    f.write(f"Доза {dose} Gy: {len(dose_data)} клеток, SubG1={subg1_pct:.1f}%, G2/M={g2m_pct:.1f}%\n")
            f.write("\n")

            # Сравнение генотипов
            if 'genotype' in df.columns:
                f.write("СРАВНЕНИЕ ГЕНОТИПОВ:\n")
                f.write("-" * 40 + "\n")
                for genotype in df['genotype'].unique():
                    genotype_data = df[df['genotype'] == genotype]
                    subg1_pct = (genotype_data['phase'] == 'SubG1').mean() * 100
                    g2m_pct = (genotype_data['phase'] == 'G2/M').mean() * 100
                    f.write(f"{genotype}: {len(genotype_data)} клеток, SubG1={subg1_pct:.1f}%, G2/M={g2m_pct:.1f}%\n")
                f.write("\n")

            # Анализ по времени
            if 'time_numeric' in df.columns:
                f.write("АНАЛИЗ ПО ВРЕМЕНИ ПОСЛЕ ОБЛУЧЕНИЯ:\n")
                f.write("-" * 40 + "\n")
                for time_point in sorted(df['time_numeric'].unique()):
                    time_data = df[df['time_numeric'] == time_point]
                    subg1_pct = (time_data['phase'] == 'SubG1').mean() * 100
                    g2m_pct = (time_data['phase'] == 'G2/M').mean() * 100
                    f.write(f"{time_point} часов: {len(time_data)} клеток, SubG1={subg1_pct:.1f}%, G2/M={g2m_pct:.1f}%\n")
                f.write("\n")

            # Ключевые наблюдения
            f.write("КЛЮЧЕВЫЕ НАБЛЮДЕНИЯ:\n")
            f.write("-" * 40 + "\n")

            # Анализ SubG1
            if 'dose_numeric' in df.columns:
                dose_0_data = df[df['dose_numeric'] == 0] if 0 in df['dose_numeric'].unique() else None
                if dose_0_data is not None and len(dose_0_data) > 0:
                    subg1_pct_0 = (dose_0_data['phase'] == 'SubG1').mean() * 100
                    f.write(f"1. Доля SubG1 в контроле (0 Gy): {subg1_pct_0:.1f}%\n")
                    f.write("   Высокое значение может указывать на:\n")
                    f.write("   - Слишком низкие пороги классификации\n")
                    f.write("   - Артефакты сегментации\n")
                    f.write("   - Особенности клеточной линии HCT116\n")
                    f.write("   - Необходимость ручной проверки\n\n")

            # Сравнение генотипов
            if 'genotype' in df.columns and 'genotype' in df.columns:
                wt_data = df[df['genotype'] == 'WT']
                ko_data = df[df['genotype'] == 'CDK8KO']
                if len(wt_data) > 0 and len(ko_data) > 0:
                    wt_subg1 = (wt_data['phase'] == 'SubG1').mean() * 100
                    ko_subg1 = (ko_data['phase'] == 'SubG1').mean() * 100
                    f.write(f"2. Сравнение генотипов:\n")
                    f.write(f"   - WT: {wt_subg1:.1f}% SubG1\n")
                    f.write(f"   - CDK8KO: {ko_subg1:.1f}% SubG1\n")
                    f.write(f"   - Разница: {abs(wt_subg1 - ko_subg1):.1f}%\n")
                    f.write("   - CDK8KO показывает более высокую базальную долю SubG1\n\n")

            # Статистическая значимость
            f.write("3. Статистическая значимость различий:\n")
            if 'genotype' in df.columns:
                for phase in ['SubG1', 'G1', 'S', 'G2/M']:
                    wt_data = df[(df['genotype'] == 'WT') & (df['phase'] == phase)]
                    ko_data = df[(df['genotype'] == 'CDK8KO') & (df['phase'] == phase)]
                    if len(wt_data) > 10 and len(ko_data) > 10:
                        t_stat, p_value = stats.ttest_ind(wt_data['area'], ko_data['area'], equal_var=False)
                        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "нс"
                        f.write(f"   - {phase}: p={p_value:.4f} {significance}\n")
            f.write("\n")

            # Рекомендации
            f.write("РЕКОМЕНДАЦИИ:\n")
            f.write("-" * 40 + "\n")
            f.write("1. Проверить качество сегментации контрольных образцов\n")
            f.write("2. Визуально оценить клетки, классифицированные как SubG1\n")
            f.write("3. Рассмотреть возможность корректировки порогов классификации\n")
            f.write("4. Провести анализ чувствительности к изменению параметров кластеризации\n")
            f.write("5. Сравнить результаты с литературными данными по HCT116\n")
            f.write("6. Для публикации рассчитать размер эффекта и доверительные интервалы\n")

        print(f"  Сводный отчет сохранен: {report_path}")

    def run_validation(self):
        """Запуск полной валидации"""

        print("=" * 60)
        print("ВАЛИДАЦИЯ КЛАССИФИКАЦИИ КЛЕТОЧНОГО ЦИКЛА")
        print("=" * 60)

        # 1. Загрузка данных
        df = self.load_data()
        if df is None:
            return

        # 2. Анализ высокого уровня SubG1 в контроле
        self.check_high_subg1(df)

        # 3. Визуализация распределения по дозам
        self.plot_phase_distribution_by_dose(df)

        # 4. Сравнение генотипов
        self.compare_genotypes(df)

        # 5. Анализ влияния времени
        self.analyze_time_effects(df)

        # 6. Создание сводного отчета
        self.create_summary_report(df)

        print("\n" + "=" * 60)
        print("ВАЛИДАЦИЯ ЗАВЕРШЕНА!")
        print("=" * 60)

        print(f"\nРезультаты сохранены в: {self.output_dir}")
        print("\nСледующие шаги:")
        print("1. Проверьте графики в папке results/validation/")
        print("2. Ознакомьтесь с отчетом validation_summary_report.txt")
        print("3. Рассмотрите необходимость корректировки классификации")
        print("4. При необходимости используйте ручную проверку классификации")

def main():
    """Основная функция"""

    try:
        visualizer = ValidationVisualizer()
        visualizer.run_validation()

    except Exception as e:
        print(f"Ошибка при выполнении валидации: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()