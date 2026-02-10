
import subprocess
import sys
from pathlib import Path
import time


def run_llp_analysis():
    """Запуск полного LLP анализа"""

    root = Path(__file__).parent.parent
    scripts_dir = root / "scripts"

    print("=" * 80)
    print("ЗАПУСК LLP АНАЛИЗА КЛЕТОЧНОГО ЦИКЛА")
    print("=" * 80)

    # 1. Извлечение crops
    print("\n1. ИЗВЛЕЧЕНИЕ CROPS КЛЕТОК")
    print("-" * 40)
    try:
        crops_script = scripts_dir / "35build_crops_from_masks.py"
        if crops_script.exists():
            subprocess.run([sys.executable, str(crops_script)], check=True)
            print("✓ Извлечение crops завершено")
        else:
            print("✗ Скрипт извлечения crops не найден")
            return
    except subprocess.CalledProcessError as e:
        print(f"✗ Ошибка при извлечении crops: {e}")
        return

    time.sleep(2)

    # 2. LLP обучение и предсказание
    print("\n2. LLP ОБУЧЕНИЕ И ПРЕДСКАЗАНИЕ")
    print("-" * 40)
    try:
        llp_script = scripts_dir / "30_train_predict_ci.py"
        if llp_script.exists():
            # Определяем рабочую директорию
            work_dir = root / "results" / "llp_analysis"
            work_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable, str(llp_script),
                "--work_dir", str(work_dir),
                "--epochs", "30",
                "--force_cpu", "0"  # Используем GPU если доступно
            ]

            print(f"Запуск LLP анализа в директории: {work_dir}")
            subprocess.run(cmd, check=True)
            print("✓ LLP анализ завершен")
        else:
            print("✗ LLP скрипт не найден")
            return
    except subprocess.CalledProcessError as e:
        print(f"✗ Ошибка при LLP анализе: {e}")
        return

    time.sleep(2)

    # 3. Улучшенная визуализация результатов
    print("\n3. ВИЗУАЛИЗАЦИЯ И АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("-" * 40)
    try:
        # Создаем скрипт для анализа результатов
        analysis_script = """
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

def analyze_llp_results():
    root = Path(__file__).parent.parent
    llp_dir = root / "results" / "llp_analysis"
    output_dir = llp_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем результаты
    pred_path = llp_dir / "predicted_phase_percentages.csv"
    cell_pred_path = llp_dir / "cell_predictions.csv"

    if not pred_path.exists():
        print("Файл предсказаний не найден!")
        return

    pred_df = pd.read_csv(pred_path)
    cell_pred_df = pd.read_csv(cell_pred_path) if cell_pred_path.exists() else None

    # Данные цитометра
    flow_data = pd.DataFrame([
        # 2h
        {'condition': 'Ctrl', 'concentration_uM': 0, 'time_h': 2, 'SubG1': 3.56, 'G1': 58.71, 'G2M': 37.10},
        {'condition': 'Aura', 'concentration_uM': 0.5, 'time_h': 2, 'SubG1': 2.94, 'G1': 59.23, 'G2M': 37.33},
        {'condition': 'Aura', 'concentration_uM': 1, 'time_h': 2, 'SubG1': 5.53, 'G1': 64.88, 'G2M': 29.08},
        {'condition': 'Aura', 'concentration_uM': 2, 'time_h': 2, 'SubG1': 7.53, 'G1': 64.16, 'G2M': 27.70},
        # 6h
        {'condition': 'Ctrl', 'concentration_uM': 0, 'time_h': 6, 'SubG1': 3.06, 'G1': 57.67, 'G2M': 38.70},
        {'condition': 'Aura', 'concentration_uM': 0.5, 'time_h': 6, 'SubG1': 8.48, 'G1': 58.12, 'G2M': 32.36},
        {'condition': 'Aura', 'concentration_uM': 1, 'time_h': 6, 'SubG1': 16.05, 'G1': 51.39, 'G2M': 31.10},
        {'condition': 'Aura', 'concentration_uM': 2, 'time_h': 6, 'SubG1': 21.09, 'G1': 53.52, 'G2M': 24.04},
        # 24h
        {'condition': 'Ctrl', 'concentration_uM': 0, 'time_h': 24, 'SubG1': 7.59, 'G1': 62.00, 'G2M': 29.72},
        {'condition': 'Aura', 'concentration_uM': 0.5, 'time_h': 24, 'SubG1': 21.98, 'G1': 57.08, 'G2M': 20.51},
        {'condition': 'Aura', 'concentration_uM': 1, 'time_h': 24, 'SubG1': 40.71, 'G1': 50.61, 'G2M': 8.59},
        {'condition': 'Aura', 'concentration_uM': 2, 'time_h': 24, 'SubG1': 62.65, 'G1': 28.89, 'G2M': 8.06},
    ])

    # Преобразуем LLP предсказания
    llp_comparison = pred_df.copy()
    llp_comparison['condition'] = llp_comparison['treatment'].apply(lambda x: 'Ctrl' if x == 'CTRL' else 'Aura')
    llp_comparison['concentration_uM'] = llp_comparison['concentration']
    llp_comparison['time_h'] = llp_comparison['time']

    # Сравниваем данные
    phases = ['SubG1', 'G1', 'G2M']

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for i, time_h in enumerate([2, 6, 24]):
        # Данные для текущего времени
        llp_time = llp_comparison[llp_comparison['time_h'] == time_h]
        flow_time = flow_data[flow_data['time_h'] == time_h]

        # Бар-графики сравнения
        ax = axes[i, 0]
        x = np.arange(len(llp_time))
        width = 0.35

        for j, phase in enumerate(phases):
            llp_vals = llp_time[phase].values
            flow_vals = flow_time[phase].values

            ax.bar(x - width/2 + j*width/len(phases), llp_vals, width/len(phases), 
                  label=f'LLP {phase}', alpha=0.7)
            ax.bar(x + width/2 + j*width/len(phases), flow_vals, width/len(phases), 
                  label=f'Flow {phase}', alpha=0.3, hatch='//')

        ax.set_xlabel('Условия')
        ax.set_ylabel('Процент (%)')
        ax.set_title(f'Сравнение методов - {time_h}ч')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{'Ctrl' if c==0 else f'{c}µM'}" for c in llp_time['concentration_uM']])
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Scatter plot корреляции
        ax = axes[i, 1]
        for phase in phases:
            llp_phase = llp_time[phase].values
            flow_phase = flow_time[phase].values

            ax.scatter(flow_phase, llp_phase, label=phase, alpha=0.7, s=50)

        # Линия идеального соответствия
        min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

        ax.set_xlabel('Проточная цитометрия (%)')
        ax.set_ylabel('LLP предсказание (%)')
        ax.set_title(f'Корреляция - {time_h}ч')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

        # Различия
        ax = axes[i, 2]
        differences = []
        labels = []
        colors = []

        for conc in sorted(llp_time['concentration_uM'].unique()):
            for phase in phases:
                llp_val = llp_time[llp_time['concentration_uM'] == conc][phase].values[0]
                flow_val = flow_time[flow_time['concentration_uM'] == conc][phase].values[0]
                diff = llp_val - flow_val

                differences.append(diff)
                labels.append(f'{conc}µM\\n{phase}')
                colors.append('red' if diff > 0 else 'blue')

        bars = ax.bar(range(len(differences)), differences, color=colors, alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax.set_xlabel('Условия и фазы')
        ax.set_ylabel('Разница (LLP - Flow)')
        ax.set_title(f'Различия - {time_h}ч')
        ax.set_xticks(range(len(differences)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'llp_vs_flow_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Анализ клеточного уровня
    if cell_pred_df is not None and 'phase_pred' in cell_pred_df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Распределение фаз по условиям
        phase_counts = cell_pred_df['phase_pred'].value_counts()
        axes[0, 0].pie(phase_counts.values, labels=phase_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Общее распределение фаз')

        # Распределение фаз по времени
        phase_by_time = pd.crosstab(cell_pred_df['time'], cell_pred_df['phase_pred'], normalize='index') * 100
        phase_by_time.plot(kind='bar', ax=axes[0, 1], stacked=True)
        axes[0, 1].set_xlabel('Время (ч)')
        axes[0, 1].set_ylabel('Процент (%)')
        axes[0, 1].set_title('Фазы по времени')
        axes[0, 1].legend(title='Фаза')

        # Распределение фаз по концентрации (для Aura)
        aura_cells = cell_pred_df[cell_pred_df['treatment'] == 'AURA']
        if len(aura_cells) > 0:
            phase_by_conc = pd.crosstab(aura_cells['concentration'], aura_cells['phase_pred'], normalize='index') * 100
            phase_by_conc.plot(kind='bar', ax=axes[1, 0], stacked=True)
            axes[1, 0].set_xlabel('Концентрация (µM)')
            axes[1, 0].set_ylabel('Процент (%)')
            axes[1, 0].set_title('Фазы по концентрации (Aura)')
            axes[1, 0].legend(title='Фаза')

        # Вероятности фаз
        if 'p_SubG1' in cell_pred_df.columns:
            phase_probs = cell_pred_df[['p_SubG1', 'p_G1', 'p_G2M']]
            axes[1, 1].boxplot([phase_probs['p_SubG1'], phase_probs['p_G1'], phase_probs['p_G2M']], 
                              labels=['SubG1', 'G1', 'G2M'])
            axes[1, 1].set_ylabel('Вероятность')
            axes[1, 1].set_title('Распределение вероятностей фаз')
            axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / 'cell_level_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Анализ сохранен в: {output_dir}")

if __name__ == "__main__":
    analyze_llp_results()
"""

        # Сохраняем и запускаем скрипт анализа
        analysis_path = scripts_dir / "40_analyze_llp_results.py"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write(analysis_script)

        subprocess.run([sys.executable, str(analysis_path)], check=True)
        print("✓ Анализ результатов завершен")

    except Exception as e:
        print(f"✗ Ошибка при анализе: {e}")

    # Вывод результатов
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)

    results_dir = root / "results"

    print("\nСОЗДАННЫЕ ФАЙЛЫ:")
    print("-" * 40)

    key_files = [
        ("📊 Маски клеток", "segmentation_fixed/data/"),
        ("🌾 Crops клеток", "cell_crops/crops/"),
        ("📋 Manifest клеток", "cell_crops/manifest_cells.csv"),
        ("🤖 LLP предсказания", "llp_analysis/predicted_phase_percentages.csv"),
        ("🔬 Предсказания клеток", "llp_analysis/cell_predictions.csv"),
        ("📈 Графики сравнения", "llp_analysis/analysis/llp_vs_flow_comparison.png"),
        ("📊 Анализ клеток", "llp_analysis/analysis/cell_level_analysis.png")
    ]

    for desc, rel_path in key_files:
        file_path = results_dir / rel_path
        if file_path.exists():
            print(f"✓ {desc}: {file_path}")
        else:
            print(f"✗ {desc}: не найден")

    print("\nРЕКОМЕНДАЦИИ:")
    print("-" * 40)
    print("1. Основные результаты в: results/llp_analysis/")
    print("2. Сравнение с цитометрией: llp_vs_flow_comparison.png")
    print("3. Для детального анализа используйте Excel файлы")
    print("4. Проверьте графики в папке analysis/")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_llp_analysis()