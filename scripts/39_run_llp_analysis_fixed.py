# scripts/39_run_llp_analysis_fixed.py
import subprocess
import sys
from pathlib import Path
import time
import shutil


def run_llp_analysis():
    """Запуск полного LLP анализа - ИСПРАВЛЕННАЯ ВЕРСИЯ"""

    root = Path(__file__).parent.parent
    scripts_dir = root / "scripts"

    print("=" * 80)
    print("ЗАПУСК LLP АНАЛИЗА КЛЕТОЧНОГО ЦИКЛА")
    print("=" * 80)

    # Создаем рабочую директорию
    work_dir = root / "results" / "llp_analysis_v2"

    # Удаляем старую директорию, если существует
    if work_dir.exists():
        print(f"Удаляем старую директорию: {work_dir}")
        shutil.rmtree(work_dir)

    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Рабочая директория: {work_dir}")

    # 1. Извлечение crops
    print("\n1. ИЗВЛЕЧЕНИЕ CROPS КЛЕТОК")
    print("-" * 40)
    try:
        crops_script = scripts_dir / "38_build_crops_from_masks_fixed.py"
        if crops_script.exists():
            cmd = [
                sys.executable, str(crops_script),
                "--work_dir", str(work_dir)
            ]

            print(f"Запуск: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("Stderr:", result.stderr)

            print("✓ Извлечение crops завершено")
        else:
            print(f"✗ Скрипт не найден: {crops_script}")
            # Проверяем другие варианты
            alt_script = scripts_dir / "38_build_crops_from_masks.py"
            if alt_script.exists():
                print(f"Найден альтернативный скрипт: {alt_script}")
                cmd = [
                    sys.executable, str(alt_script),
                    "--work_dir", str(work_dir)
                ]
                subprocess.run(cmd, check=True)
                print("✓ Извлечение crops завершено")
            else:
                print("✗ Ни один скрипт извлечения crops не найден")
                return
    except subprocess.CalledProcessError as e:
        print(f"✗ Ошибка при извлечении crops: {e}")
        print(f"Stderr: {e.stderr}")
        return

    time.sleep(2)

    # 2. Проверяем manifest
    manifest_path = work_dir / "manifest_cells.csv"
    if not manifest_path.exists():
        print(f"✗ Manifest не найден: {manifest_path}")
        return

    print(f"\n✓ Manifest создан: {manifest_path}")

    # 3. LLP обучение и предсказание
    print("\n2. LLP ОБУЧЕНИЕ И ПРЕДСКАЗАНИЕ")
    print("-" * 40)
    try:
        llp_script = scripts_dir / "30_train_predict_ci.py"
        if llp_script.exists():
            cmd = [
                sys.executable, str(llp_script),
                "--work_dir", str(work_dir),
                "--epochs", "25",
                "--force_cpu", "0"
            ]

            print(f"Запуск LLP анализа...")
            print(f"Команда: {' '.join(cmd)}")

            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("Stderr:", result.stderr)

            print("✓ LLP анализ завершен")
        else:
            print(f"✗ LLP скрипт не найден: {llp_script}")
            return
    except subprocess.CalledProcessError as e:
        print(f"✗ Ошибка при LLP анализе: {e}")
        print(f"Stderr: {e.stderr}")

        # Пробуем с force_cpu=1
        print("\nПробуем запустить с CPU...")
        try:
            cmd = [
                sys.executable, str(llp_script),
                "--work_dir", str(work_dir),
                "--epochs", "25",
                "--force_cpu", "1"
            ]
            subprocess.run(cmd, check=True)
            print("✓ LLP анализ завершен на CPU")
        except subprocess.CalledProcessError as e2:
            print(f"✗ Ошибка на CPU: {e2}")
            return

    time.sleep(2)

    # 4. Анализ результатов
    print("\n3. АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("-" * 40)

    try:
        # Создаем простой скрипт для анализа
        analysis_script = work_dir / "analyze_results.py"

        analysis_code = '''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Добавляем путь к корневой директории проекта
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def main():
    # Текущая директория
    work_dir = Path(__file__).parent

    # Загружаем результаты LLP
    pred_path = work_dir / "predicted_phase_percentages.csv"
    cell_pred_path = work_dir / "cell_predictions.csv"

    if not pred_path.exists():
        print("Файл предсказаний не найден!")
        return

    pred_df = pd.read_csv(pred_path)
    print("LLP предсказания:")
    print(pred_df.to_string())

    # Данные цитометра
    flow_data = pd.DataFrame([
        # 2h
        {"genotype": "KELLY", "time": 2, "concentration": 0.0, "treatment": "CTRL", "SubG1": 3.56, "G1": 58.71, "G2M": 37.10},
        {"genotype": "KELLY", "time": 2, "concentration": 0.5, "treatment": "AURA", "SubG1": 2.94, "G1": 59.23, "G2M": 37.33},
        {"genotype": "KELLY", "time": 2, "concentration": 1.0, "treatment": "AURA", "SubG1": 5.53, "G1": 64.88, "G2M": 29.08},
        {"genotype": "KELLY", "time": 2, "concentration": 2.0, "treatment": "AURA", "SubG1": 7.53, "G1": 64.16, "G2M": 27.70},
        # 6h
        {"genotype": "KELLY", "time": 6, "concentration": 0.0, "treatment": "CTRL", "SubG1": 3.06, "G1": 57.67, "G2M": 38.70},
        {"genotype": "KELLY", "time": 6, "concentration": 0.5, "treatment": "AURA", "SubG1": 8.48, "G1": 58.12, "G2M": 32.36},
        {"genotype": "KELLY", "time": 6, "concentration": 1.0, "treatment": "AURA", "SubG1": 16.05, "G1": 51.39, "G2M": 31.10},
        {"genotype": "KELLY", "time": 6, "concentration": 2.0, "treatment": "AURA", "SubG1": 21.09, "G1": 53.52, "G2M": 24.04},
        # 24h
        {"genotype": "KELLY", "time": 24, "concentration": 0.0, "treatment": "CTRL", "SubG1": 7.59, "G1": 62.00, "G2M": 29.72},
        {"genotype": "KELLY", "time": 24, "concentration": 0.5, "treatment": "AURA", "SubG1": 21.98, "G1": 57.08, "G2M": 20.51},
        {"genotype": "KELLY", "time": 24, "concentration": 1.0, "treatment": "AURA", "SubG1": 40.71, "G1": 50.61, "G2M": 8.59},
        {"genotype": "KELLY", "time": 24, "concentration": 2.0, "treatment": "AURA", "SubG1": 62.65, "G1": 28.89, "G2M": 8.06},
    ])

    # Создаем директорию для графиков
    plots_dir = work_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Сравнение для каждого времени
    times = [2, 6, 24]
    phases = ["SubG1", "G1", "G2M"]

    fig, axes = plt.subplots(len(times), 1, figsize=(12, 4*len(times)))
    if len(times) == 1:
        axes = [axes]

    for i, time_h in enumerate(times):
        ax = axes[i]

        # Данные LLP для этого времени
        llp_time = pred_df[pred_df["time"] == time_h].copy()
        llp_time = llp_time.sort_values(["treatment", "concentration"])

        # Данные цитометра для этого времени
        flow_time = flow_data[flow_data["time"] == time_h].copy()
        flow_time = flow_time.sort_values(["treatment", "concentration"])

        # Индексы для группировки
        conditions = []
        llp_values = {phase: [] for phase in phases}
        flow_values = {phase: [] for phase in phases}

        for _, row in llp_time.iterrows():
            cond_name = "Ctrl" if row["treatment"] == "CTRL" else f"{row['concentration']}µM"
            conditions.append(cond_name)

            for phase in phases:
                llp_values[phase].append(row[phase])

        for _, row in flow_time.iterrows():
            for phase in phases:
                flow_values[phase].append(row[phase])

        # Строим график
        x = np.arange(len(conditions))
        width = 0.35

        # LLP предсказания
        bottom = np.zeros(len(conditions))
        for phase in phases:
            ax.bar(x - width/2, llp_values[phase], width, bottom=bottom, 
                   label=f"LLP {phase}", alpha=0.8)
            bottom += np.array(llp_values[phase])

        # Данные цитометра
        bottom = np.zeros(len(conditions))
        for phase in phases:
            ax.bar(x + width/2, flow_values[phase], width, bottom=bottom,
                   label=f"Flow {phase}", alpha=0.5, hatch="//")
            bottom += np.array(flow_values[phase])

        ax.set_xlabel("Условия")
        ax.set_ylabel("Процент (%)")
        ax.set_title(f"Сравнение LLP и проточной цитометрии - {time_h}ч")
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(plots_dir / "llp_vs_flow_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Scatter plot корреляции
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, phase in enumerate(phases):
        ax = axes[i]

        llp_vals = []
        flow_vals = []
        labels = []

        for time_h in times:
            llp_time = pred_df[pred_df["time"] == time_h]
            flow_time = flow_data[flow_data["time"] == time_h]

            for conc in [0, 0.5, 1, 2]:
                llp_val = llp_time[
                    (llp_time["concentration"] == conc) & 
                    (llp_time["treatment"] == ("CTRL" if conc == 0 else "AURA"))
                ][phase].values

                flow_val = flow_time[
                    (flow_time["concentration"] == conc) & 
                    (flow_time["treatment"] == ("CTRL" if conc == 0 else "AURA"))
                ][phase].values

                if len(llp_val) > 0 and len(flow_val) > 0:
                    llp_vals.append(llp_val[0])
                    flow_vals.append(flow_val[0])
                    labels.append(f"{conc}µM {time_h}h")

        ax.scatter(flow_vals, llp_vals, alpha=0.7, s=100)

        # Линия идеального соответствия
        min_val = min(min(flow_vals), min(llp_vals))
        max_val = max(max(flow_vals), max(llp_vals))
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)

        # Корреляция
        correlation = np.corrcoef(flow_vals, llp_vals)[0, 1]
        ax.text(0.05, 0.95, f"R = {correlation:.3f}", transform=ax.transAxes,
                fontsize=12, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.set_xlabel("Проточная цитометрия (%)")
        ax.set_ylabel("LLP предсказание (%)")
        ax.set_title(f"Корреляция - {phase}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "correlation_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Графики сохранены в: {plots_dir}")

    # Сохраняем сводную таблицу
    comparison_data = []

    for _, llp_row in pred_df.iterrows():
        flow_row = flow_data[
            (flow_data["time"] == llp_row["time"]) &
            (flow_data["concentration"] == llp_row["concentration"]) &
            (flow_data["treatment"] == llp_row["treatment"])
        ]

        if not flow_row.empty:
            flow_row = flow_row.iloc[0]
            for phase in phases:
                comparison_data.append({
                    "time": llp_row["time"],
                    "concentration": llp_row["concentration"],
                    "treatment": llp_row["treatment"],
                    "phase": phase,
                    "llp": llp_row[phase],
                    "flow": flow_row[phase],
                    "difference": llp_row[phase] - flow_row[phase],
                    "abs_difference": abs(llp_row[phase] - flow_row[phase])
                })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = work_dir / "comparison_with_flow.csv"
    comparison_df.to_csv(comparison_path, index=False)

    # Статистика
    print("\nСтатистика сравнения:")
    print("-" * 40)
    mae = comparison_df.groupby("phase")["abs_difference"].mean()
    rmse = np.sqrt(comparison_df.groupby("phase")["difference"].apply(lambda x: (x**2).mean()))

    stats_df = pd.DataFrame({
        "MAE": mae,
        "RMSE": rmse
    }).round(3)

    print(stats_df)

    stats_path = work_dir / "statistics.csv"
    stats_df.to_csv(stats_path)

    print(f"\nСтатистика сохранена: {stats_path}")
    print(f"Сравнение сохранено: {comparison_path}")

if __name__ == "__main__":
    main()
'''

        with open(analysis_script, 'w', encoding='utf-8') as f:
            f.write(analysis_code)

        print("Запуск анализа результатов...")
        result = subprocess.run([sys.executable, str(analysis_script)],
                                check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)

        print("✓ Анализ результатов завершен")

    except Exception as e:
        print(f"✗ Ошибка при анализе: {e}")
        import traceback
        traceback.print_exc()

    # Вывод результатов
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)

    print(f"\n📁 РЕЗУЛЬТАТЫ В ДИРЕКТОРИИ: {work_dir}")
    print("-" * 40)

    # Показываем созданные файлы
    import os

    key_files = [
        ("📋 Manifest клеток", "manifest_cells.csv"),
        ("🤖 LLP предсказания", "predicted_phase_percentages.csv"),
        ("🔬 Предсказания клеток", "cell_predictions.csv"),
        ("📊 Сравнение с цитометрией", "comparison_with_flow.csv"),
        ("📈 Статистика", "statistics.csv"),
        ("📉 Графики сравнения", "plots/llp_vs_flow_comparison.png"),
        ("📊 Графики корреляции", "plots/correlation_scatter.png")
    ]

    for desc, rel_path in key_files:
        file_path = work_dir / rel_path
        if file_path.exists():
            size = os.path.getsize(file_path) if file_path.is_file() else "папка"
            print(f"✓ {desc}: {file_path} ({size})")
        else:
            print(f"✗ {desc}: не найден")

    print("\n🎯 РЕКОМЕНДАЦИИ ПО ИСПОЛЬЗОВАНИЮ:")
    print("-" * 40)
    print("1. Основные графики: plots/llp_vs_flow_comparison.png")
    print("2. Для публикации используйте графики из папки plots/")
    print("3. Данные для статистики: comparison_with_flow.csv")
    print("4. Все сырые данные в CSV файлах")

    print("\n📊 КАК ИНТЕРПРЕТИРОВАТЬ РЕЗУЛЬТАТЫ:")
    print("-" * 40)
    print("• R > 0.8: отличная корреляция с цитометрией")
    print("• R = 0.6-0.8: хорошая корреляция")
    print("• R < 0.6: требуется улучшение метода")
    print("• MAE < 5%: высокая точность")
    print("• MAE 5-10%: приемлемая точность")
    print("• MAE > 10%: требуется калибровка")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_llp_analysis()