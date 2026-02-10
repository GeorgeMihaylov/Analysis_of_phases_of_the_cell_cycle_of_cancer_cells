# scripts/41_quick_llp.py
# !/usr/bin/env python3
"""
Быстрый запуск LLP анализа с существующими crops
"""

import subprocess
import sys
from pathlib import Path


def main():
    print("⚡ БЫСТРЫЙ ЗАПУСК LLP АНАЛИЗА")
    print("=" * 50)

    root = Path(__file__).parent.parent
    scripts_dir = root / "scripts"

    # Проверяем, есть ли crops
    crops_dir = root / "results" / "cell_crops"
    if not crops_dir.exists():
        print("✗ Crops не найдены. Сначала запустите извлечение crops.")
        return

    # Создаем рабочую директорию для LLP
    work_dir = root / "results" / "llp_quick"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Копируем manifest и crops в рабочую директорию
    import shutil

    print("Копирование данных...")

    # Копируем manifest
    manifest_src = crops_dir / "manifest_cells.csv"
    manifest_dst = work_dir / "manifest_cells.csv"

    if manifest_src.exists():
        shutil.copy2(manifest_src, manifest_dst)
        print(f"✓ Manifest скопирован: {manifest_dst}")
    else:
        print("✗ Manifest не найден")
        return

    # Копируем crops
    crops_src = crops_dir / "crops"
    crops_dst = work_dir / "crops"

    if crops_src.exists():
        if crops_dst.exists():
            shutil.rmtree(crops_dst)
        shutil.copytree(crops_src, crops_dst)
        print(f"✓ Crops скопированы: {crops_dst}")
    else:
        print("✗ Crops не найдены")
        return

    # Запускаем LLP анализ
    print("\nЗапуск LLP анализа...")

    llp_script = scripts_dir / "30_train_predict_ci.py"
    if not llp_script.exists():
        print(f"✗ LLP скрипт не найден: {llp_script}")
        return

    cmd = [
        sys.executable, str(llp_script),
        "--work_dir", str(work_dir),
        "--epochs", "20",
        "--force_cpu", "1"  # Используем CPU для надежности
    ]

    print(f"Команда: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        if result.stderr:
            print("Предупреждения:", result.stderr)

        print("✓ LLP анализ завершен")

    except subprocess.CalledProcessError as e:
        print(f"✗ Ошибка: {e}")
        print(f"Stderr: {e.stderr}")
        return

    # Простой анализ результатов
    print("\n📊 АНАЛИЗ РЕЗУЛЬТАТОВ:")
    print("-" * 40)

    results_file = work_dir / "predicted_phase_percentages.csv"
    if results_file.exists():
        import pandas as pd
        df = pd.read_csv(results_file)
        print(df.to_string())

        # Простая визуализация
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # Группируем по времени
            times = sorted(df["time"].unique())

            fig, axes = plt.subplots(len(times), 1, figsize=(10, 4 * len(times)))
            if len(times) == 1:
                axes = [axes]

            for i, time_h in enumerate(times):
                ax = axes[i]
                time_data = df[df["time"] == time_h]

                conditions = []
                subg1_vals = []
                g1_vals = []
                g2m_vals = []

                for _, row in time_data.iterrows():
                    if row["treatment"] == "CTRL":
                        conditions.append("Ctrl")
                    else:
                        conditions.append(f"{row['concentration']}µM")

                    subg1_vals.append(row["SubG1"])
                    g1_vals.append(row["G1"])
                    g2m_vals.append(row["G2M"])

                x = np.arange(len(conditions))
                width = 0.25

                ax.bar(x - width, subg1_vals, width, label="SubG1", color="red", alpha=0.7)
                ax.bar(x, g1_vals, width, label="G1", color="green", alpha=0.7)
                ax.bar(x + width, g2m_vals, width, label="G2M", color="blue", alpha=0.7)

                ax.set_xlabel("Условия")
                ax.set_ylabel("Процент (%)")
                ax.set_title(f"Распределение фаз - {time_h}ч")
                ax.set_xticks(x)
                ax.set_xticklabels(conditions)
                ax.legend()
                ax.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            plot_path = work_dir / "llp_results.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"\n📈 График сохранен: {plot_path}")

        except Exception as e:
            print(f"Не удалось создать график: {e}")

    print(f"\n📁 Все результаты в: {work_dir}")


if __name__ == "__main__":
    main()