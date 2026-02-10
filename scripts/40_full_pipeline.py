# scripts/40_full_pipeline.py
# !/usr/bin/env python3
"""
Полный пайплайн анализа клеточного цикла
"""

import subprocess
import sys
from pathlib import Path
import time


def main():
    print("🚀 ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА АНАЛИЗА КЛЕТОЧНОГО ЦИКЛА")
    print("=" * 60)

    root = Path(__file__).parent.parent
    scripts_dir = root / "scripts"

    steps = [
        ("1. СЕГМЕНТАЦИЯ КЛЕТОК", "34kelly_segment_improved_fixed.py"),
        ("2. ИЗВЛЕЧЕНИЕ CROPS", "38_build_crops_from_masks_fixed.py"),
        ("3. LLP АНАЛИЗ", "30_train_predict_ci.py"),
        ("4. АНАЛИЗ РЕЗУЛЬТАТОВ", "39_run_llp_analysis_fixed.py")
    ]

    for step_name, script_name in steps:
        print(f"\n{step_name}")
        print("-" * 40)

        script_path = scripts_dir / script_name

        if not script_path.exists():
            print(f"✗ Скрипт не найден: {script_path}")

            # Ищем альтернативные варианты
            alternatives = list(scripts_dir.glob(f"*{script_name.split('_')[-1]}"))
            if alternatives:
                print(f"Найдены альтернативы: {[alt.name for alt in alternatives]}")
                script_path = alternatives[0]
                print(f"Используем: {script_path}")
            else:
                print("Пропускаем шаг...")
                continue

        try:
            if script_name == "38_build_crops_from_masks_fixed.py":
                # Этот скрипт требует work_dir
                work_dir = root / "results" / "llp_final"
                cmd = [sys.executable, str(script_path), "--work_dir", str(work_dir)]
            elif script_name == "30_train_predict_ci.py":
                work_dir = root / "results" / "llp_final"
                cmd = [sys.executable, str(script_path), "--work_dir", str(work_dir), "--epochs", "25"]
            elif script_name == "39_run_llp_analysis_fixed.py":
                cmd = [sys.executable, str(script_path)]
            else:
                cmd = [sys.executable, str(script_path)]

            print(f"Запуск: {' '.join(cmd)}")

            # Запускаем с таймаутом 10 минут
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            print(result.stdout)
            if result.stderr:
                print("Предупреждения:", result.stderr)

            print(f"✓ {step_name} завершен")

        except subprocess.TimeoutExpired:
            print(f"✗ {step_name} превысил таймаут")
        except subprocess.CalledProcessError as e:
            print(f"✗ Ошибка в {step_name}: {e}")
            print(f"Stderr: {e.stderr}")
        except Exception as e:
            print(f"✗ Неожиданная ошибка в {step_name}: {e}")

        time.sleep(2)

    print("\n" + "=" * 60)
    print("✅ ПАЙПЛАЙН ЗАВЕРШЕН!")
    print("=" * 60)

    # Показываем результаты
    results_dir = root / "results"

    print("\n📁 Созданные директории:")
    print("-" * 40)

    for subdir in ["segmentation_fixed", "llp_final", "llp_analysis_v2"]:
        dir_path = results_dir / subdir
        if dir_path.exists():
            files = list(dir_path.glob("*"))
            print(f"\n📂 {subdir}/")
            for f in sorted(files)[:5]:
                if f.is_file():
                    print(f"   {f.name}")
            if len(files) > 5:
                print(f"   ... и еще {len(files) - 5} файлов")

    print("\n🎯 Дальнейшие действия:")
    print("-" * 40)
    print("1. Откройте results/llp_final/plots/ для просмотра графиков")
    print("2. Проверьте results/llp_final/comparison_with_flow.csv для данных")
    print("3. Используйте Excel для дополнительного анализа")
    print("4. Для публикации используйте графики из папки plots/")


if __name__ == "__main__":
    main()