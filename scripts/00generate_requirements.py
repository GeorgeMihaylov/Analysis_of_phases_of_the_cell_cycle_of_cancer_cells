"""
Простой скрипт для получения версий пакетов
"""

import subprocess
import sys

def get_versions_simple():
    """Простой способ получения версий пакетов"""

    packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn',
        'scipy', 'tqdm', 'Pillow', 'opencv-python',
        'scikit-image', 'cellpose', 'torch', 'torchvision',
        'scikit-learn', 'joblib'
    ]

    print("Получение версий пакетов...")
    print("=" * 50)

    with open('requirements_simple.txt', 'w') as f:
        f.write("# Автоматически сгенерированный requirements.txt\n\n")

        for package in packages:
            try:
                # Запускаем pip show для получения информации о пакете
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'show', package],
                    capture_output=True,
                    text=True,
                    encoding='utf-8'
                )

                if result.returncode == 0:
                    # Парсим вывод
                    lines = result.stdout.split('\n')
                    version = None
                    for line in lines:
                        if line.startswith('Version:'):
                            version = line.split(':')[1].strip()
                            break

                    if version:
                        print(f"✅ {package}: {version}")
                        f.write(f"{package}=={version}\n")
                    else:
                        print(f"⚠️  {package}: версия не найдена")
                        f.write(f"# {package} (версия не определена)\n")
                else:
                    print(f"❌ {package}: не установлен")
                    f.write(f"# {package} (не установлен)\n")

            except Exception as e:
                print(f"❌ Ошибка при проверке {package}: {e}")
                f.write(f"# {package} (ошибка проверки)\n")

    print("\n" + "=" * 50)
    print("Файл requirements_simple.txt создан!")

if __name__ == "__main__":
    get_versions_simple()