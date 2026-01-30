"""
Скрипт для создания полной структуры проекта и кода в одном TXT файле
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import shutil


class ProjectStructureExporter:
    """Экспортер структуры проекта и кода"""

    def __init__(self, project_root='.'):
        self.project_root = Path(project_root).resolve()
        self.output_file = self.project_root / 'project_structure_with_code.txt'
        self.ignore_dirs = {
            '__pycache__', '.git', '.idea', '.vscode', 'venv',
            'env', '.env', 'node_modules', 'dist', 'build',
            'data', 'models', 'results'  # Игнорируем данные и результаты
        }
        self.ignore_files = {
            '.gitignore', '.DS_Store', 'Thumbs.db', '*.pyc',
            '*.npy', '*.pth', '*.joblib', '*.jpg', '*.png',
            '*.jpeg', '*.gif', '*.bmp', '*.tiff'
        }

    def should_include_file(self, file_path):
        """Проверяет, нужно ли включать файл в экспорт"""
        # Проверяем расширения файлов, которые нужно игнорировать
        for ignore_pattern in self.ignore_files:
            if ignore_pattern.startswith('*'):
                if file_path.suffix == ignore_pattern[1:]:
                    return False
            elif file_path.name == ignore_pattern:
                return False

        # Игнорируем большие бинарные файлы
        if file_path.stat().st_size > 10_000_000:  # 10MB
            return False

        return True

    def should_include_dir(self, dir_path):
        """Проверяет, нужно ли включать директорию в экспорт"""
        return dir_path.name not in self.ignore_dirs

    def get_file_content(self, file_path):
        """Получает содержимое файла с кодировкой"""
        try:
            # Пробуем разные кодировки
            for encoding in ['utf-8', 'cp1251', 'latin-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue

            # Если не удалось прочитать как текст, возвращаем информацию о бинарном файле
            return f"[BINARY FILE - Size: {file_path.stat().st_size} bytes]"
        except Exception as e:
            return f"[ERROR READING FILE: {e}]"

    def get_project_tree(self):
        """Получает дерево проекта в текстовом формате"""
        tree_lines = []

        def build_tree(dir_path, prefix=""):
            """Рекурсивно строит дерево директорий"""
            try:
                # Получаем элементы, отсортированные по имени
                items = sorted(os.listdir(dir_path))

                for i, item in enumerate(items):
                    item_path = dir_path / item
                    is_last = i == len(items) - 1

                    # Пропускаем игнорируемые директории
                    if item_path.is_dir() and not self.should_include_dir(item_path):
                        continue

                    # Определяем префикс для текущего уровня
                    if is_last:
                        tree_prefix = prefix + "└── "
                        next_prefix = prefix + "    "
                    else:
                        tree_prefix = prefix + "├── "
                        next_prefix = prefix + "│   "

                    # Добавляем элемент в дерево
                    if item_path.is_dir():
                        tree_lines.append(f"{tree_prefix}{item}/")
                        # Рекурсивно обходим поддиректорию
                        build_tree(item_path, next_prefix)
                    else:
                        # Пропускаем игнорируемые файлы
                        if not self.should_include_file(item_path):
                            continue
                        # Показываем размер файла
                        size = item_path.stat().st_size
                        size_str = self.format_size(size)
                        tree_lines.append(f"{tree_prefix}{item} ({size_str})")

            except PermissionError:
                tree_lines.append(f"{prefix}[Permission denied]")
            except Exception as e:
                tree_lines.append(f"{prefix}[Error: {e}]")

        # Начинаем с корня проекта
        tree_lines.append(f"{self.project_root.name}/")
        build_tree(self.project_root, "")

        return "\n".join(tree_lines)

    def format_size(self, size_bytes):
        """Форматирует размер файла в читаемом виде"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def export_project(self):
        """Экспортирует всю структуру проекта и код в TXT файл"""

        print("🚀 Начинаю экспорт проекта...")

        with open(self.output_file, 'w', encoding='utf-8') as f:
            # Заголовок
            f.write("=" * 80 + "\n")
            f.write(f"ПОЛНАЯ СТРУКТУРА ПРОЕКТА И ИСХОДНЫЙ КОД\n")
            f.write(f"Проект: {self.project_root.name}\n")
            f.write(f"Дата экспорта: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # Секция 1: Дерево проекта
            print("📁 Генерирую дерево проекта...")
            f.write("СЕКЦИЯ 1: СТРУКТУРА ПРОЕКТА\n")
            f.write("-" * 80 + "\n\n")
            f.write(self.get_project_tree())
            f.write("\n\n")

            # Секция 2: Основные файлы проекта
            print("📄 Экспортирую основные файлы проекта...")
            f.write("СЕКЦИЯ 2: ОСНОВНЫЕ ФАЙЛЫ ПРОЕКТА\n")
            f.write("-" * 80 + "\n\n")

            # Список основных файлов для детального экспорта
            main_files = [
                'requirements.txt',
                'main.py',
                'run_all_stages.py',
                'create_project_structure.py',
            ]

            for main_file in main_files:
                file_path = self.project_root / main_file
                if file_path.exists():
                    f.write(f"ФАЙЛ: {main_file}\n")
                    f.write("=" * 60 + "\n")
                    f.write(self.get_file_content(file_path))
                    f.write("\n" + "=" * 60 + "\n\n")

            # Секция 3: Исходный код из src/
            print("💻 Экспортирую исходный код из src/...")
            f.write("СЕКЦИЯ 3: ИСХОДНЫЙ КОД (src/)\n")
            f.write("-" * 80 + "\n\n")

            src_dir = self.project_root / 'src'
            if src_dir.exists():
                # Рекурсивно обходим все файлы в src/
                for root, dirs, files in os.walk(src_dir):
                    # Пропускаем игнорируемые директории
                    dirs[:] = [d for d in dirs if d not in self.ignore_dirs]

                    for file in files:
                        file_path = Path(root) / file

                        # Пропускаем не-Python файлы и игнорируемые
                        if not file.endswith('.py') or not self.should_include_file(file_path):
                            continue

                        # Относительный путь от src/
                        rel_path = file_path.relative_to(src_dir)

                        f.write(f"ФАЙЛ: src/{rel_path}\n")
                        f.write("=" * 60 + "\n")
                        f.write(self.get_file_content(file_path))
                        f.write("\n" + "=" * 60 + "\n\n")

            # НОВАЯ СЕКЦИЯ: Скрипты из папки scripts
            print("📜 Экспортирую скрипты из scripts/...")
            f.write("СЕКЦИЯ 4: СКРИПТЫ (scripts/)\n")
            f.write("-" * 80 + "\n\n")

            scripts_dir = self.project_root / 'scripts'
            if scripts_dir.exists():
                # Получаем все файлы из папки scripts
                for item in sorted(os.listdir(scripts_dir)):
                    file_path = scripts_dir / item

                    # Пропускаем директории
                    if file_path.is_dir():
                        continue

                    # Пропускаем игнорируемые файлы
                    if not self.should_include_file(file_path):
                        continue

                    f.write(f"ФАЙЛ: scripts/{item}\n")
                    f.write("=" * 60 + "\n")
                    f.write(self.get_file_content(file_path))
                    f.write("\n" + "=" * 60 + "\n\n")
            else:
                f.write("Папка scripts не найдена.\n\n")

            # Секция 5: Инструкции и README файлы
            print("📖 Экспортирую документацию...")
            f.write("СЕКЦИЯ 5: ДОКУМЕНТАЦИЯ И ИНСТРУКЦИИ\n")
            f.write("-" * 80 + "\n\n")

            # Ищем README файлы
            readme_files = []
            for pattern in ['README*', 'readme*', 'Readme*', '*.md']:
                readme_files.extend(self.project_root.rglob(pattern))

            for readme_file in readme_files:
                if readme_file.is_file() and self.should_include_file(readme_file):
                    rel_path = readme_file.relative_to(self.project_root)
                    f.write(f"ДОКУМЕНТАЦИЯ: {rel_path}\n")
                    f.write("=" * 60 + "\n")
                    f.write(self.get_file_content(readme_file))
                    f.write("\n" + "=" * 60 + "\n\n")

            # Секция 6: Краткое описание проекта
            print("📋 Создаю сводку проекта...")
            f.write("СЕКЦИЯ 6: СВОДКА ПРОЕКТА\n")
            f.write("-" * 80 + "\n\n")

            # Собираем статистику
            python_files = list(self.project_root.rglob('*.py'))
            total_lines = 0
            total_size = 0

            for py_file in python_files:
                if self.should_include_file(py_file):
                    try:
                        content = self.get_file_content(py_file)
                        if content and not content.startswith('[BINARY FILE') and not content.startswith('[ERROR'):
                            total_lines += len(content.split('\n'))
                            total_size += py_file.stat().st_size
                    except:
                        pass

            # Записываем сводку
            f.write("ОБЩАЯ ИНФОРМАЦИЯ:\n")
            f.write(f"  • Название проекта: {self.project_root.name}\n")
            f.write(f"  • Дата экспорта: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  • Python файлов: {len(python_files)}\n")
            f.write(f"  • Всего строк кода: {total_lines:,}\n")
            f.write(f"  • Общий размер кода: {self.format_size(total_size)}\n")
            f.write(f"  • Игнорируемые папки: {', '.join(sorted(self.ignore_dirs))}\n")
            f.write("\n")

            f.write("СТРУКТУРА ПРОЕКТА (кратко):\n")
            for item in sorted(os.listdir(self.project_root)):
                item_path = self.project_root / item
                if item_path.is_dir() and self.should_include_dir(item_path):
                    num_files = len(list(item_path.rglob('*.py')))
                    f.write(f"  • {item}/ - {num_files} Python файл(ов)\n")

            f.write("\n")

            f.write("ИНСТРУКЦИЯ ПО ЗАПУСКУ:\n")
            f.write("  1. Установите зависимости: pip install -r requirements.txt\n")
            f.write("  2. Поместите изображения в data/raw_images/\n")
            f.write("  3. Запустите основной скрипт: python main.py --mode all\n")
            f.write("  4. Или запустите все этапы: python run_all_stages.py\n")
            f.write("\n")

            # Финальная информация
            f.write("=" * 80 + "\n")
            f.write(f"ЭКСПОРТ ЗАВЕРШЕН УСПЕШНО!\n")
            f.write(f"Файл создан: {self.output_file.name}\n")
            # Получаем размер файла
            try:
                file_size = self.output_file.stat().st_size
                f.write(f"Размер файла: {self.format_size(file_size)}\n")
            except:
                f.write(f"Размер файла: неизвестен\n")
            f.write("=" * 80 + "\n")

        print(f"\n✅ Экспорт завершен!")
        print(f"📄 Файл создан: {self.output_file}")
        try:
            file_size = self.output_file.stat().st_size
            print(f"📏 Размер файла: {self.format_size(file_size)}")
        except:
            print(f"📏 Размер файла: неизвестен")

        # Показываем предварительный просмотр
        self.show_preview()

    def show_preview(self):
        """Показывает предварительный просмотр созданного файла"""
        print("\n" + "=" * 60)
        print("ПРЕДВАРИТЕЛЬНЫЙ ПРОСМОТР:")
        print("=" * 60)

        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Показываем первые 50 строк
            print("Первые 50 строк файла:")
            for i, line in enumerate(lines[:50]):
                print(f"{i + 1:3d}: {line.rstrip()}")

            # Показываем последние 10 строк
            print("\nПоследние 10 строк файла:")
            for i, line in enumerate(lines[-10:]):
                print(f"... {line.rstrip()}")

        except Exception as e:
            print(f"Ошибка при чтении файла: {e}")

    def create_compressed_copy(self):
        """Создает сжатую копию файла (без дублирования)"""
        print("\n📦 Создаю сжатую версию файла...")

        compressed_file = self.project_root / 'project_structure_compressed.txt'

        with open(self.output_file, 'r', encoding='utf-8') as source:
            lines = source.readlines()

        # Удаляем пустые строки и сжимаем пробелы
        compressed_lines = []
        skip_empty = 0

        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                skip_empty += 1
                if skip_empty <= 2:  # Оставляем максимум 2 пустых строки подряд
                    compressed_lines.append('')
            else:
                skip_empty = 0
                # Сжимаем множественные пробелы
                compressed = ' '.join(stripped.split())
                compressed_lines.append(compressed)

        with open(compressed_file, 'w', encoding='utf-8') as target:
            target.write('\n'.join(compressed_lines))

        original_size = self.output_file.stat().st_size
        compressed_size = compressed_file.stat().st_size
        compression_ratio = (1 - compressed_size / original_size) * 100

        print(f"✅ Сжатая версия создана: {compressed_file.name}")
        print(f"📏 Размер оригинала: {self.format_size(original_size)}")
        print(f"📏 Размер сжатого: {self.format_size(compressed_size)}")
        print(f"📊 Степень сжатия: {compression_ratio:.1f}%")


def main():
    """Основная функция"""

    print("🚀 СКРИПТ ЭКСПОРТА СТРУКТУРЫ ПРОЕКТА")
    print("=" * 60)

    # Определяем корень проекта
    current_dir = Path.cwd()
    print(f"Текущая директория: {current_dir}")

    # Проверяем, есть ли необходимые файлы проекта
    required_files = ['main.py', 'requirements.txt', 'src/']
    missing_files = []

    for req in required_files:
        if not (current_dir / req).exists():
            missing_files.append(req)

    if missing_files:
        print(f"\n⚠️  Внимание: отсутствуют некоторые файлы проекта:")
        for missing in missing_files:
            print(f"  - {missing}")
        print(f"\nУбедитесь, что вы находитесь в корневой папке проекта.")
        response = input("Продолжить экспорт? (y/n): ")
        if response.lower() != 'y':
            print("Экспорт отменен.")
            return

    # Создаем экземпляр экспортера
    exporter = ProjectStructureExporter(current_dir)

    # Настраиваем игнорируемые директории
    print("\n📁 Игнорируемые директории:")
    for dir_name in sorted(exporter.ignore_dirs):
        dir_path = current_dir / dir_name
        if dir_path.exists():
            print(f"  - {dir_name}/ (существует)")
        else:
            print(f"  - {dir_name}/ (не существует)")

    # Предлагаем добавить дополнительные директории для игнорирования
    print("\n➕ Добавить дополнительные директории для игнорирования?")
    print("  (нажмите Enter, чтобы пропустить, или введите имена через запятую)")
    additional_dirs = input("  Дополнительные директории: ").strip()

    if additional_dirs:
        for dir_name in additional_dirs.split(','):
            dir_name = dir_name.strip()
            if dir_name:
                exporter.ignore_dirs.add(dir_name)
                print(f"  Добавлено: {dir_name}")

    # Запускаем экспорт
    print("\n" + "=" * 60)
    print("НАЧИНАЮ ЭКСПОРТ...")
    print("=" * 60)

    try:
        exporter.export_project()

        # Предлагаем создать сжатую версию
        print("\n📦 Создать сжатую версию файла (удалить лишние пустые строки)?")
        if input("  (y/n): ").lower() == 'y':
            exporter.create_compressed_copy()

        # Финальное сообщение
        print("\n" + "=" * 60)
        print("🎉 ЭКСПОРТ УСПЕШНО ЗАВЕРШЕН!")
        print("=" * 60)
        print(f"\n📄 Основной файл: {exporter.output_file}")

        if (current_dir / 'project_structure_compressed.txt').exists():
            print(f"📄 Сжатая версия: project_structure_compressed.txt")

        print(f"\n📊 Размеры файлов:")
        for file_name in ['project_structure_with_code.txt', 'project_structure_compressed.txt']:
            file_path = current_dir / file_name
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"  • {file_name}: {exporter.format_size(size)}")

        print("\n📤 Файл готов для передачи коллегам или архивации!")

    except KeyboardInterrupt:
        print("\n\n⏹️  Экспорт прерван пользователем.")
    except Exception as e:
        print(f"\n❌ Ошибка при экспорте: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()