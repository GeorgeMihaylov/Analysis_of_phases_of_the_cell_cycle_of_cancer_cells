import os
import re
from pathlib import Path

# Определяем корневую директорию - в PyCharm используем локальные пути
project_root = Path(__file__).parent.parent  # предполагаем, что скрипт в src/ или аналогичной папке
raw_data_dir = project_root / 'data' / 'raw'

# Если папка не существует - создаем её
raw_data_dir.mkdir(parents=True, exist_ok=True)

# номер слайда -> (генотип, время)
slide_info = {
    '1': ('WT', '24'),
    '2': ('CDK8KO', '24'),
    '3': ('CDK8KO', '48'),
}

# Исправляем регулярное выражение - заменяем "Слайд" на "Слайд" (обычная буква й)
# пример имени: 'Слайд 1 - 04 (6 Gy).jpg'
pattern = re.compile(r'^Слайд\s+(\d)\s*-\s*(\d+)\s*\((\d+)\s*Gy\)\.jpg$', re.IGNORECASE)

# Также можно добавить альтернативный вариант для совместимости с обоими написаниями
# pattern = re.compile(r'^Сла[йй]д\s+(\d)\s*-\s*(\d+)\s*\((\d+)\s*Gy\)\.jpg$', re.IGNORECASE)

# Проверяем, есть ли файлы в директории
if not any(raw_data_dir.iterdir()):
    print(f"Директория {raw_data_dir} пуста. Пожалуйста, поместите изображения в эту папку.")
    print("Ожидаемые файлы в формате: 'Слайд 1 - 04 (6 Gy).jpg'")
    print("Список файлов в директории:", list(raw_data_dir.iterdir()))
else:
    print(f"Найдено файлов в директории: {len(list(raw_data_dir.glob('*.jpg')))}")

    # Обрабатываем только файлы с расширением .jpg
    for fname in os.listdir(raw_data_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg')):
            print(f"Пропускаем {fname} - не jpg файл")
            continue

        m = pattern.match(fname)
        if not m:
            # Попробуем найти любой файл с паттерном слайда, но другим форматом
            alt_pattern = re.compile(r'.*?(\d).*?(\d+).*?(\d+).*', re.IGNORECASE)
            alt_match = alt_pattern.match(fname)
            if alt_match:
                print(f"Файл {fname} не соответствует точному формату, но содержит числа: {alt_match.groups()}")
            else:
                print(f"Файл {fname} не соответствует ожидаемому формату имени")
            continue

        slide, field, dose = m.groups()

        # Проверяем, есть ли информация о слайде
        if slide not in slide_info:
            print(f"Нет информации для слайда {slide}")
            continue

        genotype, time_h = slide_info[slide]

        new_name = f'HCT116_{genotype}_{time_h}h_{dose}Gy_slide{slide}_field{int(field):02d}.jpg'

        src = raw_data_dir / fname
        dst = raw_data_dir / new_name

        try:
            # Проверяем, существует ли уже файл с таким именем
            if dst.exists():
                print(f"Файл {new_name} уже существует, пропускаем переименование {fname}")
                continue

            os.rename(src, dst)
            print(f'{fname} -> {new_name}')
        except Exception as e:
            print(f'Ошибка при переименовании {fname}: {e}')

    print("Обработка завершена.")
    print(f"Переименованные файлы в {raw_data_dir}:")
    for file in sorted(raw_data_dir.glob('HCT116_*.jpg')):
        print(f"  {file.name}")