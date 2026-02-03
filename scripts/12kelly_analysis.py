import sys
import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from cellpose import models, io
from tqdm import tqdm
import cv2

# --- НАСТРОЙКИ ---
# Определяем корень проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Абсолютный путь к данным
DATA_DIR = PROJECT_ROOT / 'data' / 'kelly_auranofin'

# Куда сохранять результаты
RESULTS_DIR = PROJECT_ROOT / 'results' / 'kelly_morphology'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Корень проекта: {PROJECT_ROOT}")
print(f"Данные ищутся в: {DATA_DIR}")

# Умная регулярка для имен файлов
FILENAME_REGEX = re.compile(
    r"Kelly\s+"
    r"(?P<drug>Aura|ctrl)\s*"  # Препарат или контроль
    r"(?:(?P<conc>[\d\.]+)uM)?\s*"  # Концентрация
    r"(?P<time>\d+)h",  # Время
    re.IGNORECASE
)


def analyze_kelly_dataset():
    print(f"--- Анализ морфологии Kelly в {DATA_DIR} ---")

    # Инициализация модели с CPU вместо GPU
    try:
        # Попробуем использовать GPU, но с дополнительными параметрами
        model = models.CellposeModel(gpu=True, model_type='cyto')
        print("Используется GPU для вычислений")
    except Exception as e:
        print(f"Ошибка инициализации GPU: {e}")
        print("Переключаюсь на CPU...")
        model = models.CellposeModel(gpu=False, model_type='cyto')
        print("Используется CPU для вычислений")

    files = sorted(list(DATA_DIR.glob('*.jpg')) + list(DATA_DIR.glob('*.png')))
    if not files:
        print(f"Файлы не найдены в {DATA_DIR.absolute()}! Проверьте путь.")
        return

    print(f"Найдено {len(files)} файлов.")
    all_data = []

    for f in tqdm(files):
        # Парсинг имени файла
        match = FILENAME_REGEX.search(f.name)
        if match:
            drug = match.group('drug').lower()
            time = match.group('time')
            conc = match.group('conc')

            # Нормализация названия группы
            if 'ctrl' in drug:
                group_name = "Control"
            else:
                group_name = f"{conc} uM" if conc else "Aura"
        else:
            print(f"⚠️ Не распознано имя: {f.name}")
            # Пытаемся извлечь информацию другим способом
            group_name = "Unknown"
            time = "Unknown"
            if 'ctrl' in f.name.lower():
                group_name = "Control"
            # Ищем время
            time_match = re.search(r'(\d+)h', f.name)
            if time_match:
                time = time_match.group(1)

        # Чтение изображения
        try:
            img = io.imread(str(f))
            # Если изображение цветное, конвертируем в оттенки серого
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        except Exception as e:
            print(f"Ошибка чтения файла {f.name}: {e}")
            continue

        # Сегментация с параметрами для стабильности
        try:
            # Используем фиксированный диаметр или автоматическое определение
            masks, _, _ = model.eval(
                img,
                diameter=None,  # Автоматическое определение
                channels=[0, 0],  # Для grayscale изображений
                flow_threshold=0.4,  # Более строгий порог
                cellprob_threshold=0.0,
                normalize=True  # Нормализация изображения
            )
        except Exception as e:
            print(f"Ошибка сегментации файла {f.name}: {e}")
            continue

        # Анализ формы каждой клетки
        unique_cells = np.unique(masks)
        if len(unique_cells) <= 1:  # Только фон (0)
            print(f"⚠️ Не найдено клеток на фото: {f.name}")
            continue

        for cell_id in unique_cells[1:]:  # Пропускаем фон
            mask = (masks == cell_id).astype(np.uint8)
            area = np.sum(mask)

            # Фильтрация шума и артефактов
            if area < 50 or area > 50000:  # Реалистичные границы для клеток
                continue

            # Вычисление округлости
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            contour = contours[0]
            perimeter = cv2.arcLength(contour, True)

            if perimeter <= 0 or area <= 0:
                continue

            # Вычисление округлости
            circularity = (4 * np.pi * area) / (perimeter ** 2)

            # Коррекция возможных выбросов
            circularity = min(max(circularity, 0), 1.0)

            # Дополнительные метрики
            if len(contour) >= 5:  # Нужно минимум 5 точек для эллипса
                ellipse = cv2.fitEllipse(contour)
                (center, axes, orientation) = ellipse
                major_axis = max(axes)
                minor_axis = min(axes)
                aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1.0
            else:
                aspect_ratio = 1.0

            all_data.append({
                'Filename': f.name,
                'Time': f"{time}h",
                'Group': group_name,
                'Circularity': circularity,
                'Area': area,
                'Aspect_Ratio': aspect_ratio,
                'Perimeter': perimeter
            })

    if not all_data:
        print("Не удалось извлечь данные.")
        return

    # Создание DataFrame и сохранение
    df = pd.DataFrame(all_data)
    csv_path = RESULTS_DIR / 'kelly_morphology_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"Обработано {len(df)} клеток.")
    print(f"Данные сохранены в: {csv_path}")

    # Статистический анализ
    print("\n--- Сводная статистика ---")
    summary = df.groupby(['Group', 'Time']).agg({
        'Circularity': ['mean', 'std', 'count'],
        'Area': ['mean', 'std']
    }).round(3)
    print(summary)

    # --- ВИЗУАЛИЗАЦИЯ ---
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10

    # Порядок для легенды
    hue_order = sorted(df['Group'].unique())
    if "Control" in hue_order:
        hue_order.remove("Control")
        hue_order.insert(0, "Control")

    # Порядок времени
    time_order = sorted(df['Time'].unique(),
                        key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)

    # 1. График: Округлость
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Boxplot
    sns.boxplot(data=df, x='Time', y='Circularity', hue='Group',
                hue_order=hue_order, order=time_order,
                palette="viridis", ax=axes[0])
    axes[0].set_title("Изменение формы клеток Kelly\n(Circularity)", fontsize=12)
    axes[0].set_ylabel("Circularity (1.0 = идеальный круг)")
    axes[0].set_xlabel("Время (часы)")
    axes[0].legend(title='Группа')

    # Violin plot
    sns.violinplot(data=df, x='Time', y='Circularity', hue='Group',
                   hue_order=hue_order, order=time_order,
                   palette="viridis", ax=axes[1], split=True)
    axes[1].set_title("Распределение округлости", fontsize=12)
    axes[1].set_ylabel("Circularity")
    axes[1].set_xlabel("Время (часы)")
    axes[1].legend(title='Группа')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'morphology_circularity.png', dpi=300, bbox_inches='tight')

    # 2. График: Размер клеток
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Boxplot для площади
    sns.boxplot(data=df, x='Time', y='Area', hue='Group',
                hue_order=hue_order, order=time_order,
                palette="magma", ax=axes[0])
    axes[0].set_title("Изменение размера клеток (Area)", fontsize=12)
    axes[0].set_ylabel("Площадь (пиксели)")
    axes[0].set_xlabel("Время (часы)")
    axes[0].legend(title='Группа')

    # Scatter plot: Area vs Circularity
    scatter = axes[1].scatter(df['Area'], df['Circularity'],
                              c=pd.Categorical(df['Group']).codes,
                              alpha=0.6, cmap='viridis', s=20)
    axes[1].set_title("Зависимость формы от размера", fontsize=12)
    axes[1].set_xlabel("Площадь (пиксели)")
    axes[1].set_ylabel("Circularity")

    # Добавление легенды для групп
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=plt.cm.viridis(i / len(hue_order)),
                          markersize=10) for i in range(len(hue_order))]
    axes[1].legend(handles, hue_order, title='Группа')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'morphology_area.png', dpi=300, bbox_inches='tight')

    # 3. График: Статистика по группам
    plt.figure(figsize=(10, 6))

    # Группировка данных
    group_stats = df.groupby(['Group', 'Time'])['Circularity'].mean().unstack()

    # Линейный график
    for group in group_stats.index:
        plt.plot(group_stats.columns, group_stats.loc[group],
                 marker='o', label=group, linewidth=2)

    plt.title("Динамика изменения округлости во времени", fontsize=12)
    plt.xlabel("Время (часы)")
    plt.ylabel("Средняя Circularity")
    plt.legend(title='Группа')
    plt.grid(True, alpha=0.3)
    plt.savefig(RESULTS_DIR / 'circularity_dynamics.png', dpi=300, bbox_inches='tight')

    # 4. Сохранение сводной таблицы
    summary_path = RESULTS_DIR / 'morphology_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Анализ морфологии клеток Kelly\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Всего файлов: {len(files)}\n")
        f.write(f"Всего клеток: {len(df)}\n\n")
        f.write("Статистика по группам:\n")
        f.write(summary.to_string())

    print(f"\nГотово! Результаты сохранены в {RESULTS_DIR}")
    print(f"Графики: morphology_circularity.png, morphology_area.png, circularity_dynamics.png")
    print(f"Данные: kelly_morphology_data.csv")
    print(f"Сводка: morphology_summary.txt")


if __name__ == "__main__":
    # Добавляем переменные окружения для стабильности CUDA
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    try:
        analyze_kelly_dataset()
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()