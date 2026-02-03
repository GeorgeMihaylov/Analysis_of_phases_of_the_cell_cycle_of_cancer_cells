import pandas as pd
import numpy as np
from pathlib import Path
from cellpose import models, io
from tqdm import tqdm
import cv2
import re
import os

# --- НАСТРОЙКИ ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'kelly_auranofin'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'kelly_morphology'
EXISTING_CSV = RESULTS_DIR / 'kelly_morphology_data.csv'

# Список файлов, которые вылетели (из лога)
MISSING_FILES = [
    "08) Kelly Aura 2uM 6h.jpg",
    "09) Kelly ctrl 24h.jpg",
    "10) Kelly Aura 0.5uM 24h.jpg",
    "11) Kelly Aura 1uM 24h.jpg",
    "12) Kelly Aura 2uM 24h.jpg"
]

FILENAME_REGEX = re.compile(
    r"Kelly\s+(?P<drug>Aura|ctrl)\s*(?:(?P<conc>[\d\.]+)uM)?\s*(?P<time>\d+)h",
    re.IGNORECASE
)


def repair_data():
    print("--- ДОСЧЕТ ПРОПУЩЕННЫХ ФАЙЛОВ (CPU) ---")

    # 1. Загружаем существующие данные
    if EXISTING_CSV.exists():
        df_old = pd.read_csv(EXISTING_CSV)
        print(f"Загружено {len(df_old)} строк из старого CSV.")
    else:
        df_old = pd.DataFrame()

    # 2. Инициализируем модель СТРОГО НА CPU
    print("Загрузка Cellpose на CPU (это надежно)...")
    model = models.CellposeModel(gpu=False, model_type='cyto')

    new_data = []

    # 3. Обрабатываем только список потерянных
    for fname in tqdm(MISSING_FILES, desc="Обработка"):
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"Файл не найден: {fname}")
            continue

        # Парсинг имени (тот же код)
        match = FILENAME_REGEX.search(fname)
        if match:
            drug = match.group('drug').lower()
            time = match.group('time')
            conc = match.group('conc')
            if 'ctrl' in drug:
                group_name = "Control"
            else:
                group_name = f"{conc} uM" if conc else "Aura"
        else:
            group_name = "Unknown"
            time = "24"  # Fallback

        try:
            img = io.imread(str(fpath))
            # Grayscale конверсия
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Сегментация
            masks, _, _ = model.eval(img, diameter=None, channels=[0, 0])

            # Анализ
            unique_cells = np.unique(masks)[1:]
            for cell_id in unique_cells:
                mask = (masks == cell_id).astype(np.uint8)
                area = np.sum(mask)
                if area < 50 or area > 50000: continue

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours: continue
                perimeter = cv2.arcLength(contours[0], True)
                if perimeter == 0: continue

                circularity = (4 * np.pi * area) / (perimeter ** 2)
                circularity = min(circularity, 1.0)

                new_data.append({
                    'Filename': fname,
                    'Time': f"{time}h",
                    'Group': group_name,
                    'Circularity': circularity,
                    'Area': area,
                    'Aspect_Ratio': 1.0,  # Заглушка
                    'Perimeter': perimeter
                })

        except Exception as e:
            print(f"Ошибка на {fname}: {e}")

    if not new_data:
        print("Ничего нового не обработано.")
        return

    # 4. Объединяем и сохраняем
    df_new = pd.DataFrame(new_data)
    print(f"Добавлено {len(df_new)} клеток.")

    df_final = pd.concat([df_old, df_new], ignore_index=True)
    df_final.to_csv(EXISTING_CSV, index=False)
    print(f"Итоговый файл сохранен: {EXISTING_CSV} (Всего строк: {len(df_final)})")


if __name__ == "__main__":
    repair_data()
