import sys
import os
import re
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from cellpose import models as cp_models, io
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
# Добавьте в начало скрипта проверку и отключение CUDA при ошибках
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Для детальной отладки

def safe_torch_cuda():
    """Безопасная инициализация CUDA"""
    try:
        import torch
        if torch.cuda.is_available():
            # Простой тест работы CUDA
            test_tensor = torch.tensor([1.0]).cuda()
            test_tensor = test_tensor * 2
            print(f"✅ CUDA работает: {torch.cuda.get_device_name(0)}")
            return True
        return False
    except Exception as e:
        print(f"⚠️ CUDA недоступна: {e}")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Отключаем CUDA
        return False

# В начале main()
CUDA_AVAILABLE = safe_torch_cuda()

# ================= CONFIG =================
class Config:
    # Используем абсолютный путь относительно этого файла
    PROJECT_ROOT = Path(__file__).parent.parent
    RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
    RESULTS_DIR = PROJECT_ROOT / 'results' / 'experiment_analysis'

    # Паттерн для парсинга имен файлов
    FILENAME_PATTERN = re.compile(
        r"(?:HCT116[_-]?)?(?P<genotype>WT|CDK8KO|CDK8)[_-](?P<time>\d+)h[_-](?P<dose>\d+)Gy",
        re.IGNORECASE
    )

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MIN_CELL_SIZE = 100

    def __init__(self):
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Ищу файлы в: {self.RAW_DATA_DIR.absolute()}")


config = Config()


# ================= 1. METADATA PARSING =================
def parse_metadata(filename):
    """Извлекает факторы эксперимента из имени файла"""
    match = config.FILENAME_PATTERN.search(filename)
    if match:
        data = match.groupdict()
        data['dose'] = int(data['dose'])
        data['time'] = data['time']
        # Нормализуем имена генотипов
        if data['genotype'].upper() in ['CDK8KO', 'CDK8']:
            data['genotype'] = 'CDK8KO'
        else:
            data['genotype'] = 'WT'
        return data
    else:
        # Fallback: пытаемся извлечь хотя бы дозу
        print(f"WARNING: Не удалось распарсить имя файла: {filename}")
        return {'genotype': 'Unknown', 'time': 'Unknown', 'dose': 0}


# ================= 2. DEEP FEATURE EXTRACTOR =================
class DeepFeatureExtractor:
    def __init__(self):
        print(f"Загрузка ResNet50 на {config.DEVICE}...")
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.feature_extractor.to(config.DEVICE).eval()

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def get_features(self, crops):
        if not crops:
            return np.array([])

        batch_tensors = []
        for crop in crops:
            try:
                pil_img = Image.fromarray(crop)
                batch_tensors.append(self.transform(pil_img))
            except Exception as e:
                print(f"Ошибка при обработке кропа: {e}")
                continue

        if not batch_tensors:
            return np.array([])

        batch = torch.stack(batch_tensors).to(config.DEVICE)

        with torch.no_grad():
            features = self.feature_extractor(batch).squeeze().cpu().numpy()

        # Если одна клетка, нужно добавить размерность
        if len(crops) == 1:
            features = features.reshape(1, -1)

        return features


# ================= 3. PROCESSING PIPELINE =================
def process_images():
    print("\n=== Инициализация моделей ===")
    segmentor = cp_models.CellposeModel(gpu=torch.cuda.is_available())
    extractor = DeepFeatureExtractor()

    all_data = []
    deep_features_list = []

    # Проверяем наличие директории
    if not config.RAW_DATA_DIR.exists():
        print(f"ОШИБКА: Директория не существует: {config.RAW_DATA_DIR.absolute()}")
        print("Создаю директорию...")
        config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Пожалуйста, поместите изображения в: {config.RAW_DATA_DIR.absolute()}")
        return pd.DataFrame(), np.array([])

    # Ищем файлы
    files = sorted(list(config.RAW_DATA_DIR.glob('*.jpg')) +
                   list(config.RAW_DATA_DIR.glob('*.png')) +
                   list(config.RAW_DATA_DIR.glob('*.tif')))

    if not files:
        print(f"\nОШИБКА: Не найдено изображений в {config.RAW_DATA_DIR.absolute()}")
        print("Поддерживаемые форматы: .jpg, .png, .tif")
        print("\nСписок файлов в директории:")
        for item in config.RAW_DATA_DIR.iterdir():
            print(f"  - {item.name}")
        return pd.DataFrame(), np.array([])

    print(f"\n=== Найдено {len(files)} файлов ===")
    for f in files[:5]:  # Показываем первые 5
        print(f"  - {f.name}")
    if len(files) > 5:
        print(f"  ... и еще {len(files) - 5}")

    print("\n=== Обработка изображений ===")
    for fpath in tqdm(files, desc="Сегментация и извлечение признаков"):
        try:
            meta = parse_metadata(fpath.name)
            img = io.imread(str(fpath))

            # 1. Сегментация
            masks, _, _ = segmentor.eval(img, diameter=None, channels=[0, 0])
            n_cells = masks.max()

            if n_cells == 0:
                print(f"\nWARNING: Не найдено клеток в {fpath.name}")
                continue

            # 2. Подготовка RGB версии для кропов
            if len(img.shape) == 2:
                image_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = img

            # 3. Извлечение кропов и метрик
            crops = []
            unique_cells = np.unique(masks)[1:]  # исключаем 0 (фон)

            for cell_id in unique_cells:
                mask_bool = (masks == cell_id)
                area = np.sum(mask_bool)

                if area < config.MIN_CELL_SIZE:
                    continue

                # Координаты bounding box
                y_coords, x_coords = np.where(mask_bool)
                y1, y2 = y_coords.min(), y_coords.max()
                x1, x2 = x_coords.min(), x_coords.max()

                # Добавляем padding
                pad = 5
                y1 = max(0, y1 - pad)
                x1 = max(0, x1 - pad)
                y2 = min(img.shape[0], y2 + pad)
                x2 = min(img.shape[1], x2 + pad)

                # Вырезаем кроп
                crop = image_rgb[y1:y2 + 1, x1:x2 + 1].copy()

                # Зануляем фон в кропе
                local_mask = mask_bool[y1:y2 + 1, x1:x2 + 1]
                local_mask_3ch = np.repeat(local_mask[:, :, np.newaxis], 3, axis=2)
                crop[~local_mask_3ch] = 0

                crops.append(crop)

                # Базовые метрики
                intensity = np.sum(img[mask_bool])
                contours, _ = cv2.findContours(
                    mask_bool.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                perimeter = cv2.arcLength(contours[0], True) if contours else 1
                circ = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

                all_data.append({
                    'filename': fpath.name,
                    'genotype': meta['genotype'],
                    'dose': meta['dose'],
                    'time': meta['time'],
                    'cell_id': cell_id,
                    'area': area,
                    'total_intensity': intensity,
                    'circularity': circ
                })

            # 4. Deep Features для этого изображения
            if crops:
                feats = extractor.get_features(crops)
                if feats.size > 0:
                    deep_features_list.append(feats)

        except Exception as e:
            print(f"\nОШИБКА при обработке {fpath.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Проверка на пустоту
    if not all_data:
        print("\nОШИБКА: Не удалось извлечь ни одной клетки из изображений!")
        return pd.DataFrame(), np.array([])

    df = pd.DataFrame(all_data)

    if not deep_features_list:
        print("\nWARNING: Deep features не извлечены. Использую только базовые метрики.")
        deep_features = np.zeros((len(df), 1))  # заглушка
    else:
        deep_features = np.vstack(deep_features_list)

    print(f"\n=== Извлечено {len(df)} клеток ===")
    print(f"Размерность deep features: {deep_features.shape}")

    return df, deep_features


# ================= 4. CALIBRATED CLASSIFICATION =================
def calibrate_and_classify(df, deep_features):
    if df.empty:
        print("Нет данных для классификации!")
        return df

    print("\n=== Калибровка и классификация ===")

    # Логарифмируем метрики
    X = np.log1p(df[['area', 'total_intensity']].values)

    # Выбираем калибровочную выборку (0 Gy)
    calibration_mask = (df['dose'] == 0)
    n_control = calibration_mask.sum()

    print(f"Контрольных клеток (0 Gy): {n_control}")

    if n_control < 50:
        print("WARNING: Мало контрольных клеток! Использую все данные для обучения.")
        X_train = X
    else:
        X_train = X[calibration_mask]

    # Обучаем GMM
    print("Обучение Gaussian Mixture Model...")
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    gmm.fit(X_train)

    # Сортируем кластеры по размеру (area)
    means = gmm.means_[:, 0]
    sorted_indices = np.argsort(means)
    mapping = {
        sorted_indices[0]: 'G1',
        sorted_indices[1]: 'S',
        sorted_indices[2]: 'G2M'
    }

    # Предсказание для всех клеток
    all_labels = gmm.predict(X)
    df['phase'] = [mapping[l] for l in all_labels]
    df['gmm_confidence'] = gmm.predict_proba(X).max(axis=1)

    # Уточнение: Митоз
    g2m_mask = (df['phase'] == 'G2M')
    mitosis_mask = g2m_mask & (df['circularity'] > 0.85)
    df.loc[mitosis_mask, 'phase'] = 'Mitosis'

    # Уточнение: SubG1 (апоптоз)
    if n_control > 0:
        g1_control = df[calibration_mask & (df['phase'] == 'G1')]['area']
        if len(g1_control) > 0:
            g1_threshold = g1_control.quantile(0.25) * 0.5
            df.loc[df['area'] < g1_threshold, 'phase'] = 'SubG1'

    print("\nРаспределение фаз:")
    print(df['phase'].value_counts())

    return df


# ================= 5. REPORTS =================
def generate_reports(df):
    if df.empty:
        return

    print("\n=== Генерация отчетов ===")

    # 1. Сводная таблица
    summary = df.groupby(['genotype', 'time', 'dose', 'phase']).size().reset_index(name='count')
    total_per_condition = df.groupby(['genotype', 'time', 'dose']).size().reset_index(name='total')
    summary = summary.merge(total_per_condition, on=['genotype', 'time', 'dose'])
    summary['percentage'] = (summary['count'] / summary['total'] * 100).round(2)

    summary.to_csv(config.RESULTS_DIR / 'phase_distribution_summary.csv', index=False)
    print(f"Таблица сохранена: {config.RESULTS_DIR / 'phase_distribution_summary.csv'}")

    # 2. Dose-Response график
    plt.figure(figsize=(12, 6))

    # G2M + Mitosis
    df['is_G2M'] = df['phase'].isin(['G2M', 'Mitosis'])
    g2_data = df.groupby(['genotype', 'time', 'dose'])['is_G2M'].mean().reset_index()
    g2_data['is_G2M'] *= 100

    for genotype in df['genotype'].unique():
        for time in df['time'].unique():
            subset = g2_data[(g2_data['genotype'] == genotype) & (g2_data['time'] == time)]
            if not subset.empty:
                plt.plot(subset['dose'], subset['is_G2M'],
                         marker='o', linewidth=2,
                         label=f"{genotype} {time}h")

    plt.xlabel('Доза облучения (Gy)', fontsize=12)
    plt.ylabel('% клеток в G2/M', fontsize=12)
    plt.title('Доза-зависимый G2 блок', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / 'dose_response_G2M.png', dpi=300)
    print(f"График сохранен: {config.RESULTS_DIR / 'dose_response_G2M.png'}")

    # 3. Stacked bar
    pivot = summary.pivot_table(
        index=['genotype', 'dose', 'time'],
        columns='phase',
        values='percentage',
        fill_value=0
    )

    phase_order = ['SubG1', 'G1', 'S', 'G2M', 'Mitosis']
    phase_order = [p for p in phase_order if p in pivot.columns]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, time_val in enumerate(df['time'].unique()):
        ax = axes[idx] if len(df['time'].unique()) > 1 else axes
        data = pivot.xs(time_val, level='time')[phase_order]
        data.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
        ax.set_title(f'Распределение фаз ({time_val}h)', fontsize=12)
        ax.set_ylabel('Процент клеток (%)')
        ax.set_xlabel('Генотип и доза')
        ax.legend(title='Фаза', bbox_to_anchor=(1.05, 1))
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / 'phase_distribution_bars.png', dpi=300)
    print(f"График сохранен: {config.RESULTS_DIR / 'phase_distribution_bars.png'}")


def main():
    print("=" * 60)
    print("АВТОМАТИЧЕСКИЙ АНАЛИЗ КЛЕТОЧНОГО ЦИКЛА")
    print("=" * 60)

    # 1. Обработка
    df, deep_feats = process_images()

    if df.empty:
        print("\nАНАЛИЗ ПРЕРВАН: нет данных для обработки")
        return

    # 2. Классификация
    df_classified = calibrate_and_classify(df, deep_feats)

    # 3. Сохранение полных данных
    output_csv = config.RESULTS_DIR / 'full_cell_data.csv'
    df_classified.to_csv(output_csv, index=False)
    print(f"\nПолные данные сохранены: {output_csv}")

    # 4. Отчеты
    generate_reports(df_classified)

    print("\n" + "=" * 60)
    print("АНАЛИЗ ЗАВЕРШЕН!")
    print(f"Результаты в папке: {config.RESULTS_DIR.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
