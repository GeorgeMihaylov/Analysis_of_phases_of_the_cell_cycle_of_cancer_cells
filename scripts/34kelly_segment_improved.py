# scripts/34kelly_segment_improved_fixed.py
import numpy as np
import cv2
from cellpose import models, io
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


def segment_and_save(image_path, output_dir, model_type='cyto2', use_gpu=True):
    """Исправленная сегментация с визуализацией"""

    # Создаем директории
    output_dir = Path(output_dir)
    masks_dir = output_dir / 'masks'
    visuals_dir = output_dir / 'visuals'
    data_dir = output_dir / 'data'

    for d in [masks_dir, visuals_dir, data_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Загружаем изображение
    img = io.imread(str(image_path))
    print(f"Загружено: {image_path.name}")
    print(f"  Размер: {img.shape}, Тип: {img.dtype}, Диапазон: [{img.min():.1f}, {img.max():.1f}]")

    # Препроцессинг - конвертируем в grayscale
    if img.ndim == 3:
        if img.shape[2] == 3:  # RGB
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img.shape[2] == 4:  # RGBA
            gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        else:
            gray = img.mean(axis=2)
    else:
        gray = img.copy()

    # Нормализация
    gray = gray.astype(np.float32)
    if gray.max() > gray.min():  # Избегаем деления на 0
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-6) * 255
    gray = gray.astype(np.uint8)

    # CLAHE для улучшения контраста
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Сегментация Cellpose - исправленный API
    try:
        model = models.Cellpose(gpu=use_gpu, model_type=model_type)

        # Используем правильный API для новой версии Cellpose
        channels = [0, 0]  # grayscale

        # Сегментация
        masks, flows, styles, diam = model.eval(
            gray,
            channels=channels,
            diameter=None,  # Автодетект
            flow_threshold=0.4,
            cellprob_threshold=0.0,
            resample=True,
            progress=True
        )

        print(f"  Обнаружено клеток: {len(np.unique(masks)) - 1}")
        print(f"  Использованный диаметр: {diam:.1f}")

    except Exception as e:
        print(f"  Ошибка Cellpose: {e}")
        # Попробуем альтернативный подход
        try:
            from cellpose import core
            use_GPU = core.use_gpu()
            print(f"  GPU доступен: {use_GPU}")

            model = models.CellposeModel(gpu=use_GPU, pretrained_model=model_type)
            masks, flows, styles = model.eval(
                gray,
                diameter=None,
                channels=[0, 0],
                flow_threshold=0.4,
                cellprob_threshold=0.0
            )
            diam = 30.0  # значение по умолчанию
            print(f"  Обнаружено клеток (альтернативный метод): {len(np.unique(masks)) - 1}")
        except Exception as e2:
            print(f"  Ошибка альтернативного метода: {e2}")
            return {
                'image': image_path.name,
                'n_cells': 0,
                'error': str(e)
            }

    # Сохраняем маски в правильном формате
    stem = image_path.stem
    npz_path = data_dir / f'{stem}_masks.npz'
    mask_png_path = masks_dir / f'{stem}_masks.png'
    mask_vis_path = visuals_dir / f'{stem}_masks_visual.png'
    overlay_path = visuals_dir / f'{stem}_overlay.png'

    # 1. Сохраняем NPZ с метаданными
    np.savez_compressed(
        npz_path,
        masks=masks,
        image_path=str(image_path),
        n_cells=len(np.unique(masks)) - 1,
        recommended_diameter=float(diam)
    )

    # 2. Сохраняем маску для визуализации (нормализованную)
    if masks.max() > 0 and masks.size > 0:
        mask_vis = (masks / masks.max() * 255).astype(np.uint8)
        mask_vis_colored = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
        cv2.imwrite(str(mask_vis_path), mask_vis_colored)

    # 3. Сохраняем маску как 16-битную PNG (исходные индексы)
    if masks.size > 0 and masks.max() < 65535:
        cv2.imwrite(str(mask_png_path), masks.astype(np.uint16))

    # 4. Создаем overlay
    if img.ndim == 3:
        overlay = img.copy()
        if overlay.shape[2] == 4:  # RGBA -> RGB
            overlay = overlay[:, :, :3]
    else:
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # Границы масок
    if masks.size > 0:
        boundaries = np.zeros(masks.shape, dtype=np.uint8)
        unique_labels = np.unique(masks)

        for label in unique_labels:
            if label == 0:
                continue
            mask_label = (masks == label).astype(np.uint8)
            contours, _ = cv2.findContours(mask_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(boundaries, contours, -1, 255, 1)

        if boundaries.max() > 0:
            overlay[boundaries > 0] = [255, 0, 0]  # Красные границы

    cv2.imwrite(str(overlay_path), overlay)

    # 5. Дополнительная визуализация с предпросмотром
    preview_path = visuals_dir / f'{stem}_preview.png'
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Оригинальное изображение
    axes[0].imshow(img if img.ndim == 3 else gray, cmap='gray')
    axes[0].set_title('Оригинальное изображение')
    axes[0].axis('off')

    # Сегментированные маски
    if masks.max() > 0:
        axes[1].imshow(masks, cmap='tab20c')
        axes[1].set_title(f'Маски ({len(unique_labels) - 1} клеток)')
    else:
        axes[1].imshow(gray, cmap='gray')
        axes[1].set_title('Нет обнаруженных клеток')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay с границами')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(str(preview_path), dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'image': image_path.name,
        'n_cells': int(len(np.unique(masks)) - 1),
        'mask_path': str(npz_path),
        'diameter': float(diam),
        'overlay_path': str(overlay_path),
        'preview_path': str(preview_path)
    }


def run_segmentation_pipeline():
    """Запуск пайплайна сегментации для всех изображений"""

    root = Path(__file__).parent.parent
    data_dir = root / 'data' / 'kelly_auranofin'
    output_dir = root / 'results' / 'segmentation_fixed'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Находим все изображения
    images = sorted(data_dir.glob('*.jpg'))
    print(f"Найдено {len(images)} изображений")

    results = []
    for i, img_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Обработка: {img_path.name}")
        try:
            result = segment_and_save(img_path, output_dir, use_gpu=True)
            results.append(result)
        except Exception as e:
            print(f"  Критическая ошибка: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'image': img_path.name,
                'n_cells': 0,
                'error': str(e)
            })

    # Сохраняем сводку
    df = pd.DataFrame(results)
    summary_path = output_dir / 'segmentation_summary.csv'
    df.to_csv(summary_path, index=False)

    print(f"\nСводка сохранена в: {summary_path}")
    print("\nРезультаты сегментации:")
    print(df[['image', 'n_cells']])

    # Статистика
    total_cells = df['n_cells'].sum()
    print(f"\nВсего обнаружено клеток: {total_cells}")

    # Визуализация распределения
    if total_cells > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Бар-график количества клеток
        df_sorted = df.sort_values('n_cells', ascending=False)
        bars = axes[0].bar(range(len(df_sorted)), df_sorted['n_cells'])
        axes[0].set_xlabel('Изображения')
        axes[0].set_ylabel('Количество клеток')
        axes[0].set_title('Распределение клеток по изображениям')
        axes[0].set_xticks(range(len(df_sorted)))
        axes[0].set_xticklabels([Path(name).stem[:20] + '...' for name in df_sorted['image']],
                                rotation=45, ha='right')

        # Добавляем значения на столбцы
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2., height,
                         f'{int(height)}', ha='center', va='bottom')

        # Pie chart для общего количества
        successful = df[df['n_cells'] > 0]
        failed = df[df['n_cells'] == 0]

        sizes = [len(successful), len(failed)]
        labels = [f'Успешно ({len(successful)})', f'Неудача ({len(failed)})']
        colors = ['#4CAF50', '#F44336']

        axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                    startangle=90)
        axes[1].axis('equal')
        axes[1].set_title('Статус сегментации')

        plt.tight_layout()
        stats_path = output_dir / 'segmentation_statistics.png'
        plt.savefig(stats_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nСтатистика сохранена: {stats_path}")

    return df


def check_cellpose_version():
    """Проверка версии Cellpose и доступности GPU"""
    try:
        import cellpose
        print(f"Версия Cellpose: {cellpose.__version__}")

        from cellpose import core
        use_gpu = core.use_gpu()
        print(f"GPU доступен: {use_gpu}")

        if use_gpu:
            print(f"GPU устройство: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Не определено'}")

        return True
    except Exception as e:
        print(f"Ошибка при проверке Cellpose: {e}")
        return False


if __name__ == '__main__':
    # Проверяем окружение
    print("Проверка окружения...")
    try:
        import torch

        print(f"PyTorch версия: {torch.__version__}")
        print(f"CUDA доступен: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch не установлен")

    check_cellpose_version()

    # Запускаем сегментацию
    df = run_segmentation_pipeline()