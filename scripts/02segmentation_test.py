import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import find_objects
from skimage.segmentation import find_boundaries
import sys
from pathlib import Path

# Добавляем корневую директорию в путь для импорта
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from cellpose import models, io
except ImportError:
    print("Ошибка: Cellpose не установлен. Установите его командой: pip install cellpose")
    sys.exit(1)


def segment_single_image():
    """Сегментация одного тестового изображения"""
    # Определяем путь к данным
    data_dir = project_root / 'data' / 'raw'

    # Ищем тестовое изображение
    test_files = list(data_dir.glob('HCT116_WT_24h_0Gy_slide1_field01.jpg'))

    if not test_files:
        # Если файл не найден, выводим список доступных файлов
        available_files = list(data_dir.glob('HCT116_*.jpg'))
        print(f"Тестовый файл не найден в {data_dir}")
        if available_files:
            print(f"Доступные файлы:")
            for file in sorted(available_files)[:5]:  # показываем первые 5
                print(f"  - {file.name}")
            # Используем первый доступный файл
            test_file = available_files[0]
            print(f"Используем файл: {test_file.name}")
        else:
            print("Нет доступных файлов в директории")
            return None
    else:
        test_file = test_files[0]
        print(f"Найден тестовый файл: {test_file.name}")

    try:
        # Загружаем изображение
        test_img = io.imread(str(test_file))
        print(f"Изображение загружено. Размер: {test_img.shape}, тип: {test_img.dtype}")

        # Проверяем доступность GPU
        import torch
        gpu_available = torch.cuda.is_available()
        print(f"CUDA доступен: {gpu_available}")

        # Инициализируем модель
        model = models.CellposeModel(gpu=gpu_available)
        print(f"Модель Cellpose загружена (GPU: {gpu_available})")

        # Запускаем сегментацию
        print("Запуск сегментации...")
        masks, flows, styles = model.eval(
            test_img,
            diameter=None,  # автоматическое определение диаметра
            channels=[0, 0]  # изображение в градациях серого
        )

        print(f'Найдено клеток: {masks.max()}')
        print(f'Размер масок: {masks.shape}')

        # Визуализация результатов
        visualize_results(test_img, masks)

        return test_img, masks

    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_results(image, masks):
    """Визуализация результатов сегментации"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Исходное изображение
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f'Исходное изображение\nРазмер: {image.shape}')
    axes[0].axis('off')

    # Маски
    axes[1].imshow(masks, cmap='tab20')
    axes[1].set_title(f'Маски сегментации ({masks.max()} клеток)')
    axes[1].axis('off')

    # Контуры на исходном изображении
    boundaries = find_boundaries(masks, mode='outer')
    axes[2].imshow(image, cmap='gray')
    axes[2].contour(boundaries, colors='cyan', linewidths=0.5)
    axes[2].set_title('Контуры клеток')
    axes[2].axis('off')

    plt.tight_layout()

    # Сохраняем результат
    output_dir = project_root / 'results' / 'segmentation'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'test_segmentation_result.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Результат сохранён в: {output_path}")

    plt.show()

    # Дополнительная информация о сегментации
    if masks.max() > 0:
        print("\nДополнительная статистика:")
        unique, counts = np.unique(masks, return_counts=True)
        # Исключаем фон (0)
        cell_indices = unique[1:]
        cell_sizes = counts[1:]

        print(f"Количество клеток: {len(cell_indices)}")
        print(f"Средний размер клетки: {cell_sizes.mean():.0f} пикселей")
        print(f"Мин. размер: {cell_sizes.min():.0f}, Макс. размер: {cell_sizes.max():.0f}")
        print(f"Размеры клеток: {cell_sizes}")


if __name__ == "__main__":
    print("=" * 60)
    print("Запуск сегментации с помощью Cellpose")
    print("=" * 60)

    result = segment_single_image()

    if result:
        print("\nСегментация успешно завершена!")
    else:
        print("\nСегментация не удалась.")