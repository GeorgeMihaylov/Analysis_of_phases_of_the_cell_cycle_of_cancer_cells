"""
Главный скрипт для анализа радиочувствительности HCT116 клеток
Адаптировано из Google Colab для работы в PyCharm
"""

import os
import sys
from pathlib import Path
import argparse
import logging

# Добавляем src в путь Python
sys.path.append(str(Path(__file__).parent / 'src'))

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_project_structure():
    """Создает структуру папок проекта"""
    folders = [
        'data/raw_images',
        'data/masks',
        'data/cells',
        'models',
        'results',
        'notebooks',
        'src'
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        logger.info(f'Создана папка: {folder}')

    logger.info("Структура проекта создана успешно")


def check_dependencies():
    """Проверяет наличие зависимостей"""
    required_packages = [
        'torch',
        'torchvision',
        'cellpose',
        'numpy',
        'pandas',
        'matplotlib'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} установлен")
        except ImportError:
            missing.append(package)
            logger.warning(f"✗ {package} не установлен")

    if missing:
        logger.error(f"Отсутствуют пакеты: {missing}")
        logger.info("Установите их командой: pip install -r requirements.txt")
        return False
    return True


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Анализ радиочувствительности HCT116 клеток')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'segmentation', 'extraction', 'analysis'],
                        help='Режим работы: all - все шаги, segmentation - только сегментация, etc.')
    parser.add_argument('--gpu', action='store_true', help='Использовать GPU если доступно')
    parser.add_argument('--data_path', type=str, default='data/raw_images',
                        help='Путь к исходным изображениям')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Размер батча для обработки (GPU memory dependent)')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Запуск анализа радиочувствительности HCT116 клеток")
    logger.info("=" * 60)

    # 1. Настройка структуры проекта
    logger.info("Шаг 1: Настройка структуры проекта")
    setup_project_structure()

    # 2. Проверка зависимостей
    logger.info("Шаг 2: Проверка зависимостей")
    if not check_dependencies():
        logger.error("Не все зависимости установлены. Прерывание.")
        return

    # 3. Импорт модулей после проверки
    try:
        from src.segmentation import run_segmentation
        from src.cell_extraction import extract_cells
        from src.visualization import create_summary_report
    except ImportError as e:
        logger.error(f"Ошибка импорта модулей: {e}")
        logger.info("Убедитесь, что все модули в папке src/")
        return

    # 4. Запуск процессов в зависимости от режима
    if args.mode in ['all', 'segmentation']:
        logger.info("Шаг 3: Запуск сегментации клеток")
        try:
            run_segmentation(
                data_path=args.data_path,
                use_gpu=args.gpu,
                batch_size=args.batch_size
            )
        except Exception as e:
            logger.error(f"Ошибка при сегментации: {e}")
            if args.mode == 'segmentation':
                return

    if args.mode in ['all', 'extraction']:
        logger.info("Шаг 4: Извлечение отдельных клеток")
        try:
            extract_cells()
        except Exception as e:
            logger.error(f"Ошибка при извлечении клеток: {e}")
            if args.mode == 'extraction':
                return

    if args.mode in ['all', 'analysis']:
        logger.info("Шаг 5: Анализ данных и создание отчетов")
        try:
            create_summary_report()
        except Exception as e:
            logger.error(f"Ошибка при анализе: {e}")

    logger.info("=" * 60)
    logger.info("Анализ завершен успешно!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()