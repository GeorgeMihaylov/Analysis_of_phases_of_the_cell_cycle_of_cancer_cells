"""
Главный скрипт для анализа радиочувствительности HCT116 клеток
Адаптировано из Google Colab для работы в PyCharm
"""

import os
import sys
from pathlib import Path
import argparse
import logging
import warnings
warnings.filterwarnings('ignore')

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
        'data/flow_cytometry/raw',
        'data/flow_cytometry/processed',
        'data/flow_cytometry/models',
        'models',
        'results',
        'notebooks',
        'src'
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        logger.info(f'Создана папка: {folder}')

    logger.info("Структура проекта создана успешно")

    # Создаем пустой файл __init__.py в src если его нет
    init_file = Path('src/__init__.py')
    if not init_file.exists():
        init_file.touch()
        logger.info(f'Создан файл: {init_file}')

def check_dependencies():
    """Проверяет наличие зависимостей"""
    required_packages = [
        'torch',
        'torchvision',
        'cellpose',
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-image',
        'scikit-learn',
        'opencv-python',
        'seaborn',
        'tqdm',
        'Pillow',
        'scipy'
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

def run_segmentation_mode(args):
    """Запуск режима сегментации"""
    try:
        from src.segmentation import run_segmentation
        run_segmentation(
            data_path=args.data_path,
            use_gpu=args.gpu,
            batch_size=args.batch_size
        )
    except ImportError as e:
        logger.error(f"Ошибка импорта модуля сегментации: {e}")
        logger.info("Убедитесь, что файл src/segmentation.py существует")
    except Exception as e:
        logger.error(f"Ошибка при сегментации: {e}")

def run_morphology_mode():
    """Запуск режима морфологического анализа"""
    try:
        from src.cell_extraction import extract_cells_with_morphology
        extract_cells_with_morphology()
    except ImportError as e:
        logger.error(f"Ошибка импорта модуля извлечения клеток: {e}")
        logger.info("Убедитесь, что файл src/cell_extraction.py существует")
    except Exception as e:
        logger.error(f"Ошибка при морфологическом анализе: {e}")

def run_analysis_mode():
    """Запуск режима анализа и визуализации"""
    try:
        from src.visualization import create_summary_report
        create_summary_report()
    except ImportError as e:
        logger.error(f"Ошибка импорта модуля визуализации: {e}")
        logger.info("Убедитесь, что файл src/visualization.py существует")
    except Exception as e:
        logger.error(f"Ошибка при анализе: {e}")

def run_flow_setup_mode():
    """Настройка инфраструктуры для данных цитометра"""
    try:
        from src.flow_cytometry_integration import FlowCytometryDataManager
        manager = FlowCytometryDataManager()
        manager.setup_data_structure()
    except ImportError as e:
        logger.error(f"Ошибка импорта модуля интеграции цитометра: {e}")
        logger.info("Убедитесь, что файл src/flow_cytometry_integration.py существует")
    except Exception as e:
        logger.error(f"Ошибка при настройке инфраструктуры цитометра: {e}")

def run_flow_integrate_mode():
    """Интеграция данных цитометра"""
    try:
        from src.flow_cytometry_integration import FlowCytometryDataManager
        manager = FlowCytometryDataManager()

        # Загрузка данных цитометра (берем первый файл из raw)
        raw_dir = Path('data/flow_cytometry/raw')
        flow_files = list(raw_dir.glob('*.*'))

        if not flow_files:
            logger.error(f"Файлы цитометра не найдены в {raw_dir}")
            logger.info("Поместите файлы с данными цитометра в эту папку")
            return

        # Загружаем данные цитометра
        manager.load_flow_data(flow_files[0])

        # Загружаем морфологические данные
        manager.load_cell_data()

        # Объединяем данные
        manager.merge_datasets()

        # Калибруем правила
        manager.calibrate_heuristic_rules()

        # Подготавливаем датасет для ML
        manager.prepare_ml_dataset()

    except ImportError as e:
        logger.error(f"Ошибка импорта модуля интеграции цитометра: {e}")
    except Exception as e:
        logger.error(f"Ошибка при интеграции данных цитометра: {e}")

def show_project_status():
    """Показывает текущий статус проекта"""
    logger.info("=" * 60)
    logger.info("СТАТУС ПРОЕКТА")
    logger.info("=" * 60)

    # Проверяем существование папок и файлов
    paths_to_check = [
        ('data/raw_images', 'Исходные изображения'),
        ('data/masks', 'Маски сегментации'),
        ('data/cells', 'Извлеченные клетки'),
        ('results', 'Результаты анализа'),
        ('src', 'Исходный код'),
    ]

    for path, description in paths_to_check:
        if Path(path).exists():
            if Path(path).is_dir():
                files = list(Path(path).glob('*'))
                logger.info(f"✓ {description}: {len(files)} файлов")
            else:
                logger.info(f"✓ {description}: существует")
        else:
            logger.info(f"✗ {description}: отсутствует")

    # Проверяем наличие ключевых файлов
    key_files = [
        'results/cells_metadata_full.csv',
        'results/segmentation_statistics.csv',
    ]

    logger.info("\nКлючевые файлы:")
    for file in key_files:
        if Path(file).exists():
            size = Path(file).stat().st_size
            logger.info(f"✓ {file}: {size:,} байт")
        else:
            logger.info(f"✗ {file}: отсутствует")

    logger.info("\nДля полного анализа выполните:")
    logger.info("1. python main.py --mode segmentation --gpu")
    logger.info("2. python main.py --mode morphology")
    logger.info("3. python main.py --mode analysis")

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description='Анализ радиочувствительности HCT116 клеток',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py --mode all --gpu              # Все этапы с GPU
  python main.py --mode segmentation --gpu     # Только сегментация
  python main.py --mode morphology             # Только морфологический анализ
  python main.py --mode analysis               # Только анализ и визуализация
  python main.py --mode flow-setup             # Настройка для цитометра
  python main.py --mode flow-integrate         # Интеграция данных цитометра
  python main.py --mode status                 # Показать статус проекта
        """
    )

    parser.add_argument('--mode', type=str, default='status',
                       choices=['all', 'segmentation', 'morphology', 'analysis',
                                'flow-setup', 'flow-integrate', 'status'],
                       help='Режим работы')
    parser.add_argument('--gpu', action='store_true',
                       help='Использовать GPU если доступно (только для сегментации)')
    parser.add_argument('--data_path', type=str, default='data/raw_images',
                       help='Путь к исходным изображениям')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Размер батча для обработки (GPU memory dependent)')
    parser.add_argument('--setup', action='store_true',
                       help='Создать структуру папок проекта')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Запуск анализа радиочувствительности HCT116 клеток")
    logger.info("=" * 60)

    # Создание структуры проекта если нужно
    if args.setup:
        logger.info("Создание структуры проекта...")
        setup_project_structure()

    # Проверка структуры проекта
    if not Path('src').exists():
        logger.warning("Папка src/ не найдена. Создаю структуру проекта...")
        setup_project_structure()

    # Проверка зависимостей
    logger.info("Проверка зависимостей...")
    if not check_dependencies():
        logger.error("Не все зависимости установлены. Прерывание.")
        return

    # Запуск в зависимости от режима
    if args.mode == 'status':
        show_project_status()
        return

    if args.mode in ['all', 'segmentation']:
        logger.info("ЗАПУСК СЕГМЕНТАЦИИ КЛЕТОК")
        run_segmentation_mode(args)

        if args.mode == 'segmentation':
            logger.info("Сегментация завершена")
            return

    if args.mode in ['all', 'morphology']:
        logger.info("ЗАПУСК МОРФОЛОГИЧЕСКОГО АНАЛИЗА")
        run_morphology_mode()

        if args.mode == 'morphology':
            logger.info("Морфологический анализ завершен")
            return

    if args.mode in ['all', 'analysis']:
        logger.info("ЗАПУСК АНАЛИЗА И ВИЗУАЛИЗАЦИИ")
        run_analysis_mode()

        if args.mode == 'analysis':
            logger.info("Анализ завершен")
            return

    if args.mode == 'flow-setup':
        logger.info("НАСТРОЙКА ИНФРАСТРУКТУРЫ ДЛЯ ДАННЫХ ЦИТОМЕТРА")
        run_flow_setup_mode()

    if args.mode == 'flow-integrate':
        logger.info("ИНТЕГРАЦИЯ ДАННЫХ ЦИТОМЕТРА")
        run_flow_integrate_mode()

    if args.mode == 'all':
        logger.info("=" * 60)
        logger.info("ВЕСЬ ПРОЦЕСС АНАЛИЗА ЗАВЕРШЕН УСПЕШНО!")
        logger.info("=" * 60)
        logger.info("\n📋 Результаты сохранены в папках:")
        logger.info("  • data/masks/ - маски сегментации")
        logger.info("  • data/cells/ - изображения клеток")
        logger.info("  • results/ - CSV файлы с метаданными и графики")
        logger.info("\n📊 Для просмотра статистики запустите:")
        logger.info("  python main.py --mode status")

if __name__ == "__main__":
    main()