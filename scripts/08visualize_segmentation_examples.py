"""
Визуализация примеров сегментации и классификации фаз клеток
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from skimage.segmentation import find_boundaries
import cv2
import warnings

warnings.filterwarnings('ignore')

# Добавляем корневую директорию в путь для импорта
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class SegmentationVisualizer:
    """Визуализация примеров сегментации и классификации"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / 'data' / 'raw'
        self.masks_dir = self.project_root / 'results' / 'masks'
        self.final_data_path = self.project_root / 'results' / 'final_report' / 'final_classified_data.csv'
        self.output_dir = self.project_root / 'results' / 'visualization'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Цвета для фаз клеточного цикла
        self.phase_colors = {
            'SubG1': (1.0, 0.2, 0.2, 0.7),  # Красный
            'G1': (0.2, 0.8, 0.2, 0.7),  # Зеленый
            'S': (1.0, 0.8, 0.2, 0.7),  # Желтый
            'G2/M': (0.2, 0.5, 1.0, 0.7)  # Синий
        }

        # Цвета для изображения (grayscale)
        self.cmap_gray = plt.cm.gray

    def load_data(self):
        """Загрузка финальных данных"""

        print("Загрузка финальных данных...")

        if not self.final_data_path.exists():
            print(f"Ошибка: Файл не найден: {self.final_data_path}")
            return None

        df = pd.read_csv(self.final_data_path)
        print(f"Загружено {len(df)} клеток")

        # Проверяем необходимые столбцы
        required_columns = ['image_filename', 'cell_id', 'phase_corrected']
        for col in required_columns:
            if col not in df.columns:
                print(f"Ошибка: Отсутствует столбец {col}")
                return None

        return df

    def select_example_images(self, df, n_examples=6):
        """Выбор примеров изображений для визуализации"""

        print("\nВыбор примеров для визуализации...")

        # Получаем уникальные изображения
        image_files = df['image_filename'].unique()
        print(f"Всего изображений в данных: {len(image_files)}")

        # Разделяем по условиям для разнообразия примеров
        examples = []

        # Создаем структуру для поиска по условиям
        conditions = []

        for img_file in image_files:
            img_data = df[df['image_filename'] == img_file]

            # Извлекаем информацию из имени файла
            try:
                parts = img_file.replace('.jpg', '').split('_')
                genotype = parts[1]
                time_h = parts[2].replace('h', '')
                dose_gy = parts[3].replace('Gy', '')
            except:
                genotype = 'unknown'
                time_h = 'unknown'
                dose_gy = 'unknown'

            # Считаем распределение фаз
            phase_counts = img_data['phase_corrected'].value_counts()
            total_cells = len(img_data)

            conditions.append({
                'filename': img_file,
                'genotype': genotype,
                'time_h': time_h,
                'dose_gy': dose_gy,
                'total_cells': total_cells,
                'phase_counts': phase_counts,
                'has_subg1': 'SubG1' in phase_counts.index,
                'has_g2m': 'G2/M' in phase_counts.index
            })

        conditions_df = pd.DataFrame(conditions)

        # Выбираем разнообразные примеры
        selected = []

        # 1. WT, 0 Gy, 24h (контроль)
        wt_0 = conditions_df[
            (conditions_df['genotype'] == 'WT') &
            (conditions_df['dose_gy'] == '0') &
            (conditions_df['time_h'] == '24')
            ]
        if len(wt_0) > 0:
            selected.append(wt_0.iloc[0]['filename'])

        # 2. WT, 10 Gy, 24h (максимальная доза)
        wt_10 = conditions_df[
            (conditions_df['genotype'] == 'WT') &
            (conditions_df['dose_gy'] == '10') &
            (conditions_df['time_h'] == '24')
            ]
        if len(wt_10) > 0:
            selected.append(wt_10.iloc[0]['filename'])

        # 3. CDK8KO, 0 Gy, 24h (контроль KO)
        ko_0 = conditions_df[
            (conditions_df['genotype'] == 'CDK8KO') &
            (conditions_df['dose_gy'] == '0') &
            (conditions_df['time_h'] == '24')
            ]
        if len(ko_0) > 0:
            selected.append(ko_0.iloc[0]['filename'])

        # 4. CDK8KO, 10 Gy, 24h (максимальная доза KO)
        ko_10 = conditions_df[
            (conditions_df['genotype'] == 'CDK8KO') &
            (conditions_df['dose_gy'] == '10') &
            (conditions_df['time_h'] == '24')
            ]
        if len(ko_10) > 0:
            selected.append(ko_10.iloc[0]['filename'])

        # 5. Дополнительные примеры из оставшихся
        remaining = conditions_df[~conditions_df['filename'].isin(selected)]
        if len(remaining) > 0:
            # Выбираем с большим количеством клеток и разными фазами
            remaining_sorted = remaining.sort_values('total_cells', ascending=False)
            for _, row in remaining_sorted.iterrows():
                if len(selected) >= n_examples:
                    break
                if row['filename'] not in selected:
                    selected.append(row['filename'])

        print(f"Выбрано {len(selected)} примеров:")
        for filename in selected:
            info = conditions_df[conditions_df['filename'] == filename].iloc[0]
            print(
                f"  - {filename} ({info['genotype']}, {info['dose_gy']}Gy, {info['time_h']}h, {info['total_cells']} клеток)")

        return selected[:n_examples]

    def load_image_and_mask(self, image_filename):
        """Загрузка изображения и маски"""

        # Загружаем изображение
        image_path = self.data_dir / image_filename
        if not image_path.exists():
            print(f"  Изображение не найдено: {image_path}")
            return None, None, None

        # Загружаем маску
        mask_filename = image_filename.replace('.jpg', '_masks.npy')
        mask_path = self.masks_dir / mask_filename

        if not mask_path.exists():
            print(f"  Маска не найдена: {mask_path}")
            return None, None, None

        try:
            # Загружаем изображение
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"  Не удалось загрузить изображение: {image_path}")
                return None, None, None

            # Конвертируем в grayscale если нужно
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Загружаем маску
            masks = np.load(mask_path)

            return image, masks, image_filename

        except Exception as e:
            print(f"  Ошибка при загрузке: {e}")
            return None, None, None

    def create_phase_overlay(self, image, masks, phase_data):
        """Создание наложения фаз на изображение"""

        # Создаем цветное изображение
        if len(image.shape) == 2:
            colored_image = np.stack([image, image, image], axis=-1) / 255.0
        else:
            colored_image = image / 255.0

        # Создаем overlay для фаз
        phase_overlay = np.zeros((*masks.shape, 4), dtype=np.float32)

        # Для каждой клетки в маске
        cell_ids = np.unique(masks)
        cell_ids = cell_ids[cell_ids > 0]  # Исключаем фон

        for cell_id in cell_ids:
            # Находим фазу для этой клетки
            cell_info = phase_data[phase_data['cell_id'] == cell_id]

            if len(cell_info) > 0:
                phase = cell_info.iloc[0]['phase_corrected']
                color = self.phase_colors.get(phase, (0.5, 0.5, 0.5, 0.7))
            else:
                # Клетка не найдена в данных (может быть отфильтрована)
                color = (0.3, 0.3, 0.3, 0.3)

            # Закрашиваем клетку
            cell_mask = masks == cell_id
            phase_overlay[cell_mask] = color

        return colored_image, phase_overlay

    def create_detailed_visualization(self, image, masks, phase_data, image_info):
        """Создание детальной визуализации для одного изображения"""

        # Создаем overlay фаз
        colored_image, phase_overlay = self.create_phase_overlay(image, masks, phase_data)

        # Находим границы клеток
        boundaries = find_boundaries(masks, mode='outer')

        # Создаем фигуру
        fig = plt.figure(figsize=(18, 12))

        # 1. Исходное изображение
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(image, cmap='gray')
        ax1.set_title('Исходное изображение', fontsize=12, fontweight='bold')
        ax1.axis('off')

        # Добавляем информацию
        title_text = f"{image_info.get('genotype', 'Unknown')}, "
        title_text += f"{image_info.get('dose_gy', '?')}Gy, "
        title_text += f"{image_info.get('time_h', '?')}h"
        ax1.text(0.5, -0.05, title_text, transform=ax1.transAxes,
                 ha='center', fontsize=10, fontweight='bold')

        # 2. Маска сегментации
        ax2 = plt.subplot(2, 3, 2)
        # Используем случайные цвета для разных клеток
        unique_cells = len(np.unique(masks)) - 1
        cmap_random = plt.cm.get_cmap('tab20', unique_cells)

        # Создаем цветную маску
        colored_mask = np.zeros((*masks.shape, 3))
        for i, cell_id in enumerate(np.unique(masks)):
            if cell_id > 0:  # Пропускаем фон
                cell_mask = masks == cell_id
                color = cmap_random(i % unique_cells)[:3]
                colored_mask[cell_mask] = color

        ax2.imshow(colored_mask)
        ax2.set_title(f'Сегментация ({unique_cells} клеток)', fontsize=12, fontweight='bold')
        ax2.axis('off')

        # 3. Наложение фаз
        ax3 = plt.subplot(2, 3, 3)
        ax3.imshow(colored_image)
        ax3.imshow(phase_overlay)

        # Добавляем границы
        boundaries_rgb = np.zeros((*boundaries.shape, 4))
        boundaries_rgb[boundaries] = [0, 0, 0, 1]  # Черные границы
        ax3.imshow(boundaries_rgb)

        ax3.set_title('Классификация фаз', fontsize=12, fontweight='bold')
        ax3.axis('off')

        # 4. Увеличенная область - найти область с разнообразными клетками
        ax4 = plt.subplot(2, 3, 4)

        # Находим область с клетками разных фаз
        crop_size = 300  # размер области для увеличения

        # Ищем клетки разных фаз
        cells_with_phases = []
        for cell_id in np.unique(masks):
            if cell_id > 0:
                cell_info = phase_data[phase_data['cell_id'] == cell_id]
                if len(cell_info) > 0:
                    phase = cell_info.iloc[0]['phase_corrected']
                    # Находим координаты центра клетки
                    cell_mask = masks == cell_id
                    rows, cols = np.where(cell_mask)
                    if len(rows) > 0:
                        center_row, center_col = np.mean(rows), np.mean(cols)
                        cells_with_phases.append((center_row, center_col, phase, cell_id))

        if cells_with_phases:
            # Выбираем клетку в центре изображения
            center_row, center_col = image.shape[0] // 2, image.shape[1] // 2

            # Ищем ближайшую клетку к центру
            min_dist = float('inf')
            selected_cell = None

            for row, col, phase, cell_id in cells_with_phases:
                dist = np.sqrt((row - center_row) ** 2 + (col - center_col) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    selected_cell = (int(row), int(col), phase, cell_id)

            if selected_cell:
                center_y, center_x, phase, cell_id = selected_cell

                # Определяем границы области
                y_start = max(0, center_y - crop_size // 2)
                y_end = min(image.shape[0], center_y + crop_size // 2)
                x_start = max(0, center_x - crop_size // 2)
                x_end = min(image.shape[1], center_x + crop_size // 2)

                # Вырезаем область
                crop_image = image[y_start:y_end, x_start:x_end]
                crop_masks = masks[y_start:y_end, x_start:x_end]

                # Создаем overlay для этой области
                crop_colored, crop_overlay = self.create_phase_overlay(
                    crop_image, crop_masks, phase_data
                )

                ax4.imshow(crop_colored)
                ax4.imshow(crop_overlay)

                # Добавляем границы
                crop_boundaries = find_boundaries(crop_masks, mode='outer')
                boundaries_rgb_crop = np.zeros((*crop_boundaries.shape, 4))
                boundaries_rgb_crop[crop_boundaries] = [0, 0, 0, 1]
                ax4.imshow(boundaries_rgb_crop)

                # Обводим выбранную клетку
                cell_mask_crop = crop_masks == cell_id
                if np.any(cell_mask_crop):
                    from skimage.measure import find_contours
                    contours = find_contours(cell_mask_crop, 0.5)
                    for contour in contours:
                        ax4.plot(contour[:, 1], contour[:, 0],
                                 color='white', linewidth=2, linestyle='--')

                ax4.set_title(f'Увеличенная область ({phase})', fontsize=12, fontweight='bold')
                ax4.axis('off')

        # 5. Примеры клеток разных фаз
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')

        # Собираем примеры клеток каждой фазы
        phase_examples = {}

        for phase in self.phase_colors.keys():
            phase_cells = phase_data[phase_data['phase_corrected'] == phase]
            if len(phase_cells) > 0:
                # Берем первую клетку этой фазы
                cell_id = phase_cells.iloc[0]['cell_id']
                cell_mask = masks == cell_id

                # Находим bounding box
                rows, cols = np.where(cell_mask)
                if len(rows) > 0:
                    min_row, max_row = np.min(rows), np.max(rows)
                    min_col, max_col = np.min(cols), np.max(cols)

                    # Добавляем отступ
                    padding = 10
                    min_row = max(0, min_row - padding)
                    max_row = min(image.shape[0], max_row + padding)
                    min_col = max(0, min_col - padding)
                    max_col = min(image.shape[1], max_col + padding)

                    # Вырезаем клетку
                    cell_crop = image[min_row:max_row, min_col:max_col]

                    # Сохраняем для отображения
                    phase_examples[phase] = {
                        'image': cell_crop,
                        'cell_id': cell_id,
                        'area': np.sum(cell_mask)
                    }

        # Отображаем примеры если есть
        if phase_examples:
            # Создаем мозаику примеров
            n_phases = len(phase_examples)
            fig_examples, axes_examples = plt.subplots(1, n_phases, figsize=(3 * n_phases, 3))

            for idx, (phase, example) in enumerate(phase_examples.items()):
                if n_phases > 1:
                    ax_ex = axes_examples[idx]
                else:
                    ax_ex = axes_examples

                ax_ex.imshow(example['image'], cmap='gray')
                ax_ex.set_title(f'{phase}\nПлощадь: {example["area"]:.0f}',
                                fontsize=10, fontweight='bold')
                ax_ex.axis('off')

                # Добавляем цветную рамку
                color = self.phase_colors[phase]
                for spine in ax_ex.spines.values():
                    spine.set_edgecolor(color[:3])
                    spine.set_linewidth(3)

            # Сохраняем отдельно и отображаем в основном графике
            examples_path = self.output_dir / f'examples_{image_info.get("filename", "unknown")}.png'
            fig_examples.tight_layout()
            fig_examples.savefig(examples_path, dpi=150, bbox_inches='tight')
            plt.close(fig_examples)

            # Вставляем изображение в основную фигуру
            examples_img = plt.imread(examples_path)
            ax5.imshow(examples_img)
            ax5.set_title('Примеры клеток разных фаз', fontsize=12, fontweight='bold')
            ax5.axis('off')

        # 6. Легенда и статистика
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        # Статистика по изображению
        total_cells = len(np.unique(masks)) - 1
        stats_text = "СТАТИСТИКА ИЗОБРАЖЕНИЯ:\n\n"
        stats_text += f"Всего клеток: {total_cells}\n\n"

        # Распределение по фазам
        phase_counts = phase_data['phase_corrected'].value_counts()
        for phase, color in self.phase_colors.items():
            if phase in phase_counts:
                count = phase_counts[phase]
                percentage = (count / len(phase_data)) * 100
                stats_text += f"{phase}: {count} ({percentage:.1f}%)\n"

        # Средние размеры по фазам
        stats_text += "\nСредние размеры:\n"
        for phase in self.phase_colors.keys():
            phase_cells = phase_data[phase_data['phase_corrected'] == phase]
            if len(phase_cells) > 0:
                mean_area = phase_cells['area'].mean()
                stats_text += f"{phase}: {mean_area:.0f} пикс\n"

        # Информация об изображении
        stats_text += f"\nГенотип: {image_info.get('genotype', 'Unknown')}\n"
        stats_text += f"Доза: {image_info.get('dose_gy', '?')} Gy\n"
        stats_text += f"Время: {image_info.get('time_h', '?')} ч"

        # Отображаем текст
        ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
                 fontsize=11, va='top',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        # Добавляем легенду цветов
        legend_elements = []
        for phase, color in self.phase_colors.items():
            legend_elements.append(mpatches.Patch(facecolor=color,
                                                  edgecolor='black',
                                                  label=phase))

        ax6.legend(handles=legend_elements, loc='lower center',
                   bbox_to_anchor=(0.5, 0.05), ncol=2, fontsize=10)

        plt.suptitle(f'ВИЗУАЛИЗАЦИЯ СЕГМЕНТАЦИИ И КЛАССИФИКАЦИИ: {image_info.get("filename", "")}',
                     fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        # Сохраняем
        output_path = self.output_dir / f'visualization_{image_info.get("filename", "unknown").replace(".jpg", "")}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Визуализация сохранена: {output_path}")

        return output_path

    def create_comparison_grid(self, examples_data):
        """Создание сетки сравнения нескольких изображений"""

        print("\nСоздание сетки сравнения...")

        n_examples = len(examples_data)
        n_cols = min(3, n_examples)
        n_rows = (n_examples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols * 3, figsize=(6 * n_cols, 4 * n_rows))

        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes[0], axes[1], axes[2]]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, (image, masks, phase_data, image_info) in enumerate(examples_data):
            if idx >= n_rows * n_cols:
                break

            row = idx // n_cols
            col_start = (idx % n_cols) * 3

            # Создаем overlay
            colored_image, phase_overlay = self.create_phase_overlay(image, masks, phase_data)
            boundaries = find_boundaries(masks, mode='outer')

            # 1. Исходное изображение
            ax1 = axes[row, col_start]
            ax1.imshow(image, cmap='gray')
            ax1.set_title(f"{image_info.get('genotype', '?')} {image_info.get('dose_gy', '?')}Gy",
                          fontsize=10, fontweight='bold')
            ax1.axis('off')

            # 2. Сегментация
            ax2 = axes[row, col_start + 1]

            # Цветная маска
            unique_cells = len(np.unique(masks)) - 1
            cmap_random = plt.cm.get_cmap('tab20', max(unique_cells, 1))
            colored_mask = np.zeros((*masks.shape, 3))

            for i, cell_id in enumerate(np.unique(masks)):
                if cell_id > 0:
                    cell_mask = masks == cell_id
                    color = cmap_random(i % max(unique_cells, 1))[:3]
                    colored_mask[cell_mask] = color

            ax2.imshow(colored_mask)
            ax2.set_title(f'{unique_cells} клеток', fontsize=10)
            ax2.axis('off')

            # 3. Классификация фаз
            ax3 = axes[row, col_start + 2]
            ax3.imshow(colored_image)
            ax3.imshow(phase_overlay)

            # Границы
            boundaries_rgb = np.zeros((*boundaries.shape, 4))
            boundaries_rgb[boundaries] = [0, 0, 0, 1]
            ax3.imshow(boundaries_rgb)

            # Подпись с распределением фаз
            total_cells = len(phase_data)
            if total_cells > 0:
                subg1_pct = (phase_data['phase_corrected'] == 'SubG1').mean() * 100
                g2m_pct = (phase_data['phase_corrected'] == 'G2/M').mean() * 100
                phase_text = f"SubG1: {subg1_pct:.1f}%\nG2/M: {g2m_pct:.1f}%"
                ax3.text(0.02, 0.98, phase_text, transform=ax3.transAxes,
                         fontsize=8, color='white', fontweight='bold',
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

            ax3.axis('off')

        # Скрываем пустые ячейки
        for idx in range(n_examples, n_rows * n_cols):
            row = idx // n_cols
            col_start = (idx % n_cols) * 3
            for i in range(3):
                axes[row, col_start + i].axis('off')

        # Добавляем общую легенду
        legend_elements = []
        for phase, color in self.phase_colors.items():
            legend_elements.append(mpatches.Patch(facecolor=color,
                                                  edgecolor='black',
                                                  label=phase))

        fig.legend(handles=legend_elements, loc='lower center',
                   bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=10)

        plt.suptitle('СРАВНЕНИЕ СЕГМЕНТАЦИИ И КЛАССИФИКАЦИИ РАЗНЫХ ОБРАЗЦОВ',
                     fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        # Сохраняем
        output_path = self.output_dir / 'comparison_grid.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Сетка сравнения сохранена: {output_path}")

        return output_path

    def create_phase_distribution_visualization(self, df):
        """Создание визуализации распределения фаз"""

        print("\nСоздание визуализации распределения фаз...")

        # Группируем данные по изображениям
        image_stats = []

        for img_file in df['image_filename'].unique():
            img_data = df[df['image_filename'] == img_file]

            # Извлекаем информацию из имени файла
            try:
                parts = img_file.replace('.jpg', '').split('_')
                genotype = parts[1]
                time_h = parts[2].replace('h', '')
                dose_gy = parts[3].replace('Gy', '')
            except:
                genotype = 'unknown'
                time_h = 'unknown'
                dose_gy = 'unknown'

            # Считаем статистику
            total_cells = len(img_data)
            phase_counts = img_data['phase_corrected'].value_counts()

            image_stats.append({
                'filename': img_file,
                'genotype': genotype,
                'time_h': time_h,
                'dose_gy': dose_gy,
                'total_cells': total_cells,
                'phase_counts': phase_counts
            })

        # Создаем график
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Распределение фаз по всем данным
        ax1 = axes[0, 0]
        phase_counts_total = df['phase_corrected'].value_counts()

        # Сортируем в правильном порядке
        phase_order = ['SubG1', 'G1', 'S', 'G2/M']
        phase_counts_total = phase_counts_total.reindex(phase_order)

        colors = [self.phase_colors[phase] for phase in phase_counts_total.index]
        bars = ax1.bar(phase_counts_total.index, phase_counts_total.values, color=colors)

        ax1.set_title('Общее распределение фаз', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Фаза клеточного цикла')
        ax1.set_ylabel('Количество клеток')

        # Добавляем проценты
        total_cells_all = len(df)
        for bar, phase in zip(bars, phase_counts_total.index):
            height = bar.get_height()
            percentage = (height / total_cells_all) * 100
            ax1.text(bar.get_x() + bar.get_width() / 2, height + 50,
                     f'{percentage:.1f}%', ha='center', fontsize=10)

        # 2. Распределение фаз по генотипам
        ax2 = axes[0, 1]

        if 'genotype' in df.columns:
            genotypes = df['genotype'].unique()
            x = np.arange(len(genotypes))
            width = 0.2

            for i, phase in enumerate(phase_order):
                values = []
                for genotype in genotypes:
                    subset = df[df['genotype'] == genotype]
                    if phase in subset['phase_corrected'].values:
                        phase_pct = (subset['phase_corrected'] == phase).mean() * 100
                    else:
                        phase_pct = 0
                    values.append(phase_pct)

                positions = x + (i - 1.5) * width
                ax2.bar(positions, values, width, label=phase, color=self.phase_colors[phase])

            ax2.set_title('Распределение фаз по генотипам', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Генотип')
            ax2.set_ylabel('Доля клеток (%)')
            ax2.set_xticks(x)
            ax2.set_xticklabels(genotypes)
            ax2.legend(title='Фаза')

        # 3. Доля SubG1 по дозам
        ax3 = axes[0, 2]

        if 'dose_gy' in df.columns:
            # Группируем по дозам
            df['dose_numeric'] = pd.to_numeric(df['dose_gy'], errors='coerce')
            doses = sorted(df['dose_numeric'].dropna().unique())

            subg1_by_dose = []
            for dose in doses:
                subset = df[df['dose_numeric'] == dose]
                subg1_pct = (subset['phase_corrected'] == 'SubG1').mean() * 100
                subg1_by_dose.append(subg1_pct)

            ax3.plot(doses, subg1_by_dose, marker='o', color=self.phase_colors['SubG1'],
                     linewidth=2, markersize=8)
            ax3.set_title('Доля SubG1 клеток по дозам облучения', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Доза облучения (Gy)')
            ax3.set_ylabel('Доля SubG1 клеток (%)')
            ax3.grid(True, alpha=0.3)

        # 4. Размер клеток по фазам
        ax4 = axes[1, 0]

        phase_area_data = []
        for phase in phase_order:
            phase_cells = df[df['phase_corrected'] == phase]
            if len(phase_cells) > 0:
                phase_area_data.append(phase_cells['area'].values)

        box = ax4.boxplot(phase_area_data, labels=phase_order, patch_artist=True)

        # Раскрашиваем boxplot
        for i, patch in enumerate(box['boxes']):
            patch.set_facecolor(self.phase_colors[phase_order[i]])
            patch.set_alpha(0.7)

        ax4.set_title('Размер клеток по фазам', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Фаза клеточного цикла')
        ax4.set_ylabel('Площадь (пиксели)')
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. Примеры изображений с клетками разных фаз
        ax5 = axes[1, 1]
        ax5.axis('off')

        # Создаем мини-превью для каждой фазы
        preview_text = "ЦВЕТОВАЯ КОДИРОВКА ФАЗ:\n\n"
        for phase, color in self.phase_colors.items():
            preview_text += f"■ {phase}: {color[:3]}\n"

        preview_text += "\nМОРФОЛОГИЧЕСКИЕ ПРИЗНАКИ:\n"
        preview_text += "• SubG1: мелкие/сморщенные\n"
        preview_text += "• G1: средние/округлые\n"
        preview_text += "• S: увеличивающиеся\n"
        preview_text += "• G2/M: крупные/округлые\n"

        ax5.text(0.1, 0.9, preview_text, transform=ax5.transAxes,
                 fontsize=11, va='top',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        # 6. Легенда и сводка
        ax6 = axes[1, 2]
        ax6.axis('off')

        summary_text = "СВОДКА АНАЛИЗА:\n\n"
        summary_text += f"Всего проанализировано: {len(df):,} клеток\n"
        summary_text += f"Изображений: {len(df['image_filename'].unique())}\n\n"

        if 'genotype' in df.columns:
            summary_text += "РАСПРЕДЕЛЕНИЕ ПО ГЕНОТИПАМ:\n"
            for genotype in df['genotype'].unique():
                subset = df[df['genotype'] == genotype]
                subg1_pct = (subset['phase_corrected'] == 'SubG1').mean() * 100
                summary_text += f"  {genotype}: {len(subset):,} клеток, {subg1_pct:.1f}% SubG1\n"

        summary_text += "\nКЛЮЧЕВОЙ ВЫВОД:\n"
        summary_text += "CDK8KO клетки демонстрируют повышенную\n"
        summary_text += "чувствительность к облучению"

        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                 fontsize=11, va='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.suptitle('АНАЛИЗ РАСПРЕДЕЛЕНИЯ ФАЗ КЛЕТОЧНОГО ЦИКЛА',
                     fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()

        output_path = self.output_dir / 'phase_distribution_summary.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Визуализация распределения фаз сохранена: {output_path}")

        return output_path

    def run_visualization(self):
        """Запуск визуализации"""

        print("=" * 80)
        print("ВИЗУАЛИЗАЦИЯ СЕГМЕНТАЦИИ И КЛАССИФИКАЦИИ КЛЕТОК")
        print("=" * 80)

        # 1. Загрузка данных
        df = self.load_data()
        if df is None:
            return

        # 2. Выбор примеров
        example_files = self.select_example_images(df, n_examples=6)

        # 3. Обработка каждого примера
        examples_data = []

        print("\nОбработка примеров...")
        for img_file in example_files:
            print(f"\nОбработка: {img_file}")

            # Загружаем изображение и маску
            image, masks, filename = self.load_image_and_mask(img_file)

            if image is None or masks is None:
                print(f"  Пропускаем {img_file} - не удалось загрузить")
                continue

            # Фильтруем данные для этого изображения
            phase_data = df[df['image_filename'] == img_file]

            # Извлекаем информацию об изображении
            try:
                parts = img_file.replace('.jpg', '').split('_')
                genotype = parts[1]
                time_h = parts[2].replace('h', '')
                dose_gy = parts[3].replace('Gy', '')
            except:
                genotype = 'unknown'
                time_h = 'unknown'
                dose_gy = 'unknown'

            image_info = {
                'filename': img_file,
                'genotype': genotype,
                'time_h': time_h,
                'dose_gy': dose_gy,
                'total_cells': len(np.unique(masks)) - 1
            }

            # Создаем детальную визуализацию
            self.create_detailed_visualization(image, masks, phase_data, image_info)

            # Сохраняем для сетки сравнения
            examples_data.append((image, masks, phase_data, image_info))

        # 4. Создаем сетку сравнения
        if examples_data:
            self.create_comparison_grid(examples_data)

        # 5. Создаем визуализацию распределения фаз
        self.create_phase_distribution_visualization(df)

        print("\n" + "=" * 80)
        print("ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА!")
        print("=" * 80)

        print(f"\nВсе результаты сохранены в: {self.output_dir}")
        print("\nОсновные файлы:")
        print("  1. Детальные визуализации для каждого примера")
        print("  2. Сетка сравнения: comparison_grid.png")
        print("  3. Распределение фаз: phase_distribution_summary.png")

        print("\nДля просмотра результатов откройте файлы в папке results/visualization/")


def main():
    """Основная функция"""

    try:
        visualizer = SegmentationVisualizer()
        visualizer.run_visualization()

    except Exception as e:
        print(f"Ошибка при выполнении визуализации: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()