class CellCycleDataCollector:
    """Сбор и подготовка данных для классификации фаз клеточного цикла"""

    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / 'data'
        self.features_dir = self.project_root / 'features'
        self.features_dir.mkdir(exist_ok=True)

    def extract_features_from_all_images(self, model):
        """Извлечение признаков из всех изображений"""

        all_features = []
        raw_dir = self.data_dir / 'raw'

        for img_file in raw_dir.glob('*.jpg'):
            print(f"Обработка: {img_file.name}")

            # Загрузка и сегментация
            image = io.imread(str(img_file))
            masks, _, _ = model.eval(image, diameter=None, channels=[0, 0])

            # Парсинг информации из имени файла
            info = self.parse_filename(img_file.name)

            # Извлечение признаков для каждой клетки
            for cell_id in range(1, masks.max() + 1):
                features = extract_morphological_features(image, masks, cell_id)
                features.update(info)
                features['source_image'] = img_file.name
                all_features.append(features)

        # Сохранение в CSV
        df = pd.DataFrame(all_features)
        output_path = self.features_dir / 'all_cells_features.csv'
        df.to_csv(output_path, index=False)
        print(f"Извлечено {len(df)} клеток. Данные сохранены в {output_path}")

        return df

    def parse_filename(self, filename):
        """Парсинг информации из имени файла"""
        # Пример: HCT116_WT_24h_0Gy_slide1_field01.jpg
        parts = filename.split('_')
        return {
            'genotype': parts[1],
            'time_h': parts[2].replace('h', ''),
            'dose_gy': parts[3].replace('Gy', ''),
            'slide': parts[4].replace('slide', ''),
            'field': parts[5].replace('field', '').replace('.jpg', '')
        }