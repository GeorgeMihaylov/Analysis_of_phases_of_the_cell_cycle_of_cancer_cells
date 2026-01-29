"""
Классификация клеток по фазам клеточного цикла на основе морфологических признаков
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings

warnings.filterwarnings('ignore')

# Добавляем корневую директорию в путь для импорта
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class CellCycleClassifier:
    """Классификатор фаз клеточного цикла"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / 'results' / 'features'
        self.output_dir = self.project_root / 'results' / 'cell_cycle'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Признаки для классификации
        self.feature_columns = [
            'area', 'circularity', 'aspect_ratio', 'solidity', 'eccentricity',
            'intensity_mean', 'intensity_std', 'intensity_range',
            'contrast', 'homogeneity', 'energy',
            'hu_moment_1', 'hu_moment_2', 'hu_moment_3'
        ]

        # Соответствие кластеров фазам (будет определено после анализа)
        self.cluster_to_phase = {}

    def load_data(self):
        """Загрузка данных с признаками клеток"""
        data_path = self.data_dir / 'all_cells_features.csv'

        if not data_path.exists():
            print(f"Ошибка: Файл с данными не найден: {data_path}")
            return None

        print(f"Загрузка данных из {data_path}")
        df = pd.read_csv(data_path)
        print(f"Загружено {len(df)} клеток, {len(df.columns)} признаков")

        # Проверяем наличие необходимых признаков
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            print(f"Внимание: Отсутствуют признаки: {missing_features}")
            # Используем только доступные признаки
            self.feature_columns = [col for col in self.feature_columns if col in df.columns]

        return df

    def preprocess_data(self, df):
        """Предобработка данных"""

        print(f"\nПредобработка данных...")

        # Удаляем строки с пропущенными значениями в ключевых признаках
        initial_count = len(df)
        df_clean = df.dropna(subset=self.feature_columns)
        print(f"Удалено строк с пропусками: {initial_count - len(df_clean)}")

        # Удаляем выбросы по площади (слишком маленькие или слишком большие клетки)
        Q1 = df_clean['area'].quantile(0.01)
        Q3 = df_clean['area'].quantile(0.99)
        df_filtered = df_clean[(df_clean['area'] >= Q1) & (df_clean['area'] <= Q3)]
        print(f"Удалено выбросов по площади: {len(df_clean) - len(df_filtered)}")

        # Сохраняем индексы отфильтрованных строк
        self.original_indices = df_filtered.index

        # Подготовка данных для кластеризации
        X = df_filtered[self.feature_columns].copy()

        # Логарифмируем площадь для нормализации распределения
        if 'area' in X.columns:
            X['area_log'] = np.log1p(X['area'])
            # Удаляем исходный признак площади
            X = X.drop('area', axis=1)
            # Обновляем список признаков
            self.feature_columns_processed = [col if col != 'area' else 'area_log'
                                              for col in self.feature_columns]

        # Масштабирование признаков
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        print(f"После предобработки: {len(X_scaled)} клеток, {X_scaled.shape[1]} признаков")

        return X_scaled, df_filtered

    def determine_optimal_clusters(self, X_scaled):
        """Определение оптимального количества кластеров"""

        print("\nОпределение оптимального количества кластеров...")

        silhouette_scores = []
        calinski_scores = []
        davies_scores = []
        k_values = range(2, 8)  # Пробуем от 2 до 7 кластеров

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)

            if len(np.unique(labels)) > 1:
                silhouette_scores.append(silhouette_score(X_scaled, labels))
                calinski_scores.append(calinski_harabasz_score(X_scaled, labels))
                davies_scores.append(davies_bouldin_score(X_scaled, labels))
            else:
                silhouette_scores.append(0)
                calinski_scores.append(0)
                davies_scores.append(0)

        # Визуализация метрик
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(k_values, silhouette_scores, marker='o')
        axes[0].set_xlabel('Количество кластеров')
        axes[0].set_ylabel('Silhouette Score')
        axes[0].set_title('Silhouette Score')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(k_values, calinski_scores, marker='o', color='orange')
        axes[1].set_xlabel('Количество кластеров')
        axes[1].set_ylabel('Calinski-Harabasz Score')
        axes[1].set_title('Calinski-Harabasz Score')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(k_values, davies_scores, marker='o', color='green')
        axes[2].set_xlabel('Количество кластеров')
        axes[2].set_ylabel('Davies-Bouldin Score')
        axes[2].set_title('Davies-Bouldin Score (меньше = лучше)')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'optimal_clusters.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Выбираем оптимальное k (4 для фаз клеточного цикла)
        optimal_k = 4
        print(f"Выбрано количество кластеров: {optimal_k} (соответствует фазам клеточного цикла)")

        return optimal_k

    def perform_clustering(self, X_scaled, n_clusters=4):
        """Выполнение кластеризации"""

        print(f"\nВыполнение кластеризации с {n_clusters} кластерами...")

        # Пробуем разные алгоритмы кластеризации
        algorithms = {
            'KMeans': KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
            'GaussianMixture': GaussianMixture(n_components=n_clusters, random_state=42),
        }

        best_labels = None
        best_score = -1
        best_algo = None

        for algo_name, model in algorithms.items():
            print(f"  Тестируем {algo_name}...")

            if algo_name == 'GaussianMixture':
                model.fit(X_scaled)
                labels = model.predict(X_scaled)
                probabilities = model.predict_proba(X_scaled)
                # Сохраняем вероятности для анализа уверенности
                self.cluster_probabilities = probabilities
            else:
                labels = model.fit_predict(X_scaled)

            # Оцениваем качество кластеризации
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X_scaled, labels)
                print(f"    Silhouette Score: {score:.3f}")

                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_algo = algo_name
            else:
                print(f"    Только один кластер")

        print(f"Выбран алгоритм: {best_algo} с score={best_score:.3f}")

        return best_labels

    def interpret_clusters(self, df_filtered, cluster_labels):
        """Интерпретация кластеров и назначение фаз клеточного цикла"""

        print("\nАнализ кластеров...")

        # Добавляем метки кластеров к данным
        df_analysis = df_filtered.copy()
        df_analysis['cluster'] = cluster_labels

        # Анализ средних значений признаков по кластерам
        cluster_stats = df_analysis.groupby('cluster')[self.feature_columns].agg(['mean', 'std'])

        print("\nСредние значения ключевых признаков по кластерам:")
        key_features = ['area', 'circularity', 'intensity_mean', 'aspect_ratio']
        for feature in key_features:
            if feature in df_analysis.columns:
                means = df_analysis.groupby('cluster')[feature].mean()
                print(f"\n{feature}:")
                for cluster, mean_val in means.items():
                    print(f"  Кластер {cluster}: {mean_val:.2f}")

        # Определяем фазы на основе характеристик кластеров
        print("\nОпределение фаз клеточного цикла для кластеров...")

        # Сначала сортируем кластеры по площади (ожидаемый порядок: SubG1 < G1 < S < G2/M)
        area_means = df_analysis.groupby('cluster')['area'].mean().sort_values()

        # Сопоставляем фазы в порядке увеличения площади
        phases = ['SubG1', 'G1', 'S', 'G2/M']
        self.cluster_to_phase = {}

        for i, (cluster, area_mean) in enumerate(area_means.items()):
            if i < len(phases):
                phase = phases[i]
                self.cluster_to_phase[cluster] = phase
                print(f"  Кластер {cluster} -> {phase} (средняя площадь: {area_mean:.0f})")

        # Добавляем фазы к данным
        df_analysis['phase'] = df_analysis['cluster'].map(self.cluster_to_phase)

        return df_analysis, cluster_stats

    def visualize_clusters(self, X_scaled, df_analysis):
        """Визуализация результатов кластеризации"""

        print("\nСоздание визуализаций...")

        # Создаем директорию для графиков
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)

        # 1. 2D визуализация с PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                              c=df_analysis['cluster'],
                              cmap='tab10', alpha=0.6, s=10)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
        plt.title('Кластеризация клеток (PCA проекция)')
        plt.colorbar(scatter, label='Кластер')
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / 'clusters_pca.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 2. t-SNE визуализация (более медленная, но лучше разделяет)
        print("  Выполнение t-SNE... (это может занять некоторое время)")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_scaled[:5000])  # Ограничиваем для скорости

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                              c=df_analysis.iloc[:5000]['cluster'],
                              cmap='tab10', alpha=0.6, s=10)
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('Кластеризация клеток (t-SNE проекция)')
        plt.colorbar(scatter, label='Кластер')
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / 'clusters_tsne.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 3. Распределение фаз клеточного цикла
        plt.figure(figsize=(10, 6))
        phase_counts = df_analysis['phase'].value_counts()
        bars = plt.bar(phase_counts.index, phase_counts.values, color=['red', 'green', 'orange', 'blue'])
        plt.xlabel('Фаза клеточного цикла')
        plt.ylabel('Количество клеток')
        plt.title(f'Распределение клеток по фазам клеточного цикла (всего: {len(df_analysis)})')

        # Добавляем значения на столбцы
        for bar, count in zip(bars, phase_counts.values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     f'{count}\n({count / len(df_analysis) * 100:.1f}%)',
                     ha='center', va='bottom')

        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(plots_dir / 'phase_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 4. Boxplot площади по фазам
        plt.figure(figsize=(10, 6))
        phase_order = ['SubG1', 'G1', 'S', 'G2/M']
        df_plot = df_analysis[df_analysis['phase'].isin(phase_order)]

        # Используем boxplot
        box_data = [df_plot[df_plot['phase'] == phase]['area'] for phase in phase_order]

        plt.boxplot(box_data, labels=phase_order)
        plt.xlabel('Фаза клеточного цикла')
        plt.ylabel('Площадь клетки (пиксели)')
        plt.title('Распределение площади клеток по фазам клеточного цикла')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(plots_dir / 'area_by_phase.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Графики сохранены в: {plots_dir}")

    def analyze_by_conditions(self, df_analysis):
        """Анализ распределения фаз по экспериментальным условиям"""

        print("\nАнализ распределения фаз по условиям эксперимента...")

        # Создаем директорию для анализа условий
        conditions_dir = self.output_dir / 'conditions_analysis'
        conditions_dir.mkdir(exist_ok=True)

        # 1. Распределение фаз по генотипам
        plt.figure(figsize=(10, 6))

        if 'genotype' in df_analysis.columns:
            # Создаем сводную таблицу
            genotype_phase = pd.crosstab(df_analysis['genotype'], df_analysis['phase'], normalize='index')

            # Сортируем фазы в логическом порядке
            phase_order = ['SubG1', 'G1', 'S', 'G2/M']
            genotype_phase = genotype_phase[phase_order]

            ax = genotype_phase.plot(kind='bar', stacked=True, figsize=(12, 6))
            plt.title('Распределение фаз клеточного цикла по генотипам')
            plt.xlabel('Генотип')
            plt.ylabel('Доля клеток')
            plt.legend(title='Фаза', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(conditions_dir / 'phases_by_genotype.png', dpi=150, bbox_inches='tight')
            plt.close()

            # Сохраняем таблицу
            genotype_phase.to_csv(conditions_dir / 'phases_by_genotype.csv')
            print(f"  Сохранено: phases_by_genotype.csv")

        # 2. Распределение фаз по дозам облучения
        if 'dose_gy' in df_analysis.columns:
            # Преобразуем дозу в числовой формат
            df_analysis['dose_numeric'] = pd.to_numeric(df_analysis['dose_gy'], errors='coerce')

            plt.figure(figsize=(12, 6))

            # Создаем сводную таблицу
            dose_phase = pd.crosstab(df_analysis['dose_numeric'], df_analysis['phase'], normalize='index')
            dose_phase = dose_phase[phase_order]

            # Сортируем по дозе
            dose_phase = dose_phase.sort_index()

            ax = dose_phase.plot(kind='bar', stacked=True, figsize=(12, 6))
            plt.title('Распределение фаз клеточного цикла по дозам облучения')
            plt.xlabel('Доза облучения (Gy)')
            plt.ylabel('Доля клеток')
            plt.legend(title='Фаза', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(conditions_dir / 'phases_by_dose.png', dpi=150, bbox_inches='tight')
            plt.close()

            # Также строим линейные графики для каждой фазы
            plt.figure(figsize=(12, 6))

            for phase in phase_order:
                if phase in dose_phase.columns:
                    plt.plot(dose_phase.index, dose_phase[phase], marker='o', label=phase, linewidth=2)

            plt.title('Изменение распределения фаз клеточного цикла с увеличением дозы облучения')
            plt.xlabel('Доза облучения (Gy)')
            plt.ylabel('Доля клеток')
            plt.legend(title='Фаза', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(conditions_dir / 'phases_vs_dose_lines.png', dpi=150, bbox_inches='tight')
            plt.close()

            # Сохраняем таблицу
            dose_phase.to_csv(conditions_dir / 'phases_by_dose.csv')
            print(f"  Сохранено: phases_by_dose.csv")

        # 3. Распределение фаз по времени
        if 'time_h' in df_analysis.columns:
            plt.figure(figsize=(10, 6))

            # Преобразуем время в числовой формат
            df_analysis['time_numeric'] = pd.to_numeric(df_analysis['time_h'], errors='coerce')

            time_phase = pd.crosstab(df_analysis['time_numeric'], df_analysis['phase'], normalize='index')
            time_phase = time_phase[phase_order]
            time_phase = time_phase.sort_index()

            ax = time_phase.plot(kind='bar', stacked=True, figsize=(10, 6))
            plt.title('Распределение фаз клеточного цикла по времени после облучения')
            plt.xlabel('Время после облучения (часы)')
            plt.ylabel('Доля клеток')
            plt.legend(title='Фаза', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(conditions_dir / 'phases_by_time.png', dpi=150, bbox_inches='tight')
            plt.close()

            # Сохраняем таблицу
            time_phase.to_csv(conditions_dir / 'phases_by_time.csv')
            print(f"  Сохранено: phases_by_time.csv")

        # 4. Комбинированный анализ: генотип × доза × время
        if all(col in df_analysis.columns for col in ['genotype', 'dose_gy', 'time_h', 'phase']):
            # Создаем сводную таблицу
            summary_table = df_analysis.groupby(['genotype', 'time_h', 'dose_gy', 'phase']).size().unstack(fill_value=0)

            # Сохраняем полную таблицу
            summary_table.to_csv(conditions_dir / 'full_summary_table.csv')

            # Создаем сводную статистику
            summary_stats = df_analysis.groupby(['genotype', 'time_h', 'dose_gy']).agg({
                'phase': lambda x: (x == 'SubG1').mean()  # Доля SubG1 как индикатор апоптоза
            }).rename(columns={'phase': 'SubG1_fraction'})

            summary_stats.to_csv(conditions_dir / 'SubG1_summary.csv')
            print(f"  Сохранено: full_summary_table.csv, SubG1_summary.csv")

        print(f"Анализ по условиям сохранен в: {conditions_dir}")

    def save_results(self, df_analysis):
        """Сохранение результатов классификации"""

        print("\nСохранение результатов...")

        # Сохраняем полный датасет с фазами
        output_path = self.output_dir / 'cells_with_phases.csv'
        df_analysis.to_csv(output_path, index=False)
        print(f"  Полный датасет с фазами: {output_path}")

        # Сохраняем сводную статистику
        summary_path = self.output_dir / 'classification_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Сводная статистика классификации клеток по фазам клеточного цикла\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Всего клеток: {len(df_analysis)}\n\n")

            f.write("Распределение по фазам:\n")
            for phase in ['SubG1', 'G1', 'S', 'G2/M']:
                if phase in df_analysis['phase'].values:
                    count = (df_analysis['phase'] == phase).sum()
                    percentage = count / len(df_analysis) * 100
                    f.write(f"  {phase}: {count} клеток ({percentage:.1f}%)\n")

            f.write("\nСоответствие кластеров фазам:\n")
            for cluster, phase in self.cluster_to_phase.items():
                f.write(f"  Кластер {cluster} -> {phase}\n")

            if 'genotype' in df_analysis.columns:
                f.write("\nРаспределение по генотипам:\n")
                for genotype in df_analysis['genotype'].unique():
                    subset = df_analysis[df_analysis['genotype'] == genotype]
                    f.write(f"\n  {genotype}:\n")
                    for phase in ['SubG1', 'G1', 'S', 'G2/M']:
                        if phase in subset['phase'].values:
                            count = (subset['phase'] == phase).sum()
                            percentage = count / len(subset) * 100
                            f.write(f"    {phase}: {count} клеток ({percentage:.1f}%)\n")

        print(f"  Сводная статистика: {summary_path}")

        return output_path

    def run_analysis(self):
        """Запуск полного анализа"""

        print("=" * 60)
        print("КЛАССИФИКАЦИЯ КЛЕТОК ПО ФАЗАМ КЛЕТОЧНОГО ЦИКЛА")
        print("=" * 60)

        # 1. Загрузка данных
        df = self.load_data()
        if df is None:
            return

        # 2. Предобработка данных
        X_scaled, df_filtered = self.preprocess_data(df)

        # 3. Определение оптимального количества кластеров
        n_clusters = self.determine_optimal_clusters(X_scaled)

        # 4. Кластеризация
        cluster_labels = self.perform_clustering(X_scaled, n_clusters)

        # 5. Интерпретация кластеров
        df_analysis, cluster_stats = self.interpret_clusters(df_filtered, cluster_labels)

        # 6. Визуализация
        self.visualize_clusters(X_scaled, df_analysis)

        # 7. Анализ по условиям эксперимента
        self.analyze_by_conditions(df_analysis)

        # 8. Сохранение результатов
        self.save_results(df_analysis)

        print("\n" + "=" * 60)
        print("АНАЛИЗ ЗАВЕРШЕН!")
        print("=" * 60)

        # Выводим ключевые выводы
        print("\nКЛЮЧЕВЫЕ ВЫВОДЫ:")
        print("-" * 40)

        # Анализ влияния облучения на долю SubG1 (апоптотических клеток)
        if all(col in df_analysis.columns for col in ['dose_gy', 'phase']):
            print("\nВлияние облучения на долю апоптотических клеток (SubG1):")
            doses = sorted(df_analysis['dose_gy'].unique())
            for dose in doses:
                subset = df_analysis[df_analysis['dose_gy'] == dose]
                if len(subset) > 0:
                    subg1_fraction = (subset['phase'] == 'SubG1').mean() * 100
                    print(f"  Доза {dose} Gy: {subg1_fraction:.1f}% SubG1 клеток")

        # Сравнение генотипов
        if 'genotype' in df_analysis.columns:
            print("\nСравнение генотипов:")
            for genotype in df_analysis['genotype'].unique():
                subset = df_analysis[df_analysis['genotype'] == genotype]
                subg1_fraction = (subset['phase'] == 'SubG1').mean() * 100
                g2m_fraction = (subset['phase'] == 'G2/M').mean() * 100
                print(f"  {genotype}: {subg1_fraction:.1f}% SubG1, {g2m_fraction:.1f}% G2/M")

        print("\n" + "=" * 60)


def main():
    """Основная функция"""

    try:
        # Создаем и запускаем классификатор
        classifier = CellCycleClassifier()
        classifier.run_analysis()

    except Exception as e:
        print(f"Ошибка при выполнении анализа: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()