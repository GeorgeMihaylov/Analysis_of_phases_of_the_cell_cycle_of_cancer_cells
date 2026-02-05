# scripts/04cell_cycle_classification.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  # pip install imbalanced-learn
import xgboost as xgb
import joblib

# === КОНФИГУРАЦИЯ БИОЛОГИЧЕСКОГО PEФЕРЕНСА ===
# Данные из вашего запроса
REF_DATA = {
    ('WT', 0): {'SubG1': 6.9, 'G1': 61.4, 'G2M': 20.4},  # Остаток ~11.3% это S-фаза
    ('WT', 4): {'SubG1': 22.9, 'G1': 26.4, 'G2M': 45.9},
    ('WT', 10): {'SubG1': 36.3, 'G1': 20.1, 'G2M': 36.1},
    ('CDK8KO', 0): {'SubG1': 9.0, 'G1': 63.4, 'G2M': 16.4},
    ('CDK8KO', 4): {'SubG1': 35.4, 'G1': 21.8, 'G2M': 33.2},
    ('CDK8KO', 10): {'SubG1': 48.1, 'G1': 23.2, 'G2M': 19.4},
}


def train_and_predict(df_features):
    # 1. Подготовка данных
    # Выделяем 'Genotype' и 'Dose' из имени файла (если еще не сделано)
    # df_features['Dose'] = ... (парсинг из имени)

    # 2. Создание "Silver Standard" (Псевдо-разметка для обучения)
    # Мы используем распределение интенсивности ДНК (DAPI) для грубой разметки,
    # чтобы обучить классификатор находить более тонкие морфологические паттерны.

    df_train_list = []

    for (geno, dose), data in df_features.groupby(['Genotype', 'Dose']):
        # Логарифмируем интенсивность для нормализации гистограммы
        log_int = np.log1p(data['mean_intensity'])

        # Определяем пороги на основе квантилей, соответствующих вашим данным цитометрии
        # Пример для WT 0Gy: топ 20.4% самых ярких - это G2M, нижние 6.9% тусклых/мелких - SubG1
        ref = REF_DATA.get((geno, dose))
        if not ref: continue

        q_subg1 = np.percentile(log_int, ref['SubG1'])
        # G2M - это самые яркие клетки. Порог отсекает (100 - G2M)%
        q_g2m = np.percentile(log_int, 100 - ref['G2M'])

        # Разметка
        subset = data.copy()
        subset['phase_label'] = 'S'  # Default middle
        subset.loc[log_int <= q_subg1, 'phase_label'] = 'SubG1'
        subset.loc[log_int >= q_g2m, 'phase_label'] = 'G2M'

        # G1 - это основной пик между SubG1 и S.
        # Упрощенно: берем диапазон от SubG1 до медианы оставшегося
        mask_mid = (subset['phase_label'] == 'S')
        # ... тут можно уточнить логику выделения G1 vs S, но для старта хватит G1/S/G2
        subset.loc[
            (log_int > q_subg1) & (log_int < q_g2m), 'phase_label'] = 'G1'  # Пока объединим G1/S или разделим позже

        df_train_list.append(subset)

    df_labeled = pd.concat(df_train_list)

    # 3. Обучение модели
    features = ['area', 'eccentricity', 'solidity', 'mean_intensity',
                'circularity', 'lbp_entropy', 'haralick_contrast']

    X = df_labeled[features].fillna(0)
    y = df_labeled['phase_label']

    # Балансировка классов (SubG1 часто мало в контроле)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Random Forest с весами
    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    clf.fit(X_res, y_res)

    # Предсказание на всех данных
    df_features['pred_phase'] = clf.predict(df_features[features].fillna(0))
    df_features['prob_subg1'] = clf.predict_proba(df_features[features].fillna(0))[:, 0]  # Проверьте индекс класса!

    return df_features, clf
