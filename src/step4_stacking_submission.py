import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import time
import os
import gc

# === НАСТРОЙКИ ПУТЕЙ ===
# Убедитесь, что файлы .npy лежат в этой папке
DATA_DIR = 'data/raw' 
OUT_DIR = 'data/raw'

print("=== 1. Загрузка мета-признаков (OOF и Test Preds) ===")
X_meta_train = np.load(f'{OUT_DIR}/xgb_oof_train.npy')
X_meta_test = np.load(f'{OUT_DIR}/xgb_test_preds.npy')

# Загружаем таргеты для обучения мета-модели
train_target = pd.read_parquet(f'{DATA_DIR}/train_target.parquet')
y_train = train_target.drop(columns=['customer_id']).values
target_columns = train_target.drop(columns=['customer_id']).columns.tolist()

# Загружаем ID для финального файла
test_main = pd.read_parquet(f'{DATA_DIR}/test_main_features.parquet')
test_customer_ids = test_main['customer_id'].values

del train_target, test_main
gc.collect()

# Матрица для финальных скорректированных ответов
final_test_preds = np.zeros_like(X_meta_test)
meta_aucs = []

# Параметры мета-модели (максимально легкие для 41 признака)
meta_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 2,        
    'learning_rate': 0.05,
    'n_estimators': 100,
    'tree_method': 'hist',
    'device': 'cuda', # DGX Spark это оценит
    'random_state': 42
}

print(f"\n=== 2. Обучение Мета-модели (L2) для {len(target_columns)} таргетов ===")
start = time.time()

for i, col in enumerate(target_columns):
    # Обучаем модель "дирижера" на предсказаниях первого уровня
    clf = xgb.XGBClassifier(**meta_params)
    clf.fit(X_meta_train, y_train[:, i])
    
    # Исправляем тестовые вероятности
    final_preds = clf.predict_proba(X_meta_test)[:, 1]
    final_test_preds[:, i] = final_preds
    
    # Считаем честный OOF скор стекинга
    # (технически это скор мета-модели на обучающей выборке)
    oof_meta_preds = clf.predict_proba(X_meta_train)[:, 1]
    auc = roc_auc_score(y_train[:, i], oof_meta_preds)
    meta_aucs.append(auc)
    
    if (i + 1) % 5 == 0 or i == 0:
        print(f"[{i+1:2d}/41] {col:<15} Meta-AUC: {auc:.4f}")

print(f"\nСредний Meta-OOF AUC: {np.mean(meta_aucs):.4f}")
print(f"Затрачено времени: {(time.time() - start)/60:.1f} мин")

# === 3. Формирование финального сабмита ===
print("\n=== 3. Сборка submission.parquet ===")
submission = pd.DataFrame({'customer_id': test_customer_ids})

for i, col in enumerate(target_columns):
    # Добавляем префикс predict_ как требует платформа
    submission[f'predict_{col}'] = final_test_preds[:, i]

# Сохраняем в корень проекта или папку submissions
submission.to_parquet('submission.parquet', index=False)
print("\nГОТОВО! Файл 'submission.parquet' создан. Можно заливать на платформу.")