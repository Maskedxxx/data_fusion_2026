import numpy as np
import pandas as pd
import xgboost as xgb
import time
import json
import os
import gc
import warnings

warnings.filterwarnings("ignore")

# === НАСТРОЙКИ ПУТЕЙ ===
# Предполагается, что скрипт запускается из корневой папки проекта DATA_FUSION
DATA_DIR = 'data/raw' # Укажите здесь вашу папку с parquet файлами, если она отличается
OUT_DIR = 'data/raw'  # Папка, куда вы сохранили xgb_best_features.json на прошлом шаге

N_ROUNDS_FINAL = 500

base_params = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'learning_rate': 0.05, 'max_depth': 6,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'tree_method': 'hist', 'device': 'cuda', 'seed': 42, 'verbosity': 0,
}

print("=== 1. Загрузка данных ===")
start = time.time()
train_main = pd.read_parquet(f'{DATA_DIR}/train_main_features.parquet')
train_extra = pd.read_parquet(f'{DATA_DIR}/train_extra_features.parquet')
train_target = pd.read_parquet(f'{DATA_DIR}/train_target.parquet')

X_train_df = train_main.merge(train_extra, on='customer_id', how='left').drop(columns=['customer_id'])
y_train_df = train_target.drop(columns=['customer_id'])
target_columns = y_train_df.columns.tolist()

test_main = pd.read_parquet(f'{DATA_DIR}/test_main_features.parquet')
test_extra = pd.read_parquet(f'{DATA_DIR}/test_extra_features.parquet')
X_test_df = test_main.merge(test_extra, on='customer_id', how='left').drop(columns=['customer_id'])

feature_columns = X_train_df.columns.tolist()

print("\n=== 2. Конвертация в Numpy (Турбо-режим) ===")
X_train_np = np.ascontiguousarray(X_train_df.astype(np.float32).values)
y_train_np = y_train_df.values
X_test_np = np.ascontiguousarray(X_test_df.astype(np.float32).values)

# Очистка RAM
del train_main, train_extra, train_target, test_main, test_extra, X_train_df, y_train_df, X_test_df
gc.collect()
print(f"Данные готовы! Время загрузки: {time.time() - start:.1f} сек.")

# === 3. ПОДГОТОВКА К ИНФЕРЕНСУ ===
test_predictions = np.zeros_like(X_test_np[:, :len(target_columns)], dtype=np.float32)

# Загружаем отобранные фичи
with open(f'{OUT_DIR}/xgb_best_features.json', 'r') as f:
    best_features_dict = json.load(f)

print(f"\n=== 4. Начинаем обучение и Inference для {len(target_columns)} таргетов ===")
start_total = time.time()

for t_idx, target_col in enumerate(target_columns):
    start_target = time.time()
    y_target = y_train_np[:, t_idx]
    n_pos = int(y_target.sum())
    
    if n_pos < 2:
        print(f"[{t_idx+1:2d}/41] {target_col:<15} ПРОПУСК (pos: {n_pos})")
        test_predictions[:, t_idx] = 0.0001
        continue

    current_min_child = 5 if n_pos > 500 else 1
    current_params = base_params.copy()
    current_params['min_child_weight'] = current_min_child

    # Восстанавливаем индексы фичей
    selected_feature_names = best_features_dict[target_col]
    selected_indices = [feature_columns.index(name) for name in selected_feature_names]

    # C-contiguous срезы
    X_train_target = X_train_np[:, selected_indices].copy(order='C')
    X_test_target = X_test_np[:, selected_indices].copy(order='C')

    # Обучение и предсказание
    dtrain = xgb.QuantileDMatrix(X_train_target, label=y_target)
    dtest = xgb.QuantileDMatrix(X_test_target, ref=dtrain)

    model = xgb.train(current_params, dtrain, num_boost_round=N_ROUNDS_FINAL, verbose_eval=False)
    test_predictions[:, t_idx] = model.predict(dtest)
    
    elapsed_target = (time.time() - start_target) / 60
    
    if (t_idx + 1) % 5 == 0 or t_idx == 0:
        print(f"[{t_idx+1:2d}/41] {target_col:<15} Готово | Фичей: {len(selected_indices):<4} | {elapsed_target:.1f} мин")

# === 5. ФИНАЛ ===
elapsed_total = (time.time() - start_total) / 60
print(f"\nInference завершен! Общее время: {elapsed_total:.1f} мин")

np.save(f'{OUT_DIR}/xgb_test_preds.npy', test_predictions)
print("Матрица тестовых предсказаний успешно сохранена!")