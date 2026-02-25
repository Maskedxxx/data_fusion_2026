import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import time
import gc
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = 'data/raw'
OUT_DIR = 'data/raw'
N_FOLDS = 5

print("=== Загрузка полных данных для CatBoost ===")
start_time = time.time()

# Трейн
train_main = pd.read_parquet(f'{DATA_DIR}/train_main_features.parquet')
train_extra = pd.read_parquet(f'{DATA_DIR}/train_extra_features.parquet')
train_target = pd.read_parquet(f'{DATA_DIR}/train_target.parquet')
X_train = train_main.merge(train_extra, on='customer_id', how='left').drop(columns=['customer_id'])
y_train = train_target.drop(columns=['customer_id']).values
target_columns = train_target.drop(columns=['customer_id']).columns.tolist()

# Тест
test_main = pd.read_parquet(f'{DATA_DIR}/test_main_features.parquet')
test_extra = pd.read_parquet(f'{DATA_DIR}/test_extra_features.parquet')
X_test = test_main.merge(test_extra, on='customer_id', how='left').drop(columns=['customer_id'])
test_customer_ids = test_main['customer_id'].values

del train_main, train_extra, train_target, test_main, test_extra
gc.collect()

print(f"Данные загружены за {time.time() - start_time:.1f} сек. Train: {X_train.shape}, Test: {X_test.shape}")

print("\n=== Подготовка категориальных признаков ===")
cat_cols = [c for c in X_train.columns if c.startswith('cat_feature')]
num_cols = [c for c in X_train.columns if c not in cat_cols]

# Заполнение пропусков (CatBoost отлично "ест" строки, переводим категории в string для надежности)
X_train[num_cols] = X_train[num_cols].fillna(0).astype(np.float32)
X_test[num_cols] = X_test[num_cols].fillna(0).astype(np.float32)

X_train[cat_cols] = X_train[cat_cols].fillna('missing').astype(str)
X_test[cat_cols] = X_test[cat_cols].fillna('missing').astype(str)

cat_features_idx = [X_train.columns.get_loc(c) for c in cat_cols]

# === OOF КРОСС-ВАЛИДАЦИЯ ===
print(f"\n=== Старт обучения CatBoost OOF ({N_FOLDS} Folds) на GPU ===")
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros_like(y_train, dtype=np.float32)
test_preds = np.zeros((len(X_test), len(target_columns)), dtype=np.float32)

cb_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'eval_metric': 'AUC',
    'task_type': 'GPU', # Задействуем мощности DGX
    'random_seed': 42,
    'verbose': 0
}

# Подготавливаем тестовый пул один раз для ускорения инференса
test_pool = Pool(X_test, cat_features=cat_features_idx)

for i, col in enumerate(target_columns):
    target_start = time.time()
    y = y_train[:, i]
    fold_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        train_pool = Pool(X_train.iloc[train_idx], y[train_idx], cat_features=cat_features_idx)
        val_pool = Pool(X_train.iloc[val_idx], y[val_idx], cat_features=cat_features_idx)
        
        model = CatBoostClassifier(**cb_params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)
        
        # OOF предсказания
        val_preds = model.predict_proba(val_pool)[:, 1]
        oof_preds[val_idx, i] = val_preds
        fold_aucs.append(roc_auc_score(y[val_idx], val_preds))
        
        # Инференс на тест
        test_preds[:, i] += model.predict_proba(test_pool)[:, 1] / N_FOLDS
        
    mean_target_auc = np.mean(fold_aucs)
    
    if (i + 1) % 5 == 0 or i == 0:
        print(f"[{i+1:2d}/41] {col:<15} CatBoost OOF AUC: {mean_target_auc:.4f} | Время: {(time.time() - target_start)/60:.1f} мин")

print("\n=== Сохранение матриц ===")
np.save(f'{OUT_DIR}/catboost_oof_train.npy', oof_preds)
np.save(f'{OUT_DIR}/catboost_test_preds.npy', test_preds)
print("Готово! Матрицы catboost_oof_train.npy и catboost_test_preds.npy успешно сохранены.")