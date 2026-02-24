import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import time
import gc

# === НАСТРОЙКИ ПУТЕЙ ===
DATA_DIR = 'data/raw'
OUT_DIR = 'data/raw'

print("=== 1. Загрузка данных ===")
X_meta_train = np.load(f'{OUT_DIR}/xgb_oof_train.npy')
X_meta_test = np.load(f'{OUT_DIR}/xgb_test_preds.npy')

train_target = pd.read_parquet(f'{DATA_DIR}/train_target.parquet')
y_train = train_target.drop(columns=['customer_id']).values
target_columns = train_target.drop(columns=['customer_id']).columns.tolist()

test_main = pd.read_parquet(f'{DATA_DIR}/test_main_features.parquet')
test_customer_ids = test_main['customer_id'].values

del train_target, test_main
gc.collect()

final_test_preds = np.zeros_like(X_meta_test)

# Те же параметры мета-модели
meta_params = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'max_depth': 2, 'learning_rate': 0.05, 'n_estimators': 100,
    'tree_method': 'hist', 'device': 'cuda', 'random_state': 42
}

print(f"\n=== 2. Анализ Stack vs Base для {len(target_columns)} таргетов ===")
start = time.time()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

improved_count = 0
worsened_count = 0

for i, col in enumerate(target_columns):
    y = y_train[:, i]
    
    # 1. Считаем ЧЕСТНЫЙ базовый скор (XGBoost L1)
    base_auc = roc_auc_score(y, X_meta_train[:, i])
    
    # 2. Считаем ЧЕСТНЫЙ скор мета-модели (через 5-Fold KFold)
    # Это исправит ту "оптимистичную" оценку, о которой говорил коллега
    meta_oof_preds = np.zeros_like(y, dtype=float)
    for train_idx, val_idx in skf.split(X_meta_train, y):
        clf = xgb.XGBClassifier(**meta_params)
        clf.fit(X_meta_train[train_idx], y[train_idx])
        meta_oof_preds[val_idx] = clf.predict_proba(X_meta_train[val_idx])[:, 1]
        
    meta_auc = roc_auc_score(y, meta_oof_preds)
    
    # 3. ПРИНИМАЕМ РЕШЕНИЕ (Тот самый ХАК)
    if meta_auc > base_auc:
        # Стекинг реально улучшил результат!
        # Обучаем мета-модель на 100% данных и предсказываем Test
        clf = xgb.XGBClassifier(**meta_params)
        clf.fit(X_meta_train, y)
        final_test_preds[:, i] = clf.predict_proba(X_meta_test)[:, 1]
        improved_count += 1
        status = "✅ STACK"
    else:
        # Мета-модель сделала хуже или так же. Выбрасываем ее!
        # Просто берем сырые предсказания базовой модели для Test
        final_test_preds[:, i] = X_meta_test[:, i]
        worsened_count += 1
        status = "❌ BASE"
        
    if (i + 1) % 5 == 0 or i == 0:
        print(f"[{i+1:2d}/41] {col:<15} Base: {base_auc:.4f} | Meta: {meta_auc:.4f} -> {status}")

print(f"\nИтог хака: Стекинг победил на {improved_count} таргетах, База оставлена на {worsened_count} таргетах.")
print(f"Затрачено времени: {(time.time() - start)/60:.1f} мин")

print("\n=== 3. Сборка submission_hack.parquet ===")
submission = pd.DataFrame({'customer_id': test_customer_ids})
for i, col in enumerate(target_columns):
    submission[f'predict_{col}'] = final_test_preds[:, i]

submission.to_parquet('submission_hack.parquet', index=False)
print("ГОТОВО! Файл 'submission_hack.parquet' создан.")