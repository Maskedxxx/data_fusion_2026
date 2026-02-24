import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import time
import gc

DATA_DIR = 'data/raw'
OUT_DIR = 'data/raw'

print("=== 1. Загрузка OOF матриц (XGBoost + PyTorch) ===")
xgb_train = np.load(f'{OUT_DIR}/xgb_oof_train.npy')
nn_train = np.load(f'{OUT_DIR}/pytorch_oof_train.npy')

xgb_test = np.load(f'{OUT_DIR}/xgb_test_preds.npy')
nn_test = np.load(f'{OUT_DIR}/pytorch_test_preds.npy')

train_target = pd.read_parquet(f'{DATA_DIR}/train_target.parquet')
y_train = train_target.drop(columns=['customer_id']).values
target_columns = train_target.drop(columns=['customer_id']).columns.tolist()

test_main = pd.read_parquet(f'{DATA_DIR}/test_main_features.parquet')
test_customer_ids = test_main['customer_id'].values

del train_target, test_main
gc.collect()

final_test_preds = np.zeros((len(test_customer_ids), len(target_columns)))

# Возвращаем наш быстрый и надежный XGBoost на CUDA
meta_params = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'max_depth': 2, 'learning_rate': 0.05, 'n_estimators': 100,
    'tree_method': 'hist', 'device': 'cuda', 'random_state': 42
}

print(f"\n=== 2. Обучение Мета-модели (Blending) с защитой Stack vs Base ===")
start = time.time()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

improved_count = 0
worsened_count = 0

for i, col in enumerate(target_columns):
    y = y_train[:, i]
    
    # Изолируем ТОЛЬКО 2 признака для текущего таргета (Blending)
    X_blend_train = np.column_stack((xgb_train[:, i], nn_train[:, i]))
    X_blend_test = np.column_stack((xgb_test[:, i], nn_test[:, i]))
    
    base_auc = roc_auc_score(y, xgb_train[:, i])
    
    # 5-Fold кросс-валидация для мета-модели
    meta_oof_preds = np.zeros_like(y, dtype=float)
    for train_idx, val_idx in skf.split(X_blend_train, y):
        clf = xgb.XGBClassifier(**meta_params)
        clf.fit(X_blend_train[train_idx], y[train_idx])
        meta_oof_preds[val_idx] = clf.predict_proba(X_blend_train[val_idx])[:, 1]
        
    meta_auc = roc_auc_score(y, meta_oof_preds)
    
    # Умная маршрутизация
    if meta_auc > base_auc:
        clf = xgb.XGBClassifier(**meta_params)
        clf.fit(X_blend_train, y)
        final_test_preds[:, i] = clf.predict_proba(X_blend_test)[:, 1]
        improved_count += 1
        status = "✅ BLEND (XGB)"
    else:
        final_test_preds[:, i] = xgb_test[:, i]
        worsened_count += 1
        status = "❌ BASE (XGB)"
        
    if (i + 1) % 5 == 0 or i == 0:
        print(f"[{i+1:2d}/41] {col:<15} Base: {base_auc:.4f} | Meta: {meta_auc:.4f} -> {status}")

print(f"\nИтог: Блендинг победил на {improved_count} таргетах, База оставлена на {worsened_count} таргетах.")
print(f"Затрачено времени: {(time.time() - start)/60:.1f} мин")

print("\n=== 3. Сборка submission_blend_xgb_nn.parquet ===")
submission = pd.DataFrame({'customer_id': test_customer_ids})
for i, col in enumerate(target_columns):
    submission[f'predict_{col}'] = final_test_preds[:, i]

submission.to_parquet('submission_blend_xgb_nn.parquet', index=False)
print("ГОТОВО! Файл 'submission_blend_xgb_nn.parquet' создан.")