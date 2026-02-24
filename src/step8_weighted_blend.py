import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import time

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

final_test_preds = np.zeros((len(test_customer_ids), len(target_columns)))

print(f"\n=== 2. Поиск идеальных весов (Weighted Average) ===")
start = time.time()

improved_count = 0
worsened_count = 0

for i, col in enumerate(target_columns):
    y = y_train[:, i]
    base_auc = roc_auc_score(y, xgb_train[:, i])
    
    best_alpha = 1.0 # 1.0 означает 100% XGBoost и 0% PyTorch
    best_auc = base_auc
    
    # Ищем идеальный вес для XGBoost от 0.50 до 1.0 с шагом 0.01
    for alpha in np.linspace(0.5, 1.0, 51):
        blend_oof = alpha * xgb_train[:, i] + (1.0 - alpha) * nn_train[:, i]
        auc = roc_auc_score(y, blend_oof)
        
        if auc > best_auc:
            best_auc = auc
            best_alpha = alpha
            
    # Применяем найденный идеальный вес к тестовой выборке
    if best_alpha < 1.0:
        final_test_preds[:, i] = best_alpha * xgb_test[:, i] + (1.0 - best_alpha) * nn_test[:, i]
        improved_count += 1
        status = f"✅ W-BLEND (XGB: {best_alpha:.2f}, NN: {1-best_alpha:.2f})"
    else:
        final_test_preds[:, i] = xgb_test[:, i]
        worsened_count += 1
        status = "❌ BASE (100% XGB)"
        
    if (i + 1) % 5 == 0 or i == 0:
        print(f"[{i+1:2d}/41] {col:<15} Base: {base_auc:.4f} | Best: {best_auc:.4f} -> {status}")

print(f"\nИтог: Взвешенный блендинг победил на {improved_count} таргетах, База оставлена на {worsened_count} таргетах.")
print(f"Затрачено времени: {(time.time() - start):.1f} сек")

print("\n=== 3. Сборка submission_weighted.parquet ===")
submission = pd.DataFrame({'customer_id': test_customer_ids})
for i, col in enumerate(target_columns):
    submission[f'predict_{col}'] = final_test_preds[:, i]

submission.to_parquet('submission_weighted.parquet', index=False)
print("ГОТОВО! Файл 'submission_weighted.parquet' создан.")