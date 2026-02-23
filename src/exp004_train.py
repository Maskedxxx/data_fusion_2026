"""EXP-004: CatBoost lr=0.05, iterations=3000, main+extra features."""

import polars as pl
import numpy as np
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

print("=== Загрузка данных ===")
train_main = pl.read_parquet('data/raw/train_main_features.parquet')
test_main = pl.read_parquet('data/raw/test_main_features.parquet')
train_extra = pl.read_parquet('data/raw/train_extra_features.parquet')
test_extra = pl.read_parquet('data/raw/test_extra_features.parquet')
target = pl.read_parquet('data/raw/train_target.parquet')

cat_features = [col for col in train_main.columns if col.startswith('cat_feature')]
target_columns = [col for col in target.columns if col.startswith('target')]

train_main = train_main.with_columns(pl.col(cat_features).cast(pl.Int32))
test_main = test_main.with_columns(pl.col(cat_features).cast(pl.Int32))

train_full = train_main.join(train_extra, on='customer_id', how='left')
test_full = test_main.join(test_extra, on='customer_id', how='left')

feature_columns_full = [col for col in train_full.columns if col != 'customer_id']

print(f"Train: {train_full.shape}, Test: {test_full.shape}")

# === Валидация ===
print("\n=== Train/Val split ===")
train_idx, val_idx = train_test_split(np.arange(len(train_full)), test_size=0.2, random_state=42)

X_train = train_full[train_idx].select(feature_columns_full).to_pandas()
X_val = train_full[val_idx].select(feature_columns_full).to_pandas()
y_train = target[train_idx].select(target_columns).to_pandas()
y_val = target[val_idx].select(target_columns).to_pandas()

train_pool = Pool(X_train, label=y_train, cat_features=cat_features)
val_pool = Pool(X_val, label=y_val, cat_features=cat_features)

# === Обучение ===
print("\n=== Обучение модели (val) ===")
model = CatBoostClassifier(
    iterations=3000,
    depth=6,
    learning_rate=0.05,
    loss_function='MultiLogloss',
    nan_mode='Min',
    task_type='GPU',
    devices='0',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=100
)

model.fit(train_pool, eval_set=val_pool)

val_predict = model.predict(val_pool, prediction_type='RawFormulaVal')
val_score = roc_auc_score(y_val, val_predict, average='macro')
print(f"\n>>> Local Val macro ROC-AUC: {val_score:.6f}")
print(f">>> Best iteration: {model.get_best_iteration()}")

# === Полное обучение + сабмит ===
best_iter = model.get_best_iteration()
if best_iter is None:
    best_iter = 3000

print(f"\n=== Обучение полной модели ({best_iter} iter) ===")
full_pool = Pool(
    train_full.drop('customer_id').to_pandas(),
    label=target.drop('customer_id').to_pandas(),
    cat_features=cat_features
)

full_model = CatBoostClassifier(
    iterations=best_iter,
    depth=6,
    learning_rate=0.05,
    loss_function='MultiLogloss',
    nan_mode='Min',
    task_type='GPU',
    devices='0',
    random_seed=42,
    verbose=100
)

full_model.fit(full_pool)

test_pool = Pool(test_full.drop('customer_id').to_pandas(), cat_features=cat_features)
test_predict = full_model.predict(test_pool, prediction_type='RawFormulaVal')

predict_columns = [col.replace('target_', 'predict_') for col in target_columns]
submit = pl.DataFrame(test_predict, schema=predict_columns)
submit = test_full.select('customer_id').hstack(submit)
submit.write_parquet('submissions/exp004_catboost_lr005_3000iter.parquet')

print(f"\n>>> Сабмит сохранён: {submit.shape}")
print(">>> submissions/exp004_catboost_lr005_3000iter.parquet")
print("\n=== Готово! ===")
