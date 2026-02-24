"""
EXP-006: CatBoost 41 модель — полное обучение на 750k.
best_iterations из валидационного прогона, x1.3 множитель.
Запуск: tmux → python src/exp006_full_train.py
"""

import polars as pl
import numpy as np
from catboost import Pool, CatBoostClassifier
import pyarrow as pa
import pyarrow.parquet as pq
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/exp006_full_train.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()

# best_iterations из валидационного прогона (EXP-006)
BEST_ITERS = [
    2989, 1791, 2978, 2990, 2019, 2560, 2997, 746, 2424, 2031,
    1349, 717, 939, 2989, 2999, 935, 1384, 1393, 2069, 2969,
    1094, 1977, 2207, 2763, 2009, 887, 2977, 2998, 2106, 2999,
    2997, 2995, 1427, 2396, 1867, 1152, 2356, 2999, 2999, 2932, 2999
]

def main():
    log.info('Загрузка данных...')
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
    feature_columns = [col for col in train_full.columns if col != 'customer_id']

    X_full = train_full.select(feature_columns).to_pandas()
    X_test = test_full.select(feature_columns).to_pandas()
    y_full_all = target.select(target_columns).to_pandas()

    log.info(f'X_full: {X_full.shape}, X_test: {X_test.shape}')

    test_predictions = np.zeros((len(X_test), len(target_columns)))
    start = time.time()

    for i, target_col in enumerate(target_columns):
        y_full = y_full_all[target_col].values
        n_iters = max(int(BEST_ITERS[i] * 1.3), 1500)

        train_pool = Pool(X_full, label=y_full, cat_features=cat_features)
        test_pool = Pool(X_test, cat_features=cat_features)

        model = CatBoostClassifier(
            iterations=n_iters,
            depth=6,
            learning_rate=0.05,
            loss_function='Logloss',
            nan_mode='Min',
            task_type='GPU',
            devices='0',
            random_seed=42,
            verbose=0,
        )

        model.fit(train_pool)

        pred = model.predict(test_pool, prediction_type='Probability')[:, 1]
        test_predictions[:, i] = pred

        # Сохраняем после каждой модели
        np.save('data/processed/cb_full_test_predictions.npy', test_predictions)

        elapsed = time.time() - start
        log.info(f'[{i+1:2d}/41] {target_col:15s} | iters: {n_iters:4d} | {elapsed/60:.1f} мин')

    log.info(f'Время: {(time.time()-start)/60:.1f} мин')

    # Сабмит
    predict_columns = [col.replace('target_', 'predict_') for col in target_columns]
    submit = test_full.select('customer_id').to_pandas().copy()
    for j, col in enumerate(predict_columns):
        submit[col] = test_predictions[:, j]

    table = pa.Table.from_pandas(submit, preserve_index=False)
    pq.write_table(table, 'submissions/exp006_cb_41models.parquet')
    log.info(f'Сабмит сохранён: {submit.shape}')

if __name__ == '__main__':
    main()
