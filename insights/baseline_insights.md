# Инсайты и прогресс экспериментов

## Лучший результат
- **Public LB: 0.8527** (EXP-015: "фабрика NN" + Hill Climbing per-target)
- **OOF Macro AUC: 0.8493** (Hill Climbing 4-way: XGB 47% + v3 20% + v4 8% + v6 25%)
- **NN v6 OOF: 0.8440** (wider 1024→512→256, RankGauss, drop 0.40)
- **L1 OOF**: XGB=0.8404, CB=0.8295, LGB=0.8272 (3-model, 750k 5-fold)

## Пайплайн (общий)
1. Загрузка main (199) + extra (2241) → join по customer_id = 2440 признаков
2. cat_features (67) каст Int32, extra — числовые
3. Train/val split: test_size=0.2, random_state=42 (600k/150k)
4. Обучение → замер macro ROC-AUC → полное обучение 750k → сабмит

## Данные
- 2440 признаков: 67 cat + 132 num (main) + 2241 num (extra)
- Категориальные — целые числа, числовые — нормализованы, всё обфусцировано
- 132/199 main имеют пропуски (0.4%–99.9%)
- Дисбаланс: от 83 (target_2_8, 0.01%) до 236k (target_10_1, 31.5%)
- Медиана продуктов на клиента = 1, max = 11
- target_10_1 — антагонист (отрицательная корреляция с остальными)
- Группы 3-7 — внутригрупповая корреляция (пакетные покупки)
- Слабые таргеты (AUC < 0.75): target_9_3, target_9_6, target_3_1, target_6_1, target_5_2, target_6_2
- Слабые ≠ редкие (target_9_6 = 22.3% — частый, но плохо предсказуемый)

## Результаты экспериментов
| EXP | Описание | Local Val | Public LB |
|-----|----------|-----------|-----------|
| 001 | Baseline 10 iter | — | 0.7489 |
| 002 | 1000 iter, depth=6, val split | 0.8186 | 0.8217 |
| 003 | + extra features | 0.8282 | 0.8349 |
| 004 | lr=0.05, 3000 iter | 0.8314 | 0.8391 |
| 005 | Feature engineering | 0.8311 | — (не помогло) |
| 006 | 41 CatBoost (val) | 0.8255 | — |
| 006b | CatBoost full 750k ×1.3 | — | 0.8327 |
| 006-LGB | 41 LightGBM | 0.7914 | — |
| 007 | **41 XGBoost** | **0.8351** | **0.8409** |
| 007b | XGBoost full 750k ×1.2 | — | **0.8412** |
| 008 | XGBoost + scale_pos_weight | 0.8233 | — (не помогло) |
| 009 | **Стекинг (OOF + мета-модель)** | 0.8352 (OOF 750k) | **0.8444** |
| 009-NN-blend | Стекинг XGB + PyTorch MLP бленд | — | 0.8432 (хуже) |
| 009-weighted | Взвешенный бленд XGB + NN | — | 0.8427 (хуже) |
| 010 | CatBoost OOF + стекинг 82 фичи | Meta 0.8501 | **0.8445** (+0.00002, ≈0) |
| — | confidence/consensus мета-фичи | — | +0.0003 (бесполезно) |
| — | LGB DART без FS + стекинг | OOF 0.737 | 0.8434 (хуже!) |
| 011 | **Optuna params + стекинг** | OOF 0.8407, Meta 0.8423 | **0.8472** (+0.0027) |
| 012 | **PyTorch L2 бленд (60/40)** | NN OOF 0.8382, бленд 0.8449 | **0.8505** (+0.0033) |
| 012b | **PyTorch L2 + skip connection (Optuna)** | бленд 60/40 | **0.8510** (+0.0005) |
| 013 | Pseudo Labeling + K-Fold test | OOF 0.8442, Meta 0.8447 | **0.8496** (хуже!) |
| — | Pseudo v1 (circular dep) | — | 0.8500 (хуже!) |
| — | Pseudo v2 (mismatch) | — | 0.8490 (хуже!) |
| 014 | **3-model L1 + L2 XGB + NN v2** | OOF 0.8479 | **0.8515** (+0.0005) |
| 014v3 | **L2 NN v3 (60 ep) + blend 60/40** | OOF 0.8482 | **0.8522** (+0.0007) |
| **015** | **NN фабрика (v3+v4+v6) + Hill Climbing** | OOF 0.8493 | **0.8527** (+0.0005, рекорд!) |

## Что сработало
- **~~Pseudo Labeling~~**: OOF +0.0036, но LB 0.8496 (хуже 0.8510). Закрыто — шумные метки не генерализуются.
- **PyTorch L2 бленд с XGB L2**: +0.0033 LB (0.8472→0.8505). NN на logit(OOF), Multi-Task, бленд 60/40
- **Skip connection + Optuna для NN L2**: +0.0005 LB (0.8505→0.8510). Архитектура 41→512→256+41→41
- **Logit-трансформация OOF для NN**: log(p/(1-p)) нормализует вход, NN видит линейные комбинации которые деревья пропускают
- **Стекинг OOF**: +0.012 macro AUC. Слабые таргеты взлетели: target_5_2 +0.081, target_2_5 +0.071
- **Per-target Optuna**: +0.0027 LB (EXP-011)
- **41 отдельная модель > MultiLogloss**: +0.002 LB
- **XGBoost доминирует**: на каждом таргете лучше CatBoost и LightGBM
- **Extra features**: +0.013 LB (EXP-003)

## Подтверждённые инсайты из A/B тестов (100k, 3-fold, 4-6 таргетов)
- **FS порог 85%** вместо 95%: avg +0.0024, 3/4 positive. Меньше фичей = сильнее регуляризация
- **colsample_bytree=0.10**: avg +0.003 vs default 0.80. Optuna должна искать 0.05-0.9
- **NaN PCA(20)** на всех фичах: avg +0.0017, 4/4 positive. PCA на бинарной NaN-матрице ловит сегменты клиентов
- **NaN PCA Extra(10)**: avg +0.00163, 3/4 positive. Extra-фичи несут больше NaN-сигнала чем Main

## Что НЕ сработало
- **FE (EXP-005)**: null_count, mean, std, cat_freq — CatBoost справляется нативно
- **scale_pos_weight (EXP-008)**: ROC-AUC ранговая метрика, spw ломает ранжирование (-0.012)
- **Full train ×1.2**: слишком консервативный множитель (+0.0003)
- **PyTorch MLP бленд**: NN на 100k/15 эпох слишком слабая, портит бленд (0.8432/0.8427 vs 0.8444)
- **CatBoost OOF с фичами XGBoost (EXP-010)**: предсказания слишком коррелированы → +0.00002 на LB
- **confidence/consensus мета-фичи**: std(OOF) и mean(OOF) как L2 фичи = +0.0003
- **LGB DART без feature selection**: OOF 0.737, стекинг упал 0.8445→0.8434
- **CatBoost per-target (Gemini)**: AUC 0.64 vs XGB 0.78
- **NN L1 на сырых фичах**: OOF 0.69 vs XGB 0.80 (NaN handling)
- **Mixup**: POC +0.005 на 1 таргете, расширенный тест 1/6 positive, avg -0.001
- **nan_count фича**: +0.000275 — шум
- **DAE latent features**: +0.000026 — ноль
- **Мета-синтетика L2**: sum/std/max OOF = +0.00002
- **Multi-seed NN бленд**: 0.8446 vs 0.8449 single — хуже
- **Sigmoid пост-обработка**: монотонная трансформация не меняет AUC
- **NaN indicators/groups**: шум, fill -999 нестабильно
- **All(40) PCA**: переобучение, avg -0.00216
- **Чужой пайплайн без feature selection**: OOF 0.773 → LB 0.779

## Optuna Per-Target (завершена 2026-02-26)

### Настройки
- 100k подвыборка, 3-fold StratifiedKFold, 20 trials на таргет
- Пространство: max_depth [3-8], lr [0.01-0.1], colsample [0.3-0.9], mcw [1-20], reg_alpha/lambda [1e-8..10], n_rounds [200-2000]
- Артефакт: `optuna_best_params.json` (Google Drive DATA/)

### Паттерны оптимальных параметров
- lr=0.01-0.02 (vs default 0.05) — медленнее, но точнее
- colsample=0.30-0.89 — варьируется, некоторым таргетам нужна агрессивная регуляризация
- mcw=1-20 — слабые таргеты хотят mcw=10-20 (регуляризация)
- n_rounds: 500-1500 в основном

### Топ улучшений
- target_2_8: **+0.4464** (ультра-редкий, 83 позитива — Optuna спасла от переобучения)
- target_2_3: +0.0853, target_5_2: +0.0853
- target_9_3: +0.0475, target_3_1: +0.0144
- Даже сильные улучшились: target_8_1: +0.0012

### Ключевой вывод
Слабые таргеты страдают от ПЕРЕОБУЧЕНИЯ, не недообучения. Им нужна больше регуляризации (ниже lr, агрессивнее colsample, выше mcw), а не больше итераций.

## Технические инсайты
- CatBoost early stopping не срабатывает: best_iter ~2999, модели недообучены на 3000
- LightGBM pip без GPU, падает на редких классах (min_child_samples=5 фиксит)
- Colab отключается: np.save после каждой модели обязателен
- Parquet сабмит: pyarrow.parquet.write_table() вместо pandas.to_parquet()
- **XGBoost GPU утилизация**: на 100k данных GPU загружен 10-30% — это нормально, overhead CPU↔GPU. На 750k загрузка значительно выше
- **DMatrix vs QuantileDMatrix**: при tree_method='hist' DMatrix каждый раз конвертируется в квантильный формат. QuantileDMatrix делает это один раз → быстрее на повторных обучениях. В Optuna-скриптах DMatrix ок (быстрые пробы), в полном пайплайне использовать QuantileDMatrix
- **Колонки сабмита**: платформа ожидает `predict_1_1` (НЕ `predict_target_1_1`). Ошибка повторяется каждый раз — ВСЕГДА делать `.replace('target_', '')` при формировании. Dtype float64, customer_id int32 (НЕ int64)
- **LightGBM CUDA**: pip install по умолчанию БЕЗ CUDA. Для GPU: `pip install lightgbm --no-binary lightgbm --config-settings=cmake.define.USE_CUDA=ON --force-reinstall` (нужен `--force-reinstall`, иначе pip говорит "already satisfied")
- **LGB DART медленный**: DART boosting пересчитывает dropout на всех предыдущих деревьях → 10x медленнее GBDT. На GPU ещё хуже. Использовать CPU с 200 rounds или GBDT на GPU
- **Слабый L1 вредит стекингу**: LGB DART OOF 0.737 без feature selection испортил стекинг (0.8445→0.8434). L1 модель ДОЛЖНА быть сильной, лучше одна хорошая чем две посредственные

## Параметры моделей
### XGBoost (лучший)
- num_boost_round=3000, max_depth=6, lr=0.05, early_stopping=100
- min_child_weight=5, subsample=0.8, colsample_bytree=0.8
- tree_method=hist, device=cuda
- best_iters (val): [419, 188, 675, 576, 223, 247, 534, 63, 147, 54, 92, 49, 389, 889, 1168, 46, 87, 544, 219, 131, 180, 170, 241, 187, 94, 151, 911, 876, 405, 1546, 989, 384, 135, 554, 263, 96, 238, 891, 799, 290, 1077]

### CatBoost
- iterations=3000, depth=6, lr=0.05, early_stopping=100, GPU, Logloss
- best_iters (val): [2989, 1791, 2978, 2990, 2019, 2560, 2997, 746, 2424, 2031, 1349, 717, 939, 2989, 2999, 935, 1384, 1393, 2069, 2969, 1094, 1977, 2207, 2763, 2009, 887, 2977, 2998, 2106, 2999, 2997, 2995, 1427, 2396, 1867, 1152, 2356, 2999, 2999, 2932, 2999]

### LightGBM
- num_boost_round=3000, num_leaves=63, lr=0.05, min_child_samples=5, CPU only

## Feature Analysis (notebooks/feature_analysis.ipynb)

### Ячейка 1-2: Null-анализ + importance на target_8_1
- 571 фича с >90% пропусков, 296 с >99%
- **46% фичей (1122 из 2440) не используются** моделью (gain=0) на target_8_1
- Фичи с 70-90% пропусков — **самые ценные** (81% используются). Факт наличия значения = сигнал
- **num_feature_22** — абсолютный лидер (gain 1177, в 5× больше второго), main, 0% пропусков
- Топ-20: 9 main (0% nulls) + 11 extra (25-60% nulls)

### Ячейка 3: Importance на 4 таргетах — пересечения
- target_8_1: 1171 фичей | target_9_6: 1424 | target_3_1: 1381 | target_1_1: 1031
- **Общее ядро (все 4)**: 831 фичей — «скелет» для всех моделей
- **Используются хотя бы одной**: 1688 (69%)
- **Никем не используются**: 752 (31%)
- Слабый таргет (target_9_6) использует БОЛЬШЕ всех фичей — хватается за любой сигнал
- Редкий таргет (target_1_1, 1%) использует меньше всех (1031)

### Ячейка 4: A/B тест глобального удаления 752 фичей
- target_8_1: +0.0003, target_9_6: -0.0019, target_3_1: -0.0000, **target_1_1: -0.0064**
- **Вывод: глобальное удаление опасно** — редкие таргеты теряют сигнал
- Нельзя делать global feature selection на основе части таргетов

### Ячейка 5: Per-target feature selection через Cumulative Gain 95%
- Для каждого таргета: обучаем XGBoost → сортируем фичи по gain → оставляем топ-95% cumulative gain
- **Результаты (100k, 500 iter, 8 таргетов):**
  - target_2_2: +0.0024 (979 фичей) | target_8_1: +0.0002 (970)
  - target_7_2: **+0.0027** (1003) | target_9_6: **+0.0036** (1282)
  - target_3_1: +0.0001 (1239) | target_1_1: -0.0015 (812)
  - target_2_8: -0.0000 (84) | **target_6_5: -0.0227** (334)
- **Работает на слабых/средних таргетах**, опасно для ультра-редких (40 позитивов)
- Жёсткий порог 95% не подходит всем → нужен адаптивный (через Optuna)
- **Оценка прироста к общему скору: +0.001-0.002** (дополнение, не прорыв)

## Стекинг (EXP-009)

### Быстрый тест (100k, notebooks/exp009_stacking_test.ipynb)
- 100k, 200 iter, 3-fold OOF, мета-модель XGBoost depth=2
- **Базовый OOF: 0.7977 → Стекинг OOF: 0.8096 (+0.0120)**
- Слабые таргеты взлетели: target_5_2 +0.081, target_2_5 +0.071, target_1_2 +0.038
- Сильные почти не пострадали: target_8_1 -0.0008, target_3_2 -0.0008
- Хак: для сильных таргетов (stack<=base) берём базовое предсказание

### Полный OOF прогон (750k, Colab T4) — ЗАВЕРШЁН
- 750k, 500 iter, 5-fold OOF, per-target feature selection (cumulative gain 95%)
- **OOF Macro AUC: 0.8352** (совпадает с EXP-007 val!)
- Время: 118.6 мин на Colab T4
- Фичей на таргет: 273-898 (адаптивно), target_2_8 самый узкий (273)
- Слабые: target_9_3 0.6904, target_9_6 0.6930, target_3_1 0.7003
- Сильные: target_2_8 0.9955, target_8_1 0.9826, target_3_5 0.9768
- Артефакты (Colab Drive): `xgb_oof_750k.npy` (750k, 41), `xgb_best_features.json`
- Артефакты скопированы на Spark: `data/raw/xgb_best_features.json`

### EXP-011: Optuna + стекинг (LB 0.8472)
- Ноутбук: `notebooks/exp011_optuna/exp011_optuna_stacking.ipynb`
- Артефакты: `notebooks/exp011_optuna/artifacts/`
- OOF: `xgb_oof_optuna.npy` (750k, 41), фичей на таргет: 176-1243
- Test: `xgb_test_optuna.npy` (250k, 41)
- Params: `optuna_best_params.json`, Features: `xgb_best_features_optuna.json`
- Сабмит: `submission_optuna_stacking.parquet`

### EXP-012b: PyTorch L2 + skip connection (LB 0.8510, текущий лучший)
- Ноутбук: `notebooks/exp012_nn_blend/exp012_nn_blend.ipynb`
- Архитектура: Linear(41→512)→BN→SiLU→Drop(0.19)→Linear(512→256)→BN→SiLU→Drop(0.22)→[concat input]→Linear(297→41)
- Бленд: 60% XGB L2 + 40% NN L2

### EXP-013: Pseudo Labeling + K-Fold test (в процессе)
- Soft labels из LB 0.8510 сабмита для 250k test
- L1 OOF: 0.8442 (+0.0036), L2 Meta OOF: 0.8445
- K-Fold: test разбит на 5 фолдов, каждый предсказан моделью без его фичей

### Мета-модель учит связи между таргетами
- Корреляции групп 3-7, антагонист target_10_1
- **Решает проблему дисбаланса**: редкие таргеты получают сигнал от частых связанных

## Roadmap v6 (цель: 0.86, нужно +0.009)

### В процессе
- **EXP-013 K-Fold Pseudo test** — запущен на Colab, ~33/41 таргетов готово

### Следующие эксперименты (по приоритету)
1. **Расширенная Optuna** (200-300k, 5-fold, 30-50 trials, colsample 0.05-0.9)
2. **FS порог 85%** вместо 95% — подтверждённый +0.0024
3. **NaN PCA(20) фичи** — подтверждённый +0.0017 (4/4 positive)
4. **Per-target blend weights** — нужен NN L2 OOF из fold CV
5. **Пост-обработка target_10_1** — проверить что AUC реально меняется

### Идеи для исследования
- Denoising Autoencoder → unsupervised фичи (Porto Seguro 1st place)
- CatBoost/LGB со СВОИМ feature selection → разнообразие L1
- MultilabelStratifiedKFold вместо per-target StratifiedKFold

## Пост-обработка target_10_1 (проверено на OOF)
- target_10_1=1 → ВСЕ остальные 40 = 0 (236k клиентов, абсолютное правило)
- Sigmoid: `factor = 1 - alpha * expit(5 * (pred_10_1 - median))`, other *= factor
- Лучший alpha=0.40: **+0.00069 на L1 OOF** (до стекинга)
- Иерархия (6_4/6_5, 5_1/5_2): +0.00003 — бесполезно, L2 уже знает
- Применять как финишный штрих ПОВЕРХ финального сабмита (после L2 эффект будет ~+0.0003)

## Дополнительные идеи (если будет время)
- Denoising Autoencoder → unsupervised фичи (Porto Seguro 1st place)
- CatBoost/LGB со СВОИМ feature selection → разнообразие L1
- MultilabelStratifiedKFold вместо per-target StratifiedKFold
- Взаимодействия топ-фичей: num_feature_76 × num_feature_117
- Anomaly score (Isolation Forest) как фича для редких таргетов

## Артефакты

### Spark (data/raw/) — ключевые артефакты EXP-009
- `xgb_oof_train.npy` — XGBoost OOF (750k, 41) — мета-фичи для стекинга
- `xgb_test_preds.npy` — XGBoost test (250k, 41) — базовые предсказания L1
- `xgb_best_features.json` — per-target отобранные фичи (273-898 на таргет)

### Spark (data/processed/) — старые артефакты
- `cb_val_predictions.npy` — CatBoost val (600k, 41 модель)
- `cb_full_test_predictions.npy` — CatBoost test (750k full)

### Google Drive (data_fusion/) — Colab артефакты
- `xgb_oof_train.npy`, `xgb_best_features.json` — копии с Spark
- `xgb_val_predictions.npy`, `xgb_test_predictions.npy` — XGBoost val/test (600k)
- `lgb_val_predictions.npy`, `lgb_test_predictions.npy` — LightGBM (слабый)
- `DATA/optuna_best_params.json` — Optuna параметры для 41 таргета (100k, 3-fold, 20 trials)

### Скрипты и ноутбуки
- **`notebooks/exp012_nn_blend/exp012_nn_blend.ipynb`** — ЛУЧШИЙ пайплайн (LB 0.8510)
- **`notebooks/exp011_optuna/exp011_optuna_stacking.ipynb`** — Optuna + стекинг (LB 0.8472)
- **`notebooks/exp009_stacking.ipynb`** — стекинг (LB 0.8444)
- `notebooks/feature_analysis.ipynb` — анализ фичей: nulls, importance, per-target selection
- `notebooks/exp006_41models.ipynb` — 41 CatBoost (Spark)
- `notebooks/exp007_xgboost_41models.ipynb` — 41 XGBoost
- `notebooks/main_pipeline.ipynb` — EXP-002..004
- `baseline/baseline_catboost.ipynb` — референс (не трогаем)

### Документация (insights/)
- `insights/EXPERIMENTS.md` — журнал экспериментов EXP-001..011
- `insights/PIPELINE_EXP009.txt` — ASCII-схема пайплайна
- `insights/MEMORY.md`, `baseline_insights.md`, `kaggle_insights.md`, `hackathon_details.md`
