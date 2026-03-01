# Журнал экспериментов — Data Fusion 2026 Cybershelf

## Формат записи
```
## EXP-XXX | ДАТА | Краткое название
- Описание: что именно пробовали
- Параметры: модель, гиперпараметры, фичи
- Local CV: X.XXXX | Public LB: X.XXXX
- Вывод: что узнали, сработало ли, почему
```

---

## EXP-015 | 2026-03-01 | NN фабрика + Hill Climbing (рекорд!)
- **Описание**: "Фабрика NN" — 3 версии NN L2 с разной архитектурой/скалером → N-way blend → Hill Climbing per-target
- **Базируется на**: EXP-014 (L1 OOF + L2 XGB остаются якорем)
- **Public LB: 0.8527** (было 0.8522, **+0.0005**, новый рекорд!)
- OOF Macro AUC: **0.8493** (Hill Climbing), vs 0.8482 baseline (**+0.0012**)

### Эволюция NN L2:
| Версия | Архитектура | Scaler | Dropout | OOF | Что дало |
|--------|------------|--------|---------|-----|----------|
| v3 | 512→256→128 | StandardScaler | 0.30/0.25/0.20 | 0.8415 | baseline |
| v4 | 512→256→128 | **RankGauss** | 0.30/0.25/0.20 + InputDrop 0.10 | **0.8426** | +0.0011 |
| v5 SWA | v4 + SWA | RankGauss | 0.30/0.25/0.20 | 0.8421 | **ПРОВАЛ** |
| **v6** | **1024→512→256** | **RankGauss** | **0.40/0.35/0.30** + InputDrop 0.10 | **0.8440** | **+0.0025** |

### Бленды:
| Бленд | OOF | LB | Diff vs baseline |
|-------|-----|----|-----------------|
| 2-way XGB 60% + v3 40% (baseline) | 0.8482 | 0.8522 | — |
| 3-way XGB 55% + v3 20% + v4 25% | 0.8487 | 0.8522 | +0.0005 OOF |
| 4-way fixed XGB=0.50 v3=0.15 v4=0.05 v6=0.30 | 0.8490 | — | +0.0008 OOF |
| **Hill Climbing per-target** (XGB+v3+v4+v6) | **0.8493** | **0.8527** | **+0.0012 OOF** |

### Hill Climbing результаты:
- Доминантная модель: XGB в 38/41, v6 в 3/41
- Средние веса: xgb=0.47, v3=0.20, v4=0.08, v6=0.25
- v3 держит 20% вес несмотря на худший OOF — diversity от StandardScaler ценна!
- v4 всего 8% — слишком похожа на v6 (обе RankGauss, мало diversity)

### Ключевые инсайты (подтверждены LB):
1. **RankGauss > StandardScaler** — range [-5.20, 5.20] vs [-1.89, 283.76]
2. **Wider + Higher Dropout** — 1024→512→256 с drop 0.40 >> 512→256→128 с drop 0.30
3. **"Фабрика NN"** — разные архитектуры/скалеры дают diversity → бленд сильнее
4. **Hill Climbing per-target** — свои веса на каждый таргет дают +0.0003 vs fixed blend
5. **SWA НЕ работает** на нашей задаче (OOF 0.8421 < v4 0.8426)
6. **Diversity важнее accuracy**: v3 (слабейшая) получает 20% за счёт разного скалера

### Артефакты:
- `l2_stacking/oof_l2_nn_v4.npy`, `test_l2_nn_v4.npy` — NN v4 (RankGauss)
- `l2_stacking/oof_l2_nn_v6.npy`, `test_l2_nn_v6.npy` — NN v6 (Wider)
- `l2_stacking/hill_climb_weights.json` — per-target веса для 4-way blend
- `submission_exp015_3way_blend.parquet` — 3-way LB 0.8522
- `submission_exp015_hill_climb.parquet` — Hill Climbing **LB 0.8527** (рекорд)
- Ноутбук: `notebooks/exp015_nn_boost/exp015_nn_boost.ipynb`

---

## EXP-014 | 2026-03-01 | 3-Model L1 Stacking + NN v3 (рекорд)
- **Описание**: Полный пайплайн с 3 GBDT моделями на L1, Optuna L2 XGB + NN v3 бленд
- **Public LB: 0.8522** (было 0.8510, **+0.0012**, рекорд!)
- OOF Macro AUC: 0.8482

### Эволюция NN в рамках EXP-014:
| Версия | Epochs | Patience | OOF AUC | best_ep | LB (blend 60/40) |
|--------|--------|----------|---------|---------|-------------------|
| v1     | 30     | 10       | 0.8202  | 7-10    | 0.8502 (90/10)    |
| v2     | 30     | 10       | 0.8414  | 27-29   | **0.8515**        |
| **v3** | **60** | **15**   | **0.8415** | **44-50** | **0.8522**    |

- v1→v2: исправлена архитектура (LayerNorm, StandardScaler, lr=0.001) → +0.0212
- v2→v3: увеличены epochs 30→60, patience 10→15 → OOF +0.0001, **LB +0.0007**
- v3 модель НЕ переобучилась: best_ep=44-50, stopped=59-60 (patience не сработал в 4/5 фолдов)
- **Инсайт**: OOF почти не растёт (+0.0001), но LB растёт (+0.0007) — NN дотренировала хвосты распределения

### Архитектура от начала до конца:
```
Step 0.5: Per-target Optuna (params + FS 85% для 3 моделей)
├── XGBoost v2: 200k, 3-fold, 30 trials, mean AUC 0.8278 (31-534 фичи/таргет)
├── CatBoost:   200k, 3-fold, 30 trials, mean AUC 0.8211 (25-194 фичи/таргет)
└── LightGBM:   100k, 3-fold, 20 trials, mean AUC 0.8080 (9-631 фичи/таргет)

Step 1: L1 OOF (750k, 5-fold, 397 мин)
├── XGBoost GPU: device='cuda', 2000 rounds, early_stop=50 → OOF 0.8404
├── CatBoost GPU: task_type='GPU', 2000 iter, early_stop=50 → OOF 0.8295
├── LightGBM CPU: n_jobs=-1, 2000 rounds, early_stop=50  → OOF 0.8272
└── Артефакты: oof_*.npy (750k×41) × 3 + test_*.npy (250k×41) × 3

Step 2a: L2 XGBoost (per-target Optuna, 15 trials, 141 мин)
├── Вход: 123 OOF + 82 мета (mean/std по 3 моделям) = 205 фичей
├── Search space: depth 2-4, colsample 0.1-0.6, lr 0.01-0.3, lambda 0.1-50
├── OOF Macro AUC: 0.8457
└── Типичные params: depth=4, lr=0.03, colsample=0.15-0.35

Step 2b: L2 NN v3 (LayerNorm + StandardScaler, 45 мин)
├── Архитектура: 205→512→256→128→41 (3 блока + residual)
│   └── LayerNorm (не BN!), SiLU, Dropout 0.30/0.25/0.20
│   └── Residual: skip_proj(block1) → block2 (scaled 0.5)
├── Вход: StandardScaler(OOF) — НЕ logit! (logit range [-1.89, 283.76])
├── Training: AdamW lr=0.001, OneCycleLR, batch=512, 60 epochs, patience=15
│   └── Gradient clipping max_norm=1.0, BCEWithLogitsLoss (без label smoothing)
├── Per-fold: best_ep=50/47/44/47/45, stopped=60/60/59/60/60
├── Per-fold AUC: 0.8426/0.8400/0.8440/0.8416/0.8408
└── OOF Macro AUC: 0.8415

Final: Бленд 60% XGB L2 + 40% NN L2 v3
├── OOF Macro AUC: 0.8482
└── Public LB: 0.8522
```

### Ключевые инсайты NN (из Kaggle ресёрча, подтверждены):
- **LayerNorm вместо BatchNorm**: BN нестабилен с tabular + StandardScaler
- **StandardScaler вместо logit**: logit на 205 фичах создаёт range [-14, 283] → saturation
- **lr=0.001 вместо 0.025**: lr=0.025 переобучал за 7-10 эпох (Optuna подтвердила lr=0.0017)
- **batch=512 вместо 4096**: больше обновлений за эпоху → стабильнее learning
- **Без label smoothing на L2**: OOF уже "гладкие" вероятности
- **Residual между hidden слоями**: вместо concat input (мисматч масштабов)
- **60 epochs > 30**: модель продолжает учиться до ep 44-50 (v2 останавливалась на 27-29)
- **OOF ≈ LB при больших epochs**: v3 OOF +0.0001, но LB +0.0007 — хвосты предсказаний дотачиваются

### Диагностика L2 переобучения:
- L2 XGB (205 фичей): OOF 0.8457 → LB 0.8502 (gap 0.005)
- L2 XGB (123 OOF): OOF 0.8447 → LB 0.8496
- L2 XGB (41 XGB OOF): OOF 0.8427
- CB/LGB OOF помогают (+0.002 на OOF), мета-фичи тоже (+0.001)
- Бленд с NN v3: OOF 0.8482 → LB 0.8522 (gap 0.0040)

### Неудачные попытки в рамках EXP-014:
- NN v1 (старые params от EXP-012b): OOF 0.8202 — lr=0.025 на 205 фичах = катастрофа
- Бленд 90/10 XGB+NN v1: LB 0.8502 (хуже 0.8510)
- XGB L2 alone (без NN): LB 0.8502 (хуже 0.8510)
- XGB L2 123 OOF (без мета): LB 0.8496 (хуже)

## EXP-013 | 2026-02-27 | Pseudo Labeling + K-Fold Test Inference
- Описание: добавили Pseudo Labeling (soft labels из LB 0.8510 сабмита) к выигрышному пайплайну EXP-011
- L1 OOF: 5-fold, каждый fold train = 600k real + 250k pseudo test → OOF **0.8442** (было 0.8407, **+0.0036**)
- Топ улучшения: target_2_7 +0.026, target_2_5 +0.020, target_2_3 +0.008
- L2 Meta OOF: **0.8445** (было 0.8423, +0.0022)
- **ПРОБЛЕМА v1**: full train на 750k+250k pseudo → predict на тех же 250k → circular dependency → LB 0.8500 (ХУЖЕ 0.8510)
- **ПРОБЛЕМА v2**: OOF pseudo + old test → distribution mismatch → LB 0.8490 (ЕЩЁ ХУЖЕ)
- **РЕШЕНИЕ v3**: K-Fold Pseudo для test — разбить 250k test на 5 фолдов, каждый предсказан моделью которая НЕ видела его фичи (NVIDIA Playbook).
- **LB v3: 0.8496** — ХУЖЕ 0.8510! K-Fold не спас. Pseudo Labeling раздувает OOF но не генерализуется на LB.
- **ЗАКРЫТО.** Причина: soft labels из модели 0.8510 содержат ~15% ошибок. Модель учит шум как сигнал → OOF растёт (confirmation bias), LB падает.
- Артефакты: `xgb_oof_pseudo.npy`, `xgb_test_pseudo_kfold.npy`

## Комплексные A/B тесты | 2026-02-27
- Условия: 100k, 3-fold, 4-6 таргетов (8_1, 2_3, 9_3, 9_6 + иногда 2_5, 3_2)

### Тест 1: NaN-фичи
- NaN indicators (бинарные isnan): avg +0.0003 — шум
- NaN groups (сумма NaN по группам): avg -0.0002 — шум
- fill_value=-999: avg +0.001 — нестабильно (1/4 negative)
- **Вывод: XGBoost нативно обрабатывает NaN, дополнительные фичи не помогают**

### Тест 2: colsample_bytree
- 0.80 (default): baseline
- 0.30: avg +0.001
- **0.10: avg +0.003** ← лучший
- 0.05: avg +0.002
- **Вывод: при 2440 фичах агрессивный sampling помогает. Optuna должна искать 0.05-0.9**

### Тест 3: FS threshold (cumulative gain)
- 95% (текущий): baseline (avg 619 фичей)
- 90%: avg +0.0014
- **85%: avg +0.0024** ← лучший (avg 494 фичи)
- 80%: avg +0.0010
- **Вывод: 85% порог = меньше фичей = сильнее регуляризация = +0.0024**

### Тест 4: Sigmoid пост-обработка
- p^(1/(1+α)) — монотонная трансформация → **не меняет ROC-AUC по определению**
- **Вывод: тест некорректный, закрыто**

### Тест 5: Кластеризация / PCA-фичи
- NaN PCA(20) на всех фичах: avg +0.0017, **4/4 positive** ← лучший
- NaN_extra PCA(10): avg +0.00163, 3/4 positive
- Val_extra PCA(10): avg +0.00129, 3/4 positive
- NaN_main PCA(10): avg +0.00096, 3/4 positive
- NaN_both(20) (10 main + 10 extra): avg +0.00089, 4/4 positive
- Val_main PCA(10): avg +0.00014, 2/4 positive — слабо
- All(40): avg -0.00216 — переобучение
- **Вывод: NaN PCA(20) — лучший вариант. Extra-фичи (2241, NaN до 99.9%) несут больше NaN-сигнала**

### Тест 6: Mixup (расширенный, 6 таргетов)
- POC на 1 таргете показал +0.005, но расширенный тест: 1/6 positive, avg -0.001
- **Вывод: ложноположительный POC, закрыто**

## Тесты Pseudo Labeling / Mixup / Синтетика | 2026-02-27
- **Pseudo Labeling POC** (100k+33k): target_8_1 +0.0006, target_9_6 **+0.006** — подтверждён
- **Mixup POC** (100k, target_9_6): **+0.005** — но на 1 таргете, рискованно
- **nan_count фича** (100k): +0.000275 — шум, XGBoost и так видит NaN-паттерны
- **Мета-синтетика L2** (sum/std/max OOF): +0.000018 ~ +0.002 — = старый тест confidence
- **DAE latent features** (100k): +0.000026 — ноль
- **NN L1 на сырых фичах** (100k, union 1773 фичей): OOF 0.69 vs XGB 0.80 — тупик

## EXP-012b | 2026-02-27 | Optuna для PyTorch L2 + Skip Connection
- Описание: Optuna-тюнинг архитектуры PyTorch L2. Добавлен skip connection (конкатенация входа с hidden → classifier). Optuna нашёл: hidden=512, drop1=0.19, drop2=0.22, lr=0.02558, wd=1.16e-5
- Архитектура: Linear(41→512)→BN→SiLU→Drop(0.19)→Linear(512→256)→BN→SiLU→Drop(0.22)→Linear(256+41→41)
- Параметры: AdamW lr=0.002558, OneCycleLR max_lr=0.02558, weight_decay=1.157e-5, batch=4096, 100 эпох
- Full train на 750k (без fold CV), бленд 60% XGB L2 + 40% NN L2
- **Public LB: 0.8510** (было 0.8505, **+0.0005**, новый рекорд!)
- Вывод: skip connection + Optuna-тюнинг дали скромный, но подтверждённый буст. Потолок NN L2 близок.

## EXP-012 | 2026-02-27 | PyTorch L2 бленд с XGBoost стекингом
- Описание: обучили PyTorch Multi-Task NN как вторую L2 мета-модель на logit-трансформированных OOF (750k, 41). Архитектура: Linear(41→256)→BN→SiLU→Drop(0.3)→Linear(256→128)→BN→SiLU→Drop(0.2)→Linear(128→41). Бленд 60% XGBoost L2 + 40% PyTorch L2.
- Параметры: AdamW lr=3e-3, weight_decay=1e-4, OneCycleLR max_lr=1e-2, batch=4096, 100 эпох
- Вход: logit(OOF) = log(p/(1-p)) — нормализация распределения для NN
- OOF PyTorch L2: 0.8382 | OOF бленд L1+L2_NN: 0.8449 (+0.0043 к L1)
- **Public LB: 0.8505** (было 0.8472, **+0.0033**, новый рекорд!)
- Ключевой инсайт: NN видит гладкие линейные комбинации (корреляции target_10_1, пакеты групп 3-7), которые деревья depth=2 аппроксимируют грубо
- Артефакты: `nn_l2_oof.npy`, `submission_EXP012_nn_blend.parquet`
- Время: 40 сек обучение на A100
- Вывод: **разнообразие L2 моделей (деревья + NN) — новый прорыв.** До 0.86 осталось <0.01.

## EXP-011 | 2026-02-26 | Optuna per-target params + стекинг
- Описание: применили per-target гиперпараметры из Optuna (100k, 3-fold, 20 trials) к EXP-009 пайплайну. Каждый из 41 таргетов получил свои depth, lr, colsample, mcw, reg_alpha, reg_lambda, n_rounds.
- Параметры: индивидуальные из `optuna_best_params.json` (lr=0.01-0.02, colsample=0.3-0.9, mcw=1-20, n_rounds=500-1500)
- Фичи: per-target feature selection (cumulative gain 95%), 176-1243 фичей на таргет
- Данные: 750k, 5-fold OOF → L2 XGBoost depth=2
- L1 OOF Macro AUC: 0.8407 (было 0.8352 в EXP-009, **+0.0055**)
- L2 Meta OOF: 0.8423
- **Public LB: 0.8472** (было 0.8445, **+0.0027**, новый рекорд!)
- Топ улучшения L1: target_2_7 +0.028, target_2_3 +0.023, target_5_2 +0.019, target_3_3 +0.018
- Слабые таргеты (9_3, 9_6) улучшились мало — Optuna на 100k не перенеслась на 750k для них
- Артефакты: `xgb_oof_optuna.npy`, `xgb_test_optuna.npy`, `xgb_best_features_optuna.json`, `submission_optuna_stacking.parquet`
- Время: OOF 135 мин + full train 22 мин (A100)
- Вывод: **Optuna подтверждена на LB.** Основной прирост от средних/редких таргетов, слабые (9_3, 9_6) нуждаются в Optuna на большей выборке (200-300k).

## Тест: confidence/consensus мета-фичи | 2026-02-26
- Описание: добавили std(OOF) и mean(OOF) как 2 доп. фичи для L2 мета-модели (43 фичи вместо 41)
- Результат: **+0.0003** — пренебрежимо
- Вывод: L2 depth=2 сама выводит эту информацию из 41 фичи. Закрыто.

## Тест: LGB DART без feature selection | 2026-02-26
- Описание: LightGBM DART (200 rounds, CPU) без per-target feature selection + XGBoost OOF → стекинг на 82 мета-фичах
- OOF AUC: XGB 0.835 + LGB 0.737 → стекинг
- **Public LB: 0.8434** (было 0.8445 — стало ХУЖЕ)
- Вывод: **слабый L1 вредит стекингу.** Без feature selection LGB = 0.737, портит мета-модель. Для реального разнообразия LGB нужен СВОЙ feature selection. Закрыто в текущем виде.

## EXP-010 | 2026-02-25 | CatBoost OOF + стекинг 82 мета-фичи
- Описание: добавили CatBoost OOF (750k, 5-fold, 500 iter) как второй L1. Мета-фичи: 41 XGB + 41 CB = 82. L2 тот же XGBoost depth=2, 100 iter.
- Параметры: CatBoost depth=6, lr=0.05, GPU, Logloss, bootstrap=Bernoulli, subsample=0.8, border_count=64, early_stopping=50
- Фичи L1: те же что у XGBoost (xgb_best_features.json) — переиспользовали отбор
- Meta-AUC (train): 0.8501 (было 0.8484 с 41 фичей)
- **Public LB: 0.8445** (было 0.8444, разница +0.00002 — по сути ноль)
- Вывод: **CatBoost OOF не дал прироста.** Причина: использовали те же фичи что XGBoost → предсказания сильно коррелируют → мета-модель не получила нового сигнала. Для реального разнообразия нужен либо свой отбор фичей, либо принципиально другая модель.

## EXP-009b | 2026-02-24 | PyTorch MLP бленд со стекингом
- Описание: обучили PyTorch MLP (Embedding + BatchNorm + SiLU, 512→256→41) на 100k, 15 эпох с early stopping. Сгенерировали OOF через 5-fold. Блендили с XGBoost стекингом.
- Параметры: lr=0.003, AdamW, ReduceLROnPlateau, batch=1024, dropout=0.3/0.2
- Скрипты: step6 (тест), step7 (OOF), step8 (стекинг v2), step8_weighted (бленд)
- Public LB: **0.8432** (blend XGB+NN) / **0.8427** (weighted) — **оба хуже 0.8444**
- Вывод: NN на 100k/15 эпох слишком слабая, ухудшает бленд. Нужно 750k + больше эпох, либо отложить PyTorch до более зрелого пайплайна.

## EXP-009 | 2026-02-24 | Стекинг (мета-модель на OOF предсказаниях)
- Описание: двухуровневый пайплайн. Уровень 1: XGBoost 41 модель с 3-fold StratifiedKFold → OOF матрица (100k, 41). Уровень 2: мета-модель XGBoost (depth=2, 100 iter) предсказывает каждый таргет на основе 41 OOF-фичи
- Параметры: L1 — XGBoost 200 iter, depth=6, lr=0.05, 3 фолда. L2 — XGBoost 100 iter, depth=2, lr=0.05, 5 фолдов
- Фичи: L1 — main + extra (2440). L2 — 41 OOF предсказание
- Данные: 100k подвыборка (быстрый тест)
- Quick test (100k): OOF 0.7977 → 0.8096 (+0.0120)
- Full run (750k): OOF Macro AUC 0.8352, Meta-OOF AUC 0.8484
- **Public LB: 0.8444** (лучший результат! +0.0032 к EXP-007b)
- Per-target feature selection: 273-898 фичей на таргет (cumulative gain 95%)
- L1: XGBoost 500 iter, 5-fold OOF (Colab T4, 118 мин). L2: XGBoost depth=2, 100 iter (Spark, 0.5 мин)
- Вывод: **стекинг — главный прорыв.** Мета-модель учит корреляции между таргетами, слабые получают сигнал от частых. Следующий шаг: добавить CatBoost OOF (82 мета-фичи), хак if stack>base, увеличить iter L1.

## EXP-008 | 2026-02-24 | XGBoost + scale_pos_weight
- Описание: XGBoost 41 модель с автоматическим scale_pos_weight для каждого таргета (компенсация дисбаланса классов)
- Параметры: те же что EXP-007 + scale_pos_weight = n_neg/n_pos (от 2.2 до 9035)
- Фичи: main + extra (2440 признаков)
- Local CV: 0.8233 | Public LB: не загружали
- Вывод: **не помогло** (-0.012 к EXP-007). ROC-AUC — ранговая метрика, scale_pos_weight ломает ранжирование. Модель переобучается на редких примерах, early stopping срабатывает на 100-300 итерациях вместо 400-1500. Дисбаланс для ROC-AUC нужно решать иначе: Focal Loss, per-target Optuna, стекинг.

## EXP-007b | 2026-02-24 | XGBoost 41 модель — full train 750k (×1.2)
- Описание: переобучение EXP-007 на полных 750k. Итерации = best_iter × 1.2
- Параметры: те же что EXP-007, без early stopping, фиксированные итерации
- Фичи: main + extra (2440 признаков)
- Local CV: — | **Public LB: 0.8412** (лучший результат!)
- Время: 44 мин на T4 GPU
- Вывод: множитель ×1.2 консервативный, прирост всего +0.0003. Нужен ×2.0 с min 500 iter.

## EXP-007 | 2026-02-23 | XGBoost 41 отдельная модель (GPU, Colab T4)
- Описание: 41 отдельная бинарная модель XGBoost на GPU (T4). Обучение на 600k (80%), валидация 150k (20%)
- Параметры: XGBoost, num_boost_round=3000, max_depth=6, lr=0.05, early_stopping=100, device=cuda, min_child_weight=5, subsample=0.8, colsample_bytree=0.8
- Фичи: main + extra (2440 признаков)
- Local CV: 0.8351 | **Public LB: 0.8409** (на 600k, без полного обучения!)
- Время: 44 мин на T4 GPU
- Вывод: **лучший одиночный фреймворк.** XGBoost 41 модель > CatBoost MultiLogloss (+0.0018). Доминирует на каждом таргете vs CatBoost и LightGBM.

## EXP-006b | 2026-02-24 | CatBoost 41 модель — full train 750k (×1.3)
- Описание: переобучение EXP-006 на полных 750k. Итерации = max(best_iter × 1.3, 1500)
- Параметры: CatBoost, depth=6, lr=0.05, GPU, Logloss, фиксированные итерации
- Фичи: main + extra (2440 признаков)
- Local CV: — | **Public LB: 0.8327**
- Время: 604 мин (~10ч) на Spark GPU
- Файлы: `src/exp006_full_train.py`, `submissions/exp006_cb_41models.parquet`
- Вывод: слабее XGBoost на 0.0085. Полезен для бленда (разные ошибки).

## EXP-006 | 2026-02-23 | CatBoost 41 отдельная модель (GPU, Spark)
- Описание: 41 отдельная бинарная модель CatBoost (Logloss вместо MultiLogloss). Валидация на 600k/150k
- Параметры: CatBoost, iterations=3000, depth=6, lr=0.05, early_stopping=100, GPU, Logloss
- Фичи: main + extra (2440 признаков)
- Local CV: 0.8255 | Public LB: —
- Время: 362 мин (~6ч)
- Вывод: вторая по силе модель после XGBoost. Early stopping почти не срабатывал (~2999 итерация), модели недообучены — нужно больше итераций.

## EXP-006-LGB | 2026-02-23 | LightGBM 41 отдельная модель (Colab CPU)
- Описание: 41 бинарная модель LightGBM на Colab. pip LightGBM без GPU, работал на CPU
- Параметры: LightGBM, num_boost_round=3000, num_leaves=63, lr=0.05, early_stopping=100, min_child_samples=5
- Фичи: main + extra (2440 признаков)
- Local CV: 0.7914 | Public LB: —
- Время: ~80 мин
- Вывод: **слабый результат.** target_2_8 = 0.434 (хуже рандома). LightGBM без GPU плохо работает с категориальными фичами и редкими классами. Для бленда полезен с малым весом.

## EXP-005 | 2026-02-23 | Feature Engineering (null_count, num_mean, num_std, cat_freq)
- Описание: добавили 70 новых признаков — кол-во пропусков, статистики числовых, частоты категориальных
- Параметры: CatBoost, iterations=3000, depth=6, lr=0.05, MultiLogloss, GPU
- Фичи: main + extra + FE (2511 признаков)
- Local CV: 0.8311 | Public LB: не загружали
- Вывод: **фичи не помогли** (-0.0003 к val). CatBoost и так обрабатывает пропуски и категории нативно. Размытые агрегаты (mean/std по всем столбцам) не несут полезного сигнала. Откат к данным без FE.

## EXP-004 | 2026-02-23 | CatBoost lr=0.05, 3000 iter
- Описание: снизили lr с 0.1 до 0.05, увеличили итерации до 3000
- Параметры: CatBoost, iterations=3000, depth=6, lr=0.05, MultiLogloss, GPU, early_stopping=100
- Фичи: main + extra (2440 признаков)
- Local CV: 0.8314 | **Public LB: 0.8391**
- Вывод: +0.004 к LB. Early stopping снова не сработал (best=2999), но прирост замедлился. Потолок гиперпараметров близок, пора переходить к feature engineering / 41 отдельная модель.

## EXP-003 | 2026-02-21 | CatBoost 1000 iter + extra features
- Описание: подключили extra_features (2241 доп. числовых признака), объединили с main по customer_id
- Параметры: CatBoost, iterations=1000, depth=6, lr=0.1, MultiLogloss, GPU, early_stopping=100
- Фичи: main + extra (2440 признаков)
- Local CV: 0.8282 | **Public LB: 0.8349**
- Вывод: extra фичи дали +0.013 на LB. Early stopping снова не сработал — модель ещё растёт, нужно больше итераций.

## EXP-002 | 2026-02-21 | CatBoost 1000 iter + валидация
- Описание: увеличили итерации, добавили val split 80/20, depth=6
- Параметры: CatBoost, iterations=1000, depth=6, lr=0.1, MultiLogloss, GPU, early_stopping=100
- Фичи: только main_features (199 признаков)
- Local CV: 0.8186 | **Public LB: 0.8217**
- Вывод: +0.073 к baseline. Локальная валидация коррелирует с LB (разница ~0.003). Early stopping не сработал — модель ещё росла на 999 итерации, можно добавить ещё.

## EXP-001 | 2026-02-21 | Baseline CatBoost (оригинальный)
- Описание: baseline от организаторов, без изменений
- Параметры: CatBoost, iterations=10, depth=4, lr=0.25, MultiLogloss, GPU
- Фичи: только main_features (199 признаков)
- Валидация: нет
- Local CV: нет | **Public LB: 0.7489**
- Вывод: отправная точка, модель сильно недообучена (10 деревьев)

<!-- Новые эксперименты добавлять сверху -->
