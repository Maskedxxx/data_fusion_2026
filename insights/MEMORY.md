# Data Fusion 2026 — Cybershelf (Задача 2 "Киберполка")

## Хакатон
- **Платформа**: [ods.ai](https://ods.ai/tracks/data-fusion-2026-competitions/competitions/data-fusion2026-cybershelf)
- **Организатор**: ВТБ + ODS
- **Дедлайн**: 30 марта 2026, 09:00 UTC
- **Призы**: 1 000 000 ₽ (500k / 300k / 100k) + спецноминация Companion (50k + 50k)

## Задача
Multi-label классификация: предсказать вероятность открытия **41 финансового продукта**. Метрика: **macro ROC-AUC**.

## Данные
- 1 000 000 клиентов: 750k train, 250k test
- 2440 признаков: 67 cat + 132 num (main) + 2241 num (extra)
- Обфусцировано, пропуски до 99.9%

## Текущее состояние (2026-02-27)
- **Лучший LB: 0.8510** (EXP-012b: PyTorch L2 skip connection + XGBoost стекинг)
- **EXP-012b**: бленд 60% XGB L2 + 40% PyTorch L2 (skip conn, Optuna) → LB 0.8510
- **EXP-011**: Optuna per-target → OOF 0.8407 → XGB L2 0.8423 → LB 0.8472

## Выигрышный пайплайн (EXP-012b)
1. **Optuna** (100k, 3-fold, 20 trials) → `optuna_best_params.json` (41 набора параметров)
2. **L1 OOF** (750k, 5-fold) — XGBoost per-target с Optuna params + feature selection 95% gain
3. **Full train** (750k × 1.2 iter) → test inference (250k)
4. **L2a стекинг** — XGBoost depth=2 на 41 OOF-фиче
5. **L2b PyTorch** — Multi-Task NN (41→512→256+41→41, skip connection) на logit(OOF), 100 эпох
6. **Бленд** — 60% XGB L2 + 40% PyTorch L2 → финальный сабмит

## Ключевые выводы
- **L2 разнообразие (XGB + NN бленд) = новый прорыв** (+0.0033 на LB)
- **Pseudo Labeling ЗАКРЫТ**: OOF +0.0036, но LB 0.8496 (ХУЖЕ 0.8510). K-Fold не помог. Шумные метки не генерализуются
- **Logit-трансформация OOF для NN**: log(p/(1-p)) нормализует входы, NN работает лучше
- **Multi-Task NN**: одна сеть на все 41 таргет учит скрытые связи (пакеты, антагонист)
- **Стекинг = главный прорыв** (+0.012 macro AUC, слабые таргеты +0.08)
- **Per-target Optuna** подтверждён на LB. Топ: target_2_7 +0.028, target_2_3 +0.023
- **Feature selection КРИТИЧНА**: без неё OOF 0.773 vs 0.835
- **Слабый L1 ВРЕДИТ стекингу**: LGB DART OOF 0.737 → LB упал с 0.8445 до 0.8434
- **Слабые таргеты** страдают от ПЕРЕОБУЧЕНИЯ → нужна регуляризация
- **Optuna на 100k не переносится на слабые таргеты** (9_3, 9_6) → нужна на 200-300k
- **target_10_1=1 → все остальные 40=0** (абсолютное правило, 236k клиентов)

## Подтверждённые инсайты из A/B тестов (100k, 3-fold, 4-6 таргетов)
- **FS порог 85%** вместо 95%: avg +0.0024, 3/4 positive. Меньше фичей = сильнее регуляризация
- **colsample_bytree=0.10**: avg +0.003 vs default 0.80. Optuna должна искать в диапазоне 0.05-0.9
- **NaN PCA(20)** на всех фичах: avg +0.0017, **4/4 positive**. PCA на бинарной NaN-матрице (isnan) ловит сегменты клиентов
- **NaN PCA Extra(10)**: avg +0.00163, 3/4 positive. Extra-фичи (2241) несут больше NaN-сигнала чем Main (199)
- **~~Pseudo Labeling~~**: OOF +0.0036 но LB 0.8496 (хуже 0.8510). Шумные метки раздувают OOF, не генерализуются
- **Multi-seed NN бленд**: не помогает (0.8446 vs 0.8449 single seed)

## Roadmap v6 (цель: 0.86, нужно +0.009)

### Фаза 1: Быстрые улучшения XGBoost (от EXP-012b, LB 0.8510)
Всё подтверждено A/B тестами на 100k. Нужно применить к полному пайплайну.

1. **FS порог 85%** вместо 95% — подтверждён +0.0024, 3/4 positive
   - Просто: заменить threshold в feature selection при OOF прогоне
2. **Расширенная Optuna** (200-300k, 5-fold, 30-50 trials)
   - Расширить colsample_bytree: 0.05-0.9 (было 0.3-0.9), подтверждён +0.003
   - Больше данных для Optuna → слабые таргеты (9_3, 9_6) наконец подтянутся
   - Больше trials → лучшие параметры
3. **NaN PCA фичи** — добавить к 2440 фичам для L1
   - Ждём результат комбинированного теста (какой вариант лучший)
   - Предварительно: NaN PCA(20) на всех фичах, +0.0017, 4/4 positive

**Порядок**: сначала FS 85% + NaN PCA → новый OOF → проверить на LB.
Потом Optuna v2 (долго, ~4-6 часов на Colab) → новый OOF → LB.

### Фаза 2: Разнообразие L1 (глубокая работа)
Цель: 2-3 сильных L1 модели → стекинг на 82-123 OOF фичах → путь к 0.86.

4. **CatBoost со СВОИМ FS + СВОЯ Optuna** → цель OOF > 0.80
   - Предыдущие попытки (0.64) были без FS и без Optuna
   - Нужно: feature selection на CatBoost importance → Optuna 100k → полный OOF
5. **LightGBM со СВОИМ FS + СВОЯ Optuna (GPU)** → цель OOF > 0.80
   - Предыдущая попытка (0.74) была без FS, на CPU
   - Нужно: `pip install lightgbm --force-reinstall` с CUDA, свой FS, Optuna
6. **Стекинг 3 модели** → L2 на 123 OOF фичах (41×3)
   - Только если CatBoost/LGB OOF > 0.80 (слабый L1 ВРЕДИТ стекингу)

### Фаза 3: Финиш (если время останется)
7. Pseudo Labeling с ансамблем 3 моделей (ошибка ~7% вместо 15%)
8. Per-target blend weights (нужен NN L2 OOF из fold CV)
9. MultilabelStratifiedKFold вместо per-target StratifiedKFold

### Закрытые направления
- ~~confidence/consensus мета-фичи~~ → +0.0003
- ~~LGB DART без feature selection~~ → OOF 0.737, вредит стекингу
- ~~CatBoost OOF с чужими фичами~~ → +0.00002
- ~~PyTorch MLP бленд (EXP-009b)~~ → -0.001/-0.002
- ~~FE агрегаты, scale_pos_weight~~ → не помогло
- ~~Иерархическая пост-обработка (6_4/6_5, 5_1/5_2)~~ → +0.00003
- ~~NN L1 на сырых фичах~~ → OOF 0.69 vs XGB 0.80, тупик
- ~~CatBoost per-target (Gemini)~~ → AUC 0.64 vs XGB 0.78
- ~~Mixup~~ → 1/6 positive, avg -0.001
- ~~nan_count фича~~ → +0.000275, шум
- ~~DAE latent features~~ → +0.000026
- ~~Мета-синтетика L2 (sum/std/max OOF)~~ → +0.00002
- ~~Multi-seed NN бленд~~ → 0.8446 vs 0.8449 single
- ~~Sigmoid пост-обработка~~ → монотонная трансформация не меняет AUC
- ~~NaN indicators / NaN groups~~ → шум, fill -999 нестабильно
- ~~All(40) PCA~~ → переобучение, avg -0.00216
- ~~Val_main PCA(10)~~ → avg +0.00014, слишком слабо
- ~~Pseudo Labeling (EXP-013)~~ → OOF +0.0036, но LB 0.8496 < 0.8510. K-Fold не помог
- ~~Co-occurrence L2 фичи (bundle sums/products)~~ → avg +0.00055 (D), нестабильно

## Структура репозитория
- **`insights/`** — журнал экспериментов, инсайты, документация
- **`notebooks/exp011_optuna/`** — пайплайн EXP-011 + артефакты
- **`notebooks/exp012_nn_blend/`** — EXP-012b PyTorch L2 ноутбук
- **`baseline/`** — референс от организаторов (не трогаем)

## Подробности
- **`insights/EXPERIMENTS.md`** — журнал экспериментов (EXP-001..013 + A/B тесты)
- **`insights/baseline_insights.md`** — инсайты, параметры, артефакты, feature analysis
- **`insights/kaggle_insights.md`** — техники из Kaggle-соревнований со ссылками
- **`insights/hackathon_details.md`** — описание хакатона
