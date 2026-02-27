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
- **Лучший LB: 0.8472** (EXP-011: Optuna per-target params + стекинг)
- **EXP-011 завершён**: OOF 0.8407 → Meta OOF 0.8423 → LB 0.8472 (+0.0027 к EXP-009)
- **Артефакты EXP-011**: `notebooks/exp011_optuna/artifacts/`

## Выигрышный пайплайн (EXP-011)
1. **Optuna** (100k, 3-fold, 20 trials) → `optuna_best_params.json` (41 набора параметров)
2. **L1 OOF** (750k, 5-fold) — XGBoost per-target с Optuna params + feature selection 95% gain
3. **Full train** (750k × 1.2 iter) → test inference (250k)
4. **L2 стекинг** — XGBoost depth=2 на 41 OOF-фиче → финальный сабмит

## Ключевые выводы
- **Стекинг = главный прорыв** (+0.012 macro AUC, слабые таргеты +0.08)
- **Per-target Optuna** подтверждён на LB. Топ: target_2_7 +0.028, target_2_3 +0.023, target_5_2 +0.019
- **Feature selection КРИТИЧНА**: без неё OOF 0.773 vs 0.835
- **Слабый L1 ВРЕДИТ стекингу**: LGB DART OOF 0.737 → LB упал с 0.8445 до 0.8434
- **Слабые таргеты** страдают от ПЕРЕОБУЧЕНИЯ → нужна регуляризация
- **Optuna на 100k не переносится на слабые таргеты** (9_3, 9_6) → нужна Optuna на 200-300k
- **Пост-обработка target_10_1**: +0.0007 на L1 OOF, применять как финишный штрих
- **target_10_1=1 → все остальные 40=0** (абсолютное правило, 236k клиентов)

## Roadmap v4 (цель: 0.86, нужно +0.013)
1. **Optuna на 200-300k** — подтянуть слабые таргеты (9_3, 9_6, 10_1)
2. **Pseudo Labeling** (soft labels 250k test → train 1M)
3. **CatBoost со своим feature selection** → реальное разнообразие L1
4. **Denoising Autoencoder** → unsupervised фичи
5. **Адаптивный порог FS** (85-90% вместо 95%) → меньше фичей, сильнее регуляризация
6. **Пост-обработка target_10_1** (sigmoid alpha=0.4) → бесплатный финишный штрих

Закрытые направления:
- ~~confidence/consensus мета-фичи~~ → +0.0003
- ~~LGB DART без feature selection~~ → OOF 0.737, вредит стекингу
- ~~CatBoost OOF с чужими фичами~~ → +0.00002
- ~~PyTorch MLP бленд~~ → -0.001/-0.002
- ~~FE агрегаты, scale_pos_weight~~ → не помогло
- ~~Иерархическая пост-обработка (6_4/6_5, 5_1/5_2)~~ → +0.00003

## Структура репозитория
- **`insights/`** — журнал экспериментов, инсайты, документация
- **`notebooks/exp011_optuna/`** — выигрышный пайплайн + артефакты
- **`baseline/`** — референс от организаторов (не трогаем)

## Подробности
- **`memory/baseline_insights.md`** — инсайты, параметры, артефакты, feature analysis
- **`memory/kaggle_insights.md`** — техники из Kaggle-соревнований со ссылками
- **`memory/hackathon_details.md`** — описание хакатона
- **`insights/EXPERIMENTS.md`** — журнал экспериментов (EXP-001..011)
