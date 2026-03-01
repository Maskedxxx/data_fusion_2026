# Инсайты из похожих Kaggle-соревнований

## Похожие соревнования

### 1. Santander Product Recommendation (сходство 9/10)
- **Задача**: предсказание покупки 24 банковских продуктов (multi-label, cross-sell)
- **2-е место (Tom Van de Wiele)**: 24 XGBoost per-product, 10-fold CV, rank averaging
- **Ссылки**: [Write-up](https://ttvand.github.io/Second-place-in-the-Santander-product-Recommendation-Kaggle-competition/), [Kaggle Blog](https://medium.com/kaggle-blog/santander-product-recommendation-competition-2nd-place-winners-solution-write-up-3384f2a34d5b), [GitHub](https://github.com/ttvand/Santander-Product-Recommendation)

### 2. Mechanisms of Action (MoA) Prediction (сходство 8/10)
- **Задача**: multi-label classification, ~200 таргетов, обфусцированные фичи
- **1-е место ("Hungry for Gold")**: 7 моделей (3-stage NN, TabNet, ResNeSt), MultilabelStratifiedKFold
- **Ссылки**: [GitHub](https://github.com/guitarmind/kaggle_moa_winner_hungry_for_gold), [Kaggle](https://www.kaggle.com/competitions/lish-moa)

### 3. American Express Default Prediction (сходство 7/10)
- **Задача**: 5.5M строк, обфусцированные финансовые фичи, binary classification
- **1-е место (jxzly/daishu)**: 7-этапный пайплайн, LGB DART, GRU+Dense NN
- **Ссылки**: [GitHub](https://github.com/jxzly/Kaggle-American-Express-Default-Prediction-1st-solution), [DeepWiki](https://deepwiki.com/jxzly/Kaggle-American-Express-Default-Prediction-1st-solution)

### 4. Porto Seguro Safe Driver (сходство 7/10)
- **Задача**: полностью обфусцированные фичи, binary classification, страхование
- **1-е место (Michael Jahrer)**: RankGauss, Denoising Autoencoder, 1 LGB + 5 NN
- **Ссылки**: [Winner Solution](https://kaggler.com/2017/12/01/winners-solution-porto-seguro.html), [Kaggle](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction)

### 5. NVIDIA Kaggle Grandmasters Playbook (мета-гайд)
- **Ссылки**: [7 Techniques](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/), [Stacking 1st Place](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/), [Feature Engineering](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/)

---

## Техники для нашей задачи (по приоритету)

### ВЫСШИЙ приоритет (быстро, доказано)

**1. ~~confidence/consensus мета-фичи для L2~~ — ПРОВЕРЕНО, +0.0003 (бесполезно)**
- L2 depth=2 сама выводит эту информацию из 41 фичи

**2. MultilabelStratifiedKFold**
- `pip install iterative-stratification`
- Единые фолды для всех 41 таргетов с сохранением баланса
- Источник: MoA 1st Place. Критично для multi-label с дисбалансом
- Мы используем per-target StratifiedKFold — фолды разные для каждого таргета

**3. ~~Per-target Optuna для L1~~ — СДЕЛАНО (EXP-011, LB 0.8472)**
- OOF +0.0055, LB +0.0027. Основной прирост от средних/редких таргетов
- Слабые (9_3, 9_6) не улучшились — Optuna на 100k не переносится, нужна 200-300k

### ВЫСОКИЙ приоритет (часы, хороший потенциал)

**4. ~~Pseudo Labeling (soft labels)~~ — ЗАКРЫТО, LB ХУЖЕ**
- OOF +0.0036, но LB 0.8496 (хуже 0.8510). Все 3 варианта провалились.
- Причина: soft labels содержат ~15% ошибок, модель учит шум → OOF раздувается, LB падает
- Источник: NVIDIA Playbook, AmEx solutions

**5. LightGBM DART как 2-й L1** — ЧАСТИЧНО ПРОВЕРЕНО
- Без feature selection OOF 0.737, ВРЕДИТ стекингу (LB 0.8434 vs 0.8445)
- Нужен СВОЙ feature selection. DART медленный (10x GBDT), лучше GBDT на GPU
- AmEx 1st place: num_leaves=64, lr=0.035, feature_fraction=0.05, lambda_l2=30

**6. colsample_bytree=0.05-0.10 (агрессивный feature sampling)** — ПОДТВЕРЖДЁН A/B ТЕСТОМ
- При 2440 фичах каждое дерево видит ~120-240 — сильная регуляризация
- A/B тест: colsample=0.10 даёт avg +0.003 vs default 0.80
- Optuna должна искать в диапазоне 0.05-0.9 (текущий 0.3-0.9 слишком узкий)
- Источник: AmEx 1st place

### СРЕДНИЙ приоритет (сложнее, но потенциально сильный буст)

**7. Denoising Autoencoder со swap noise**
- Обучаем автоэнкодер на train+test (unsupervised, 1M строк)
- 15% значений заменяем случайными из того же столбца (swap noise)
- Латентные представления → фичи для GBDT
- Источник: Porto Seguro 1st Place
- Ссылки: [tabular_dae GitHub](https://github.com/ryancheunggit/tabular_dae), [PyTorch Tabular](https://pytorch-tabular.readthedocs.io/en/latest/tutorials/08-Self-Supervised%20Learning-DAE/)

**8. RankGauss нормализация для NN**
- rank → ErfInv → нормальное распределение. Критично для NN на обфусцированных данных
- Источник: Porto Seguro 1st Place (Michael Jahrer)

**9. 3-уровневый стек**
- L1: XGB + LGB DART + CatBoost + NN (каждый 41 OOF) → 164+ мета-фичей
- L2: GBDT + NN на мета-фичах + confidence/consensus
- L3: weighted average из L2
- Источник: NVIDIA Stacking, Kaggle Playground 1st Place

### НИЗКИЙ приоритет (если будет время)

**10. Hill climbing для весов ансамбля**
- Начать с лучшей модели, итеративно добавлять другие с оптимальными весами
- Источник: NVIDIA Playbook

**11. floor(value*100) denoising**
- Квантизация числовых фичей — уменьшает шум в обфусцированных данных
- Источник: AmEx 1st Place (S1 Denoising)

**12. Entity Embeddings → фичи для GBDT**
- Обучить NN с embedding layers → извлечь веса → подать в XGBoost как фичи
- Источник: [Entity Embeddings paper](https://arxiv.org/abs/1604.06737), Porto Seguro

---

## Ключевой вывод

Текущий лучший: **LB 0.8527** (EXP-015: "фабрика NN" + Hill Climbing per-target).

### Что сработало (EXP-015, подтверждено LB):
- ✅ NN wider 1024→512→256 + dropout 0.40 → OOF 0.8440 (+0.0025 vs v3)
- ✅ RankGauss вместо StandardScaler → +0.0011 OOF
- ✅ Hill Climbing per-target weights → +0.0003 vs fixed blend
- ✅ "Фабрика NN" (разные архитектуры/скалеры) → diversity в бленде

### Что НЕ сработало:
- ✗ SWA (Stochastic Weight Averaging) — OOF 0.8421 < v4 0.8426

### Следующие шаги:
1. TabM BatchEnsemble (k=16 голов) → ещё одна diverse NN
2. Ещё NN вариации (другой seed, другой dropout) → расширить фабрику
3. Forward Selection OOF
4. 80-100 epochs для v6
