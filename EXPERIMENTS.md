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
