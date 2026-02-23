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
