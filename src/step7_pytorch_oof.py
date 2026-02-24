import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import time
import gc
import warnings
import os

warnings.filterwarnings("ignore")

# === 1. НАСТРОЙКИ ПУТЕЙ И ПАРАМЕТРОВ ===
DATA_DIR = 'data/raw'
OUT_DIR = 'data/raw'
BATCH_SIZE = 2048  # Увеличили батч для 750k строк и 128 ГБ памяти
EPOCHS = 20
N_FOLDS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"=== 1. Загрузка полных данных ({DEVICE}) ===")
start_time = time.time()

# Трейн
train_main = pd.read_parquet(f'{DATA_DIR}/train_main_features.parquet')
train_extra = pd.read_parquet(f'{DATA_DIR}/train_extra_features.parquet')
train_target = pd.read_parquet(f'{DATA_DIR}/train_target.parquet')
X_train = train_main.merge(train_extra, on='customer_id', how='left').drop(columns=['customer_id'])
y_train = train_target.drop(columns=['customer_id']).values
target_columns = train_target.drop(columns=['customer_id']).columns.tolist()

# Тест
test_main = pd.read_parquet(f'{DATA_DIR}/test_main_features.parquet')
test_extra = pd.read_parquet(f'{DATA_DIR}/test_extra_features.parquet')
X_test = test_main.merge(test_extra, on='customer_id', how='left').drop(columns=['customer_id'])

del train_main, train_extra, train_target, test_main, test_extra
gc.collect()

print(f"Данные загружены за {time.time() - start_time:.1f} сек. Train: {X_train.shape}, Test: {X_test.shape}")

# === 2. ОБРАБОТКА ПРИЗНАКОВ ===
print("\n=== 2. Подготовка тензоров и эмбеддингов ===")
cat_cols = [c for c in X_train.columns if c.startswith('cat_feature')]
num_cols = [c for c in X_train.columns if c not in cat_cols]

# Заполнение пропусков
X_train[num_cols] = X_train[num_cols].fillna(0).astype(np.float32)
X_test[num_cols] = X_test[num_cols].fillna(0).astype(np.float32)

X_train[cat_cols] = X_train[cat_cols].fillna(0).astype(np.int64)
X_test[cat_cols] = X_test[cat_cols].fillna(0).astype(np.int64)

# Определяем размерности эмбеддингов по всему датасету (чтобы избежать OutOfBounds)
cat_dims = []
for col in cat_cols:
    max_val = max(X_train[col].max(), X_test[col].max())
    num_unique = int(max_val) + 2 
    emb_dim = min(50, max(4, num_unique // 2)) 
    cat_dims.append((num_unique, emb_dim))

# Перевод в тензоры
X_train_cat_t = torch.tensor(X_train[cat_cols].values, dtype=torch.long)
X_train_num_t = torch.tensor(X_train[num_cols].values, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

X_test_cat_t = torch.tensor(X_test[cat_cols].values, dtype=torch.long)
X_test_num_t = torch.tensor(X_test[num_cols].values, dtype=torch.float32)

# Тестовый даталоадер
test_dataset = TensorDataset(X_test_cat_t, X_test_num_t)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

del X_train, X_test
gc.collect()

# === 3. АРХИТЕКТУРА НЕЙРОСЕТИ ===
class TabularNN(nn.Module):
    def __init__(self, cat_dims, num_features, output_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, emb_dim) for num_classes, emb_dim in cat_dims
        ])
        total_emb_dim = sum(emb_dim for _, emb_dim in cat_dims)
        self.num_bn = nn.BatchNorm1d(num_features)
        
        self.fc = nn.Sequential(
            nn.Linear(total_emb_dim + num_features, 512),
            nn.SiLU(), 
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )

    def forward(self, x_cat, x_num):
        x_cat_emb = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_cat_emb = torch.cat(x_cat_emb, dim=1)
        x_num = self.num_bn(x_num) 
        x = torch.cat([x_cat_emb, x_num], dim=1)
        return self.fc(x)

# === 4. КРОСС-ВАЛИДАЦИЯ (5-FOLD OOF) ===
print(f"\n=== 4. Старт обучения OOF (5 Folds) ===")
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros_like(y_train, dtype=np.float32)
test_preds = np.zeros((len(test_dataset), len(target_columns)), dtype=np.float32)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_num_t)):
    print(f"\n--- ФОЛД {fold + 1}/{N_FOLDS} ---")
    
    train_ds = TensorDataset(X_train_cat_t[train_idx], X_train_num_t[train_idx], y_train_t[train_idx])
    val_ds = TensorDataset(X_train_cat_t[val_idx], X_train_num_t[val_idx], y_train_t[val_idx])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = TabularNN(cat_dims, len(num_cols), output_dim=len(target_columns)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_auc = 0
    patience_counter = 0
    EARLY_STOP_PATIENCE = 4
    
    for epoch in range(EPOCHS):
        model.train()
        for batch_cat, batch_num, batch_y in train_loader:
            batch_cat, batch_num, batch_y = batch_cat.to(DEVICE), batch_num.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_cat, batch_num)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_preds_fold = []
        val_targets_fold = []
        with torch.no_grad():
            for batch_cat, batch_num, batch_y in val_loader:
                batch_cat, batch_num = batch_cat.to(DEVICE), batch_num.to(DEVICE)
                outputs = model(batch_cat, batch_num)
                val_preds_fold.append(torch.sigmoid(outputs).cpu().numpy())
                val_targets_fold.append(batch_y.numpy())
                
        val_preds_fold = np.vstack(val_preds_fold)
        val_targets_fold = np.vstack(val_targets_fold)
        
        aucs = [roc_auc_score(val_targets_fold[:, i], val_preds_fold[:, i]) for i in range(len(target_columns))]
        mean_auc = np.mean(aucs)
        
        scheduler.step(mean_auc)
        
        if mean_auc > best_auc:
            best_auc = mean_auc
            patience_counter = 0
            torch.save(model.state_dict(), f'model_fold_{fold}.pth')
            # Сохраняем лучшие OOF предсказания для текущего фолда
            oof_preds[val_idx] = val_preds_fold
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early Stopping! Эпоха {epoch+1} | Лучший Macro AUC: {best_auc:.4f}")
                break
                
    cv_scores.append(best_auc)
    
    # Инференс на тесте лучшими весами фолда
    model.load_state_dict(torch.load(f'model_fold_{fold}.pth', weights_only=True))
    model.eval()
    fold_test_preds = []
    with torch.no_grad():
        for batch_cat, batch_num in test_loader:
            batch_cat, batch_num = batch_cat.to(DEVICE), batch_num.to(DEVICE)
            outputs = model(batch_cat, batch_num)
            fold_test_preds.append(torch.sigmoid(outputs).cpu().numpy())
    
    # Усредняем предсказания (делим на 5 фолдов)
    test_preds += np.vstack(fold_test_preds) / N_FOLDS
    
    torch.cuda.empty_cache()
    gc.collect()

print(f"\n=== ИТОГИ ===")
print(f"Средний 5-Fold Macro AUC: {np.mean(cv_scores):.4f}")

print("\n=== 5. Сохранение матриц ===")
np.save(f'{OUT_DIR}/pytorch_oof_train.npy', oof_preds)
np.save(f'{OUT_DIR}/pytorch_test_preds.npy', test_preds)
print("Готово! Матрицы pytorch_oof_train.npy и pytorch_test_preds.npy успешно сохранены.")