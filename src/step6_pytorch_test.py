import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import time
import gc
import warnings
warnings.filterwarnings("ignore")

# === 1. НАСТРОЙКИ ===
DATA_DIR = 'data/raw'
SAMPLE_SIZE = 100000 
BATCH_SIZE = 1024
EPOCHS = 15 # Увеличили запас эпох
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=== 1. Загрузка тестовой подвыборки ===")
start = time.time()
train_main = pd.read_parquet(f'{DATA_DIR}/train_main_features.parquet').head(SAMPLE_SIZE)
train_extra = pd.read_parquet(f'{DATA_DIR}/train_extra_features.parquet').head(SAMPLE_SIZE)
train_target = pd.read_parquet(f'{DATA_DIR}/train_target.parquet').head(SAMPLE_SIZE)

X_df = train_main.merge(train_extra, on='customer_id', how='left').drop(columns=['customer_id'])
y_df = train_target.drop(columns=['customer_id'])
target_columns = y_df.columns.tolist()

# === 2. ОБРАБОТКА ПРИЗНАКОВ ===
cat_cols = [c for c in X_df.columns if c.startswith('cat_feature')]
num_cols = [c for c in X_df.columns if c not in cat_cols]

X_df[num_cols] = X_df[num_cols].fillna(0).astype(np.float32)
X_df[cat_cols] = X_df[cat_cols].fillna(0).astype(np.int64) 

cat_dims = []
for col in cat_cols:
    num_unique = int(X_df[col].max()) + 2 
    emb_dim = min(50, max(4, num_unique // 2)) 
    cat_dims.append((num_unique, emb_dim))

X_cat_tensor = torch.tensor(X_df[cat_cols].values, dtype=torch.long)
X_num_tensor = torch.tensor(X_df[num_cols].values, dtype=torch.float32)
y_tensor = torch.tensor(y_df.values, dtype=torch.float32)

X_cat_tr, X_cat_val, X_num_tr, X_num_val, y_tr, y_val = train_test_split(
    X_cat_tensor, X_num_tensor, y_tensor, test_size=0.2, random_state=42
)

# Оптимизация DataLoader
train_dataset = TensorDataset(X_cat_tr, X_num_tr, y_tr)
val_dataset = TensorDataset(X_cat_val, X_num_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# === 3. АРХИТЕКТУРА МОДЕЛИ ===
class TabularNN(nn.Module):
    def __init__(self, cat_dims, num_features, output_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, emb_dim) for num_classes, emb_dim in cat_dims
        ])
        total_emb_dim = sum(emb_dim for _, emb_dim in cat_dims)
        
        # Динамическая нормализация числовых фичей
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

# === 4. ОБУЧЕНИЕ + EARLY STOPPING ===
model = TabularNN(cat_dims, len(num_cols), output_dim=len(target_columns)).to(DEVICE)
criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

best_auc = 0
patience_counter = 0
EARLY_STOP_PATIENCE = 4

print(f"\n=== 4. Старт обучения на {DEVICE} ===")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for batch_cat, batch_num, batch_y in train_loader:
        batch_cat, batch_num, batch_y = batch_cat.to(DEVICE), batch_num.to(DEVICE), batch_y.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(batch_cat, batch_num)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for batch_cat, batch_num, batch_y in val_loader:
            batch_cat, batch_num = batch_cat.to(DEVICE), batch_num.to(DEVICE)
            outputs = model(batch_cat, batch_num)
            val_preds.append(torch.sigmoid(outputs).cpu().numpy())
            val_targets.append(batch_y.numpy())
            
    val_preds = np.vstack(val_preds)
    val_targets = np.vstack(val_targets)
    
    aucs = [roc_auc_score(val_targets[:, i], val_preds[:, i]) for i in range(len(target_columns))]
    mean_auc = np.mean(aucs)
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}/{EPOCHS}] | LR: {current_lr:.5f} | Train Loss: {train_loss/len(train_loader):.4f} | Val Macro AUC: {mean_auc:.4f}")
    
    scheduler.step(mean_auc)
    
    if mean_auc > best_auc:
        best_auc = mean_auc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_nn_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early Stopping на эпохе {epoch+1}! Лучший AUC: {best_auc:.4f}")
            break

    torch.cuda.empty_cache()
    gc.collect()

print("\nТест завершен!")