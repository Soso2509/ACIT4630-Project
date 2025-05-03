import ast
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

print(torch.version.hip)
print(torch.backends.mps.is_available())  
print(torch.cuda.is_available()) 
print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print(torch.device("mps" if torch.backends.mps.is_available() else "cpu"))

# === Load and preprocess data ===
# NOTE: This might have data leakage problems? 
df = pd.read_csv('demonstration_filtered.csv')

df['Action'] = df['Action'].apply(ast.literal_eval)
df['LiDAR_processed'] = df['LiDAR'].apply(lambda c: ast.literal_eval(c)[0] if c else [])

df = df.drop(columns=["LiDAR"])

x = np.array([[row['Velocity']] + row['LiDAR_processed'] for _, row in df.iterrows()])
y = np.stack(df['Action'].values)

# === Train/Validation Split ===
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# === PyTorch DataLoaders ===
batch_size = 100 # 100 seems good
SEQ_LEN = 20 # increasing massively increases training time, 20 seems good

x_seq = []
y_seq = []

for i in range(len(x) - SEQ_LEN):
    x_seq.append(x[i:i + SEQ_LEN])
    y_seq.append(y[i + SEQ_LEN - 1])

x_seq = np.array(x_seq)
y_seq = np.array(y_seq)

train_dataset = TensorDataset(torch.tensor(x_seq[:len(x_seq)*4//5], dtype=torch.float32),
                              torch.tensor(y_seq[:len(y_seq)*4//5], dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(x_seq[len(x_seq)*4//5:], dtype=torch.float32),
                            torch.tensor(y_seq[len(y_seq)*4//5:], dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# === Model Definition ===
class BCNet(nn.Module):
    def __init__(self, input_dim, output_dim=3, hidden_dim=256):
        super().__init__()
        self.rnn = nn.LSTM(
            input_dim, 
            hidden_dim,
            num_layers=1, 
            batch_first=True)

        self.fc1 = nn.Linear(hidden_dim, 512)
        self.tanh1 = nn.Tanh()

        self.fc2 = nn.Linear(512, 512)
        self.tanh2 = nn.Tanh()

        self.fc3 = nn.Linear(512, 512)
        self.tanh3 = nn.Tanh()

        self.fc4 = nn.Linear(512, 512)
        self.tanh4 = nn.Tanh()

        self.fc5 = nn.Linear(512, 512)
        self.tanh5 = nn.Tanh()

        self.fc6 = nn.Linear(512, 512)
        self.tanh6 = nn.Tanh()

        self.fc7 = nn.Linear(512, 512)
        self.tanh7 = nn.Tanh()

        self.fc8 = nn.Linear(512, 512)
        self.tanh8 = nn.Tanh()

        self.fc9 = nn.Linear(512, 512)
        self.tanh9 = nn.Tanh()

        self.fc10 = nn.Linear(512, 256)
        self.tanh10 = nn.Tanh()

        self.out = nn.Linear(256, output_dim)

        self.throttle_head = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
        self.brake_head = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
        self.steer_head = nn.Sequential(nn.Linear(256, 1), nn.Tanh())

    dropout = nn.Dropout(p=0.3)



    # Forward pass
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        rnn_out, _ = self.rnn(x)

        if rnn_out.shape[1] == 1:
            x = rnn_out.squeeze(1)
        else:
            x = rnn_out[:, -1, :]

        x = self.dropout(self.tanh1(self.fc1(x)))
        x = self.dropout(self.tanh2(self.fc2(x)))
        x = self.dropout(self.tanh3(self.fc3(x)))
        x = self.dropout(self.tanh4(self.fc4(x)))
        x = self.dropout(self.tanh5(self.fc5(x)))
        x = self.dropout(self.tanh6(self.fc6(x)))
        x = self.dropout(self.tanh7(self.fc7(x)))
        x = self.dropout(self.tanh8(self.fc8(x)))
        x = self.dropout(self.tanh9(self.fc9(x)))
        x = self.dropout(self.tanh10(self.fc10(x)))

        throttle = self.throttle_head(x)
        brake = self.brake_head(x)
        steer = self.steer_head(x)

        return torch.cat([throttle, brake, steer], dim=1)

# === Setup ===
device = torch.device("cuda" if torch.version.cuda else "cpu")
model = BCNet(input_dim=x.shape[1]).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5) # L2 regularization and 0.0001 learning rate
criterion = nn.MSELoss() # Is MSE the best loss here?

# def set_seed(seed=42): # for reproducability when tuning ! Might not be working correctly
#     random.seed()
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
    
# set_seed(42)

# === Training Loop ===
num_epochs = 400 # Usually early stops anyways
best_val_loss = float('inf')
patience = 10
counter = 0
train_losses = []
val_losses = []
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        preds = model(batch_X)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

# === Metrics ===
finish_time = time.time() - start_time
print(f'Model finished training in {finish_time:.2f} seconds')
print(f'Best loss was {best_val_loss:.4f}')


# === Saving the trained model ===
model_path = "bc_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# === Plotting ===

import matplotlib.pyplot as plt

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()