import ast
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# === Load and preprocess data ===
df = pd.read_csv('demonstration_filtered.csv')

df['Action'] = df['Action'].apply(ast.literal_eval)
df['LiDAR_processed'] = df['LiDAR'].apply(lambda c: ast.literal_eval(c)[0] if c else [])

df = df.drop(columns=["LiDAR"])

x = np.array([[row['Velocity']] + row['LiDAR_processed'] for _, row in df.iterrows()])
y = np.stack(df['Action'].values)

# === Train/Validation Split ===
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# === PyTorch DataLoaders ===
batch_size = 100
SEQ_LEN = 10

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
    def __init__(self, input_dim, output_dim=3, hidden_dim=32):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        self.fc1 = nn.Linear(hidden_dim, 32)
        self.tanh1 = nn.Tanh()

        self.fc2 = nn.Linear(32, 32)
        self.tanh2 = nn.Tanh()

        self.out = nn.Linear(32, output_dim)

    dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        rnn_out, _ = self.rnn(x)

        if rnn_out.shape[1] == 1:
            x = rnn_out.unsqueeze(1)

        else:
            x = rnn_out[:, -1, :]

        x = self.dropout(self.tanh1(self.fc1(x)))
        x = self.dropout(self.tanh2(self.fc2(x)))
        return self.out(x)

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BCNet(input_dim=x.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.MSELoss()

# === Training Loop ===
num_epochs = 200
best_val_loss = float('inf')
patience = 15
counter = 0

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

# === Saving the trained model ===
model_path = "bc_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
