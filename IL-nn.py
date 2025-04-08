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

# Parse strings into Python objects
df['Action'] = df['Action'].apply(ast.literal_eval)
df['LiDAR_processed'] = df['LiDAR'].apply(lambda x: ast.literal_eval(x)[0] if x else [])

# Drop raw LiDAR string column
df = df.drop(columns=["LiDAR"])

# Convert into NumPy arrays
X = np.array([[row['Velocity']] + row['LiDAR_processed'] for _, row in df.iterrows()])
y = np.stack(df['Action'].values)

# === Train/Validation Split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === PyTorch DataLoaders ===
batch_size = 100

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# === Model Definition ===
class BCNet(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim)
    )


    def forward(self, x):
        return self.net(x)

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BCNet(input_dim=X.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# === Training Loop ===
num_epochs = 100
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


# === Save the trained model ===
model_path = "bc_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
