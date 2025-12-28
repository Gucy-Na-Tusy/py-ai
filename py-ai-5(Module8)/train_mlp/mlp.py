import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

df = pd.read_csv("../data/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

model = nn.Sequential(
    nn.Linear(X_train.shape[1], 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 30
losses = []
accuracies = []

for epoch in range(epochs):
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    preds = (outputs >= 0.5).float()
    acc = accuracy_score(y_train.numpy(), preds.numpy())

    losses.append(loss.item())
    accuracies.append(acc)

    print(f"Епоха [{epoch+1}/{epochs}] | Loss: {loss.item():.4f} | Accuracy: {acc:.4f}")

model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    test_preds = (test_outputs >= 0.5).float()
    test_acc = accuracy_score(y_test.numpy(), test_preds.numpy())

print("\nПідсумкові Метрики:")
print(f"Loss: {test_loss.item():.4f}")
print(f"Accuracy: {test_acc:.4f}")

os.makedirs("../results", exist_ok=True)

plt.figure()
plt.plot(losses, label="Loss")
plt.xlabel("Епоха")
plt.ylabel("Loss")
plt.legend()
plt.savefig("../results/loss_healthrisk_mlp.png")
plt.close()

plt.figure()
plt.plot(accuracies, label="Accuracy")
plt.xlabel("Епоха")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("../results/accuracy_healthrisk_mlp.png")
plt.close()
