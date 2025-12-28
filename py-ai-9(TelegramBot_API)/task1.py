import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

dates = pd.date_range("2024-01-01", periods=30)

users = np.random.randint(100, 600, 30)

sessions = (
    users * np.random.uniform(1.2, 1.5, 30)
    + np.random.randint(-20, 20, 30)
).astype(int)

revenue = (
    sessions * np.random.uniform(10, 15, 30)
).astype(int)

df = pd.DataFrame({
    "date": dates,
    "users": users,
    "sessions": sessions,
    "revenue": revenue
})

corr = df[["users", "sessions", "revenue"]].corr()
print("Кореляційна матриця:\n", corr)

plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.scatter(df["users"], df["sessions"])
plt.xlabel("Users")
plt.ylabel("Sessions")
plt.title("Users vs Sessions")
plt.subplot(1, 3, 2)
plt.scatter(df["users"], df["revenue"])
plt.xlabel("Users")
plt.ylabel("Revenue")
plt.title("Users vs Revenue")
plt.subplot(1, 3, 3)
plt.scatter(df["sessions"], df["revenue"])
plt.xlabel("Sessions")
plt.ylabel("Revenue")
plt.title("Sessions vs Revenue")
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(df["date"], df["revenue"], marker="o")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.title("Revenue over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
