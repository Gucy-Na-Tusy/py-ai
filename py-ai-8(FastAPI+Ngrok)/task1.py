import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
dates = pd.date_range(start="2024-01-01", periods=30)

users = np.random.randint(100, 500, size=30)
sessions = users * np.random.uniform(1.2, 1.5, size=30) + np.random.randint(-20, 20, size=30)
revenue = sessions * np.random.uniform(5, 10, size=30)

data = pd.DataFrame({
    "date": dates,
    "users": users,
    "sessions": sessions.astype(int),
    "revenue": revenue.astype(int)
})

corr_matrix = data[["users", "sessions", "revenue"]].corr()
print("Кореляційна матриця:\n", corr_matrix)

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.scatter(data["users"], data["sessions"], alpha=0.7, color='blue')
plt.title("Users vs Sessions")
plt.xlabel("Users")
plt.ylabel("Sessions")
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(1, 3, 2)
plt.scatter(data["users"], data["revenue"], alpha=0.7, color='green')
plt.title("Users vs Revenue")
plt.xlabel("Users")
plt.ylabel("Revenue")
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(1, 3, 3)
plt.scatter(data["sessions"], data["revenue"], alpha=0.7, color='red')
plt.title("Sessions vs Revenue")
plt.xlabel("Sessions")
plt.ylabel("Revenue")
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(data["date"], data["revenue"])
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.title("Revenue over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
