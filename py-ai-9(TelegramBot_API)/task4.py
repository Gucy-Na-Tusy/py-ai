import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dates = pd.date_range("2024-01-01", periods=90)
sales = np.random.randint(100, 600, 90)

df = pd.DataFrame({
    "date": dates,
    "sales": sales
})

window = 7
df["rolling_mean"] = df["sales"].rolling(window).mean()
df["rolling_std"] = df["sales"].rolling(window).std()

print(df.head(10))

plt.figure(figsize=(10, 5))
plt.plot(df["date"], df["sales"], label="Sales")
plt.plot(df["date"], df["rolling_mean"], label="Rolling Mean")
plt.legend()
plt.title("Sales and Rolling Mean")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df["date"], df["rolling_std"])
plt.title("Rolling Standard Deviation")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
