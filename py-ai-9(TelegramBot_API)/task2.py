import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n = 120
group_A = np.random.binomial(1, 0.10, n)
group_B = np.random.binomial(1, 0.14, n)

df = pd.DataFrame({
    "group": ["A"] * n + ["B"] * n,
    "converted": np.concatenate([group_A, group_B])
})

conv_A = group_A.mean()
conv_B = group_B.mean()

abs_diff = conv_B - conv_A
rel_diff = abs_diff / conv_A

print(f"Conversion A: {conv_A:.3f}")
print(f"Conversion B: {conv_B:.3f}")
print(f"Absolute difference: {abs_diff:.3f}")
print(f"Relative change: {rel_diff:.2%}")

def ci_95(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n)

ci_A = ci_95(conv_A, n)
ci_B = ci_95(conv_B, n)

plt.bar(["A", "B"], [conv_A, conv_B],
        yerr=[ci_A, ci_B], capsize=10)

plt.ylabel("Conversion rate")
plt.title("A/B Test Conversion with 95% CI")
plt.show()
