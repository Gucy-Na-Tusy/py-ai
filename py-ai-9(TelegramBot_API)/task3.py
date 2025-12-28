import numpy as np
import matplotlib.pyplot as plt

population = np.random.exponential(scale=2, size=50000)

sample_sizes = [5, 30]

for n in sample_sizes:
    sample_means = []

    for _ in range(1000):
        sample = np.random.choice(population, n)
        sample_means.append(sample.mean())

    sample_means = np.array(sample_means)

    print(f"\nn = {n}")
    print("Mean of sample means:", sample_means.mean())
    print("Std of sample means:", sample_means.std())

    plt.hist(sample_means, bins=30, density=True)
    plt.title(f"Sampling Distribution (n={n})")
    plt.xlabel("Sample Mean")
    plt.ylabel("Density")
    plt.show()
