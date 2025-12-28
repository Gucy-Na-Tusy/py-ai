import numpy as np
import matplotlib.pyplot as plt

population = np.random.exponential(scale=2, size=50000)

sample_sizes = [5, 30]
means_dict = {}

for n in sample_sizes:
    sample_means = []
    for _ in range(1000):
        sample = np.random.choice(population, size=n)
        sample_means.append(sample.mean())

    means_dict[n] = np.array(sample_means)

    print(f"n = {n}")
    print("Mean:", np.mean(sample_means))
    print("Std:", np.std(sample_means))

    plt.hist(sample_means, bins=30, density=True)
    plt.title(f"Sampling Distribution (n={n})")
    plt.xlabel("Sample Mean")
    plt.ylabel("Density")
    plt.show()
