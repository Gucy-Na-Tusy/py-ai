import numpy as np

low = int(input("Нижня межа: "))
high = int(input("Верхня межа: "))

matrix = np.random.randint(low, high + 1, (5, 5))
print("Матриця:\n", matrix)

diag = np.diagonal(matrix)
print("Головна діагональ:", diag)
print("Сума діагоналі:", diag.sum())

matrix[np.triu_indices(5, k=1)] = 0
print("Змінена матриця:\n", matrix)
