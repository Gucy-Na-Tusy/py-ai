import numpy as np

low = int(input("Введіть нижню межу діапазону: "))
high = int(input("Введіть верхню межу діапазону: "))

matrix = np.random.randint(low, high + 1, size=(5, 5))
print("Вихідна матриця:\n", matrix)

diag = np.diag(matrix)
print("Головна діагональ:", diag)
print("Сума головної діагоналі:", np.sum(diag))

matrix_modified = matrix.copy()
matrix_modified[np.triu_indices(5, k=1)] = 0
print("Матриця після обнулення елементів вище діагоналі:\n", matrix_modified)
