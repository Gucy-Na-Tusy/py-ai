import numpy as np

start = int(input("Введіть початок діапазону: "))
end = int(input("Введіть кінець діапазону: "))

while (end - start) < 30:
    print(f"Помилка! Різниця між числами має бути мінімум 30 (зараз {end - start}).")
    start = int(input("Введіть початок діапазону ще раз: "))
    end = int(input("Введіть кінець діапазону ще раз: "))

sequence = np.arange(start, end)

matrix = sequence[:30].reshape(6, 5)
print("Матриця 6x5:\n", matrix)

row_sums = np.sum(matrix, axis=1)
print("Суми елементів у кожному рядку:", row_sums)

col_max = np.max(matrix, axis=0)
print("Максимальні значення по стовпцях:", col_max)
