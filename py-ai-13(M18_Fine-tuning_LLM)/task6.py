import numpy as np

rows = int(input("Кількість рядків: "))
cols = int(input("Кількість стовпців: "))
total_elements = rows * cols

matrix = np.arange(1, total_elements + 1).reshape(rows, cols)
print("Вихідна матриця:\n", matrix)

print(f"Поточна кількість елементів: {total_elements}")
new_rows = int(input("Нові рядки: "))
new_cols = int(input("Нові стовпці: "))

if new_rows * new_cols != total_elements:
    print(f"Помилка: Неможливо перетворити {total_elements} елементів у матрицю {new_rows}x{new_cols}.")
else:
    new_matrix = matrix.reshape(new_rows, new_cols)
    print("Нова матриця:\n", new_matrix)

    row_min = new_matrix.min(axis=1)
    row_max = new_matrix.max(axis=1)

    print("Мінімальні значення:", row_min)
    print("Максимальні значення:", row_max)
    print("Сума всіх елементів:", new_matrix.sum())
