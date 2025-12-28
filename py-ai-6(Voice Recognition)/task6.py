import numpy as np

rows = int(input("Введіть кількість рядків: "))
cols = int(input("Введіть кількість стовпців: "))

total_elements = rows * cols

matrix = np.arange(1, rows * cols + 1).reshape(rows, cols)
print("Вихідна матриця:\n", matrix)

print(f"\n--- Зміна розміру (всього елементів: {total_elements}) ---")
new_rows = int(input("Введіть нову кількість рядків: "))
new_cols = int(input("Введіть нову кількість стовпців: "))


if new_rows * new_cols != total_elements:
    print(f"Помилка! Неможливо перетворити матрицю.")
    print(f"У вас є {total_elements} чисел, а для розміру {new_rows}x{new_cols} потрібно {new_rows * new_cols}.")
else:
    reshaped = matrix.reshape(new_rows, new_cols)
    print("Перетворена матриця:\n", reshaped)

    row_min = np.min(reshaped, axis=1)
    row_max = np.max(reshaped, axis=1)

    print("Мінімальні значення в рядках:", row_min)
    print("Максимальні значення в рядках:", row_max)

    print("Загальна сума всіх елементів:", np.sum(reshaped))
