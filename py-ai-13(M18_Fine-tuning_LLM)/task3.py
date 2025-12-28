import numpy as np

start = int(input("Початок діапазону: "))
end = int(input("Кінець діапазону: "))

sequence = np.arange(start, end)

if sequence.size < 30:
    print(f"Помилка: У діапазоні лише {sequence.size} чисел, а потрібно мінімум 30.")
else:
    matrix = sequence[:30].reshape(6, 5)
    print("Матриця:\n", matrix)

    row_sums = matrix.sum(axis=1)
    print("Суми рядків:", row_sums)

    col_max = matrix.max(axis=0)
    print("Максимуми стовпчиків:", col_max)
