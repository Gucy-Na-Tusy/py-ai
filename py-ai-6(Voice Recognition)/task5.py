import numpy as np

length = int(input("Введіть довжину масивів: "))

arr1 = np.random.randint(0, 11, size=length)
arr2 = np.random.randint(10, 21, size=length)

print("Перший масив:", arr1)
print("Другий масив:", arr2)

combined = np.concatenate((arr1, arr2))
print("Об'єднаний масив:", combined)

sum_arrays = arr1 + arr2
diff_arrays = arr1 - arr2

print("Поелементне додавання:", sum_arrays)
print("Поелементна різниця:", diff_arrays)
