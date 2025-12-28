import numpy as np

length = int(input("Довжина масивів: "))

arr1 = np.random.randint(0, 11, length)
arr2 = np.random.randint(10, 21, length)

print("Перший масив:", arr1)
print("Другий масив:", arr2)

combined = np.concatenate((arr1, arr2))
print("Об'єднаний масив:", combined)

sum_arr = arr1 + arr2
diff_arr = arr1 - arr2

print("Поелементна сума:", sum_arr)
print("Поелементна різниця:", diff_arr)
