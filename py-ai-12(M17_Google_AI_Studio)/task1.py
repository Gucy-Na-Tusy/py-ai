import numpy as np

arr = np.random.randint(0, 51, 20)
print("Масив:", arr)

threshold = int(input("Введіть порогове значення: "))

count = np.sum(arr > threshold)
print("Кількість елементів більших за поріг:", count)

max_value = arr.max()
max_index = np.where(arr == max_value)[0][0]
print("Максимальне значення:", max_value)
print("Позиція першої появи:", max_index)

sorted_arr = np.sort(arr)[::-1]
print("Відсортований масив:", sorted_arr)
