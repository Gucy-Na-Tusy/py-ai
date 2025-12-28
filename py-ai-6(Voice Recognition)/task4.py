import numpy as np

low = int(input("Введіть нижню межу: "))
high = int(input("Введіть верхню межу: "))

arr = np.random.randint(low, high + 1, size=15)
print("Вихідний масив:", arr)

negatives = arr[arr < 0]
print("Від'ємні елементи:", negatives)

modified = arr.copy()
modified[modified < 0] = 0
print("Масив після заміни від'ємних на нулі:", modified)

zero_count = np.sum(modified == 0)
print("Кількість нульових елементів:", zero_count)
