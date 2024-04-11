#%%
import numpy as np
# Tworzenie tablicy 1D z listy
array_1d = np.array([1, 2, 3, 4, 5])
print("Tablica 1D:", array_1d)
'''
Dlaczego tablice?
NumPy używa natywnych funkcji napisanych w języku C, co przyspiesza operacje na danych.
Mniej pamięci niż zwykłe listy Pythona.
NumPy dostarcza wiele wbudowanych funkcji i operacji do pracy na tablicach.
'''
# Tworzenie tablicy 2D z listy list
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Tworzenie tablicy o wartościach losowych
print(np.random.rand(3, 3))
# Tworzenie macierzy jednostkowej 3x3
print(np.eye(3))
# Tworzenie macierzy zerowej 4x4
print(np.zeros((4, 4)))
# Tworzenie macierzy wypełnionej wartościami 5 3x3
print(np.full((3, 3), 5))

#%%
# Obliczanie sumy wszystkich elementów tablicy
print("Suma:", np.sum(array_1d))
# Obliczanie średniej arytmetycznej
print("Średnia:", np.mean(array_1d))
# Znajdowanie największej wartości w tablicy
print("Maksimum:", np.max(array_1d))
# Tworzenie nowej tablicy z pierwiastkami kwadratowymi wartości oryginalnej tablicy
print(np.sqrt(array_1d))

# %%
# Indeksowanie i wycinanie tablic NumPy
# Wyświetlanie wartości na konkretnych indeksach
print("Wartość na indeksie (1, 1):", array_2d[1, 1])
# Wycinanie fragmentu tablicy
print(array_2d[:2, 1:])
# Zaawansowane indeksowanie w NumPy
# array_2d > 5 generuje tablicę maski True/False
print(array_2d > 5)
# Wybieranie wartości większych niż 5
print(array_2d[array_2d > 5])

# %%
# Tworzenie dwóch macierzy
matrix_A = np.array([[1, 2], [3, 4]])
matrix_B = np.array([[5, 6], [7, 8]])
# Dodawanie macierzy
print(matrix_A + matrix_B)
# Mnożenie macierzy
print(np.dot(matrix_A, matrix_B))
# Transpozycja macierzy
print(np.transpose(matrix_A))
# Odwracanie macierzy
print(np.linalg.inv(matrix_A))

# %%