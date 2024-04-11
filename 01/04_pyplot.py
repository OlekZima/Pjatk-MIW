#%%
import matplotlib.pyplot as plt
'''
Matplotlib to popularna biblioteka do tworzenia wykresów i wizualizacji danych w języku Python.
matplotlib.pyplot to moduł biblioteki Matplotlib.
pyplot zawiera wiele funkcji, które umożliwiają tworzenie różnych rodzajów wykresów, manipulowanie nimi oraz dostosowywanie ich wygląd.
'''

#%%
# Dane początkowe
x1 = [1, 2, 3, 4, 5]
y1 = [2, 3, 5, 7, 11]
data1 = {'x':[1, 2, 3, 4, 5],'y':[2, 3, 5, 7, 11]}
# Nowe dane
x2 = list(range(1,6))
y2 = [1, 4, 7, 7, 7]
data2 = {'x':[1, 2, 3, 4, 5],'y':[1, 4, 7, 7, 7]}

#%%
# Tworzenie wykresu liniowego dla pierwszego zestawu danych
plt.plot(x1, y1)
plt.plot / plt.scatter / plt.bar
plt.scatter(x1, y1, color='blue', linestyle='dashed', marker='o', label='Dane 1')
plt.scatter(data1['x'], data1['y'], color='blue', linestyle='dashed', marker='o', label='Dane 1')
'''
'o': kropa,
'.': kropka,
's': kwadrat,
'+': krzyżyk,
'*': gwiazdka.
'''
# Dodanie nowego zestawu danych do istniejącego wykresu
plt.bar(x2, y2, color='red', linestyle='solid', marker='x', label='Dane 2')
plt.bar(x2, y2, color='red',linestyle='solid',label='Dane 2')

# Dodanie tytułu i etykiet osi
plt.title('Wykres liniowy z dwoma zestawami danych')
plt.xlabel('Oś X')
plt.ylabel('Oś Y')

# Dodanie legendy
plt.legend()

# Wyświetlenie wykresu
plt.show()

# %%
