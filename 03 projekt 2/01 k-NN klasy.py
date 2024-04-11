import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Przykładowy zbiór danych 2D
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [1, 3], [2, 4], [3, 5],
              [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [6, 8], [7, 9], [8, 10]])
y = np.array([0, 0, 0, 1, 1, 0,1 , 0, 1, 0, 0, 1, 1, 1, 1, 1])  # Przykładowe etykiety klas (0 lub 1)


# Utworzenie i dopasowanie klasyfikatora k-NN
knn = KNeighborsClassifier(n_neighbors=1)
'''
KNeighborsClassifier to klasa w bibliotece scikit-learn,
która implementuje algorytm klasyfikacji k najbliższych sąsiadów (k-NN).
Algorytm ten jest często używany ze względu na swoją prostotę i skuteczność,
szczególnie gdy dane mają dobrze zdefiniowane struktury klastrów.
Jednakże, dla dużych zbiorów danych może być kosztowny obliczeniowo,
ponieważ wymaga przechowywania wszystkich punktów w pamięci.
'''
knn.fit(X, y)
'''
Podczas wywoływania knn.fit(X, y), model k-NN "uczy się" na danych treningowych,
co oznacza, że analizuje je i buduje wewnętrzną reprezentację,
która umożliwia późniejsze klasyfikowanie nowych punktów danych.
Proces ten polega na przechowywaniu lub tworzeniu struktury danych,
która umożliwia szybkie znajdowanie najbliższych sąsiadów dla nowych danych,
gdy model jest wykorzystywany do przewidywania klas.
'''

# Wizualizacja zbioru danych
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('2D Data Visualization with Classes')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Dodanie dodatkowych punktów
X_extra = np.array([[3, 6], [8, 4], [6, 6]])
y_extra = knn.predict(X_extra)
plt.scatter(X_extra[:, 0], X_extra[:, 1], marker='x', c=y_extra, cmap='viridis', label='Additional Points')

plt.legend()
plt.show()

'''Zadanie na zajęcia - zmień wartości niektórych klas 0 i 1 na 2 i zobacz jak zmieni się rezultat'''