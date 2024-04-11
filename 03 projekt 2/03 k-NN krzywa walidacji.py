# %%
# biblioteki i dane
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# Wczytanie zbioru danych Iris
iris = load_iris()
X, y = iris.data, iris.target

# %%
# Prezentacja danych
'''
Na wykresie przedstawione są dane z zestawu danych Iris.
Każdy punkt na wykresie reprezentuje jedną próbkę danych.
Kolor punktu odpowiada klasie, do której należy dana próbka, zgodnie z legendą kolorów.
'''
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('Iris Data Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Target')
plt.show()

# %%
# Definicja wartości liczby sąsiadów do sprawdzenia
neighbors = np.arange(1, 20)

# Walidacja krzyżowa dla różnych liczności sąsiadów
cv_scores = []
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')#cv w tej funkcji jest parametrem określającym liczbę podziałów (foldów), 
    cv_scores.append(scores.mean())
'''
Walidacja krzyżowa (ang. Cross-Validation) to technika oceny wydajności modelu,
która polega na podziale dostępnych danych na zbiór treningowy i zbiór testowy,
aby ocenić, jak dobrze model generalizuje się do nowych, nieznanych danych.
W przeciwieństwie do tradycyjnego podziału danych na zbiór treningowy i zbiór testowy,
gdzie dostępna jest tylko jedna taka podział, walidacja krzyżowa wykonuje tę procedurę wielokrotnie,
aby uzyskać bardziej wiarygodne estymacje wydajności modelu.

Idea walidacji krzyżowej polega na tym, że dane dzielone są na k części (ang. folds),
a następnie model jest trenowany k razy, każdorazowo na k-1 częściach danych (zbiór treningowy),
a testowany na pozostałej części (zbiór testowy). Proces ten powtarza się k razy,
przy czym każda z k części służy jako zbiór testowy dokładnie raz.
Ostateczna wydajność modelu to średnia wyników testowych uzyskanych w każdym z k eksperymentów.
'''

# %%
# Wykres walidacji krzyżowej
'''
Pokazuje średnią dokładność modelu dla różnych wartości liczby sąsiadów.
Im wyższa wartość na osi y, tym lepsza skuteczność modelu.
Można zauważyć, że dokładność zwykle maleje wraz ze wzrostem liczby sąsiadów, co może oznaczać zjawisko przetrenowania (overfitting) dla dużych wartości k.
Optymalna wartość liczby sąsiadów to ta, dla której uzyskujemy najwyższą dokładność.
'''
plt.figure()
plt.plot(neighbors, cv_scores, marker='o')
plt.title('Cross-Validation Scores for Different Number of Neighbors')
plt.xlabel('Number of Neighbors')
plt.ylabel('Mean CV Accuracy')
plt.xticks(neighbors)
plt.grid(True)
plt.show()

# %%
# Krzywa walidacji dla liczby sąsiadów
train_scores, test_scores = validation_curve(KNeighborsClassifier(), X, y, param_name="n_neighbors", param_range=neighbors, cv=5, scoring="accuracy")
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

#%%
# Wykres krzywej walidacji
'''
Pokazuje dokładność modelu na danych treningowych i testowych w zależności od liczby sąsiadów.
Linie reprezentują średnie wyniki walidacji dla danych treningowych i testowych.
Obszary wypełnione kolorem przedstawiają odchylenie standardowe wyników.
Zwykle optymalna liczba sąsiadów to ta, dla której dokładność na zbiorze testowym (krzywa walidacji) jest najwyższa, a odchylenie standardowe jest akceptowalne.
'''
plt.figure()
plt.plot(neighbors, train_mean, label="Training score", color="darkorange", marker='o')
plt.fill_between(neighbors, train_mean - train_std, train_mean + train_std, alpha=0.2, color="darkorange")
plt.plot(neighbors, test_mean, label="Cross-validation score", color="navy", marker='o')
plt.fill_between(neighbors, test_mean - test_std, test_mean + test_std, alpha=0.2, color="navy")
plt.title("Validation Curve with k-NN")
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.xticks(neighbors)
plt.legend(loc="best")
plt.grid(True)
plt.show()

# %%
'''
PS.
Odchylenie standardowe jest miarą rozproszenia wartości
wokół średniej arytmetycznej w zbiorze danych. 

Jeśli odchylenie standardowe jest niskie, oznacza to,
że większość wyników walidacji jest zbliżona do średniej,
co sugeruje stabilność modelu.
Natomiast wysokie odchylenie standardowe oznacza,
że wyniki walidacji różnią się od siebie bardziej znacząco,
co może wskazywać na niestabilność modelu
lub wrażliwość na sposób podziału danych.
'''

'''Zadanie na zajęcia : przeanalizują walidację krzyżową'''