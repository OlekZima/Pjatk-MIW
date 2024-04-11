import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generowanie przykładowego zbioru danych za pomocą funkcji make_blobs
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
'''
X jest to macierz zawierająca współrzędne punktów danych w przestrzeni cech. 
_ to wektor zawierający etykiety przypisane do każdego punktu danych. W kontekście funkcji make_blobs, ta wartość nie jest używana
cluster_std odchylenie standardowe klastrów generowanych danych.
Jest to wartość liczbową, która kontroluje, jak bardzo punkty danych w każdym klastrze są rozproszone wokół swojego centrum.
'''

# Inicjalizacja i dopasowanie modelu KMeans z 4 klastrami
kmeans = KMeans(n_clusters=4, n_init=10)
'''
jeśli n_init wynosi 10, algorytm KMeans zostanie uruchomiony 10 razy
z różnymi losowymi początkowymi pozycjami centroidów,
a następnie wybierze najlepszą końcową konfigurację klastrów,
która ma najniższą wartość funkcji kosztu (np. sumę kwadratów odległości punktów od swoich centroidów).
'''
kmeans.fit(X)

# Pobranie współrzędnych centrów klastrów
centers = kmeans.cluster_centers_
'''
Atrybut cluster_centers_ jest częścią obiektu modelu
po dopasowaniu algorytmu k-means do danych za pomocą biblioteki scikit-learn.
Zawiera on współrzędne centrów klastrów,
które zostały znalezione przez algorytm k-means.
'''
# Przypisanie etykiet klastrów dla każdej próbki
labels = kmeans.labels_

# Wizualizacja klastrów i ich centrów na wykresie
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)  # Punkty danych pokolorowane na podstawie przypisanych im klastrów
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centroids')  # Środki klastrów oznaczone na czerwono
plt.title('K-Means Clustering')  # Tytuł wykresu
plt.xlabel('Feature 1')  # Etykieta osi x
plt.ylabel('Feature 2')  # Etykieta osi y
plt.legend()  # Legenda
plt.show()  # Wyświetlenie wykresu


'''
Wyciek pamięci (memory leak) to sytuacja, w której program zużywa coraz więcej pamięci podczas jego działania,
ale nie zwalnia jej prawidłowo po zakończeniu używania danych.
W rezultacie program zużywa coraz więcej pamięci RAM,
co może prowadzić do spowolnienia systemu lub nawet jego zawieszenia, gdy pamięć RAM jest całkowicie wypełniona.

Istnieje znany wyciek pamięci w algorytmie KMeans, który występuje na systemie Windows z biblioteką MKL (Math Kernel Library)

Rada z forum: otwórz wiersz polecenia, wydaj polecenie set OMP_NUM_THREADS=2 i naciśnij Enter.
'''