# Importowanie potrzebnych bibliotek i modułów
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import keras
from keras import layers

# Tworzenie zbioru danych
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
# Podział danych na zbiory uczący i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tworzenie modelu sieci neuronowej
'''uzupelnij w zależności od architektury sieci, może wymagać przekształcenia danych wejściowych'''

# wizualizacja modelu
model.summary()
keras.utils.plot_model(model, "my_first_model.png",show_shapes=True)

# Kompilacja modelu
model.compile(optimizer='''uzupelnij''', loss='''uzupelnij''', metrics=['''uzupelnij'''])

# Trenowanie modelu
model.fit('''uzupelnij w zależności od architektury sieci''')

# Ocena modelu na danych testowych
_, test_accuracy = model.evaluate('''uzupelnij w zależności od architektury sieci''', y_test)
_, train_accuracy = model.evaluate('''uzupelnij w zależności od architektury sieci''', y_train)

# Rysowanie granicy decyzyjnej dla sieci neuronowej
plt.figure(figsize=(8, 6))
# Rysowanie punktów danych treningowych
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='.', label='Train')
# Rysowanie punktów danych testowych
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', label='Test')

# Tworzenie siatki punktów dla wykresu konturu decyzji
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))

# Obliczanie wyników dla wszystkich punktów siatki jednocześnie
temp = model.predict('''uzupelnij w zależności od architektury sieci''')
Z = np.round(temp).reshape(xx.shape)

# Rysowanie granicy decyzyjnej na wykresie konturu
plt.contourf(xx, yy, Z, alpha=0.3)

# Ustawienie tytułu wykresu
plt.title(f"Neural Network\nTest accuracy: {test_accuracy:.3f} and train accuracy: {train_accuracy:.3f}")
# Dodanie legendy
plt.legend()
# Wyświetlenie wykresu
plt.tight_layout()
plt.show()