#%%
#import
import numpy as np
from scipy.linalg import eig

#%%
# Definicja macierzy przejść
P = np.array([[0.2, 0.3, 0.5],
              [0.4, 0.1, 0.5],
              [0.6, 0.0, 0.4]])

#%%
# Obliczenie wartości własnych i wektorów własnych
vals, vecs = eig(P, left=True, right=False)
# Ustawiamy left=True i right=False, aby uzyskać wektory własne lewostronne (stacjonarne).
print(vals)
print(vecs)

#%%
# Index wartości własnej bliskiej 1
print(np.isclose(vals, 1.))
print(np.where(np.isclose(vals, 1.)))
index = np.where(np.isclose(vals, 1.))[0][0]
# np.isclose(vals, 1.) zwraca tablicę wartości logicznych, które są True, jeśli odpowiadający element w vals jest bliski 1, a False w przeciwnym razie.
# Funkcja np.where() w bibliotece NumPy jest używana do znalezienia indeksów elementów w tablicy, które spełniają określone warunki.
print(index)

#%%
# Znalezienie wektora własnego dla wartości własnej równiej 1
stationary_vector = np.real(vecs[:, index])  # Rzeczywista część wektora własnego
print(stationary_vector)