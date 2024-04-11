#%%
#import
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#%%
# Dane wejściowe x
x = np.linspace(0, 10, num=100)
# Tworzy 11 równoodległych punktów na odcinku od 0 do 10 i przypisuje je do zmiennej x
print(x)

#%%
# Dane wejściowe y
y = np.cos(x)
print(y)

#%%
# Interpolacja danych
f = interp1d(x, y)
#  zwraca obiekt funkcji interpolacyjnej, dlatego print jest kiepski
print(f)

#%%
# Nowe punkty do interpolacji
x_new = np.linspace(0, 10, num=100)

# Wyświetlanie wyników
plt.plot(x, y, 'o', label='Dane oryginalne')
plt.plot(x_new, f(x_new), '-', label='Interpolacja')
plt.legend()
plt.show()