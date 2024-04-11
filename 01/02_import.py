#%%
#Import pełnej nazwy biblioteki:
import numpy
# Użycie funkcji lub klas z biblioteki numpy
numpy.array([1, 2, 3])

#%%
#Import pełnej nazwy biblioteki z aliasem:
import numpy as np
# Użycie funkcji lub klas z biblioteki numpy za pomocą aliasu np
np.array([1, 2, 3])

#%%
#Import konkretnej funkcji lub klasy z biblioteki:
from numpy import array
# Można używać funkcji array bez prefixu numpy
array([1, 2, 3])

#%%
#Import konkretnej funkcji lub klasy z biblioteki z aliasem:
from numpy import array as arr
# Użycie funkcji array z aliasem arr
arr([1, 2, 3])

#%%
#Import całej biblioteki, ale jedynie wybrane funkcje lub klasy:
from numpy import array, zeros
# Można używać funkcji array i zeros bez prefixu numpy
array([1, 2, 3])
zeros((3, 3))

#%%
#Import wszystkiego z biblioteki (niezalecane z powodu ryzyka nadpisania istniejących funkcji lub klas):
from numpy import *
# Uwaga: ta forma importu importuje wszystko z biblioteki numpy, co może prowadzić do konfliktów nazw