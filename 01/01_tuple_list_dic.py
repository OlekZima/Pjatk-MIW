#%%
'''
Listy są dynamiczne i mutowalne.
Możemy dodawać, usuwać i modyfikować elementy listy po jej utworzeniu.
'''
# Definicja listy
my_list = [1, 2, 3, 4, 5]
# Drukowanie listy
print("Lista:", my_list)
print("Lista:", my_list[1:4])
print("Lista:", my_list[0:3:2])
print("Lista:", my_list[::-1])
my_list[1] = 11
# Modyfikacja listy - dodanie nowego elementu
my_list.append(6)
# Drukowanie zmienionej listy
print("Lista po dodaniu nowego elementu:", my_list)

#%%
'''
Krotki są niemutowalne, co oznacza, że nie możemy ich zmieniać po ich utworzeniu.
'''
# Definicja krotki
my_tuple = (1, 2, 3, 4, 5)
# Drukowanie krotki
print("Krotka:", my_tuple)
# Próba zmiany elementu krotki (spowoduje błąd)
# my_tuple[0] = 6

#%%
'''
Jeśli używasz jednego return z wieloma wartościami,
Python automatycznie zamieni te wartości na krotkę.
'''
def example_function():
    return 1, 2, 3
result = example_function()
print(result)  # Output: (1, 2, 3)
print(type(result))  # Output: <class 'tuple'>
'''
Wewnętrznie Python po prostu zwraca te trzy wartości jako jedną krotkę.
'''
result1, result2, resul3 = example_function()
print(result1)  # Output: (1, 2, 3)
print(type(result1))  # Output: <class 'tuple'>

# %%
# Definicja słownika
my_dict = {'klucz1': 'wartość1', 'klucz2': 'wartość2', 'klucz3': 'wartość3'}
# Drukowanie słownika
print("Słownik:", my_dict)
# Dostęp do wartości za pomocą klucza
print("Wartość dla klucza 'klucz2':", my_dict['klucz2'])
# Modyfikacja słownika - dodanie nowej pary klucz-wartość
my_dict['klucz4'] = 'wartość4'
# Drukowanie zmienionego słownika
print("Słownik po dodaniu nowej pary klucz-wartość:", my_dict)