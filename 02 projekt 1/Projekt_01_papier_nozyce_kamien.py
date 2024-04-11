import random
import numpy as np
import matplotlib.pyplot as plt

# Opis: Program symuluje grę w "Kamień, Papier, Nożyce" pomiędzy graczem a komputerem,
#       aktualizując macierz przejść na podstawie wyników i uczenia się w czasie rzeczywistym.

# Inicjalizacja stanu gotówki gracza
cash = 0
cash_history = [cash]

##### KOMPUTER #####
# Definicja ruchów/taktyki komputera
states_computer = ["Paper", "Rock", "Scissors"]
transition_matrix_computer = {
    "Paper": {"Paper": 2/3, "Rock": 1/3, "Scissors": 0/3},
    "Rock": {"Paper": 0/3, "Rock": 2/3, "Scissors": 1/3},
    "Scissors": {"Paper": 2/3, "Rock": 0/3, "Scissors": 1/3}
}
# Przekształcenie macierzy przejść transition_matrix_computer do postaci tablicy numpy
'''do uzupelnienia'''
# Funkcja wybierająca ruch komputera na podstawie macierzy przejść tj. na podstawie swojego poprzedniego wyboru
def choose_move(player_previous_move):
    '''do uzupelnienia'''

##### GRACZ #####
# Definicja ruchów gracza:
#   wersja 1: na podstawie wektora stacjonarnego transition_matrix_computer,
#   wersja 2: w trakcie gry(iteracji) nauczenie gracza taktyki w postaci jego macierzy przejść
#             (inicjujemy macierz przejść gracza wypełnioną np. 1/3, a w trakcie gry po każdej rundzie aktualizujemy ją). 
# Należy napisać kod dla obu wersji (w osobnych plikach, albo w jednym pliku z możliwością zmiany taktyki jakimś parametrem)

# Obliczanie wektora stacjonarnego macierzy przejść transition_matrix_computer (wersja 1 taktyki gracza)
'''do uzupelnienia'''

# Funkcja aktualizująca macierz przejść gracza (wersja 2 taktyki gracza)
'''do uzupelnienia'''
# Funkcja wybierająca ruch gracza na podstawie macierzy przejść tj. na podstawie swojego poprzedniego wyboru (wersja 2 taktyki gracza)
'''do uzupelnienia'''

# Główna pętla gry
for _ in range(1000):
    '''do uzupelnienia'''

# Wykres zmiany stanu gotówki w każdej kolejnej grze
plt.plot(range(100001), cash_history)
plt.xlabel('Numer Gry')
plt.ylabel('Stan Gotówki')
plt.title('Zmiana Stanu Gotówki w Grze "Kamień, Papier, Nożyce"')
plt.show()