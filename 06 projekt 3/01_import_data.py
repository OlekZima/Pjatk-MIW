import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Wczytanie danych z pliku
with open("Dane/dane15.txt", "r") as file:
    data = file.readlines()

# Przetwarzanie danych do postaci potrzebnej do dopasowania modelu
x_data = []
y_data = []

for line in data:
    x, y = map(float, line.split())
    x_data.append(x)
    y_data.append(y)

# Wykres
plt.scatter(x_data, y_data, color='blue', label='Dane')
plt.xlabel('Wartość X')
plt.ylabel('Wartość Y')
plt.title('Wykres punktów danych')
plt.legend()
plt.show()