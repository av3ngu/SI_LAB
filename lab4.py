# ZADANIE 1
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

samochody = fetch_openml('cars1')

print(samochody.keys())
print(samochody['feature_names'])
print(samochody['categories'])
print(samochody['data'][0])

# WYBOR PIERWSZEJ CECHY - MPG, CZYLI ILOSC PRZEJECHANYCH MIL NA GALON PALIWA
# WYBOR CECHY CZWARTEJ - HORSEPOWER, CZYLI MOC SILNIKA
X = samochody.data[:, [0, 3]]
y = samochody['target']
y = [int(elem) for elem in y]

# UZYCIE FUNKCJI DO PODZIALU ZBIORU NA ZBIOR UCZACY I TESTOWY
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# TWORZENIE KLASYFIKATORA Z PIĘCIOMA KLASTRAMI
kmn = KMeans(n_clusters=5)

# NAUKA KLASYFIKATORA NA DANYCH TRENINGOWYCH
kmn.fit(X_train)

# WYCIAGNIECIE PUNKTOW CENTRALNYCH KLASTROW - BEDA WIDOCZNE OBOK PUNKTOW ZE ZBIORU UCZACEGO
centra = kmn.cluster_centers_

fig, ax = plt.subplots(1, 2)
# PIERWSZY WYKRES TO ZBIOR UCZACY Z PRAWDZIWYMI KLASAMI
ax[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20)

# UZYCIE DANYCH TRENINGOWYCH ABY SPRAWDZIC CO KLASYFIKATOR O NICH MYSLI
y_pred_train = kmn.predict(X_train)
ax[1].scatter(X_train[:, 0], X_train[:, 1], c=y_pred_train, s=20)

# DOKŁADAMY NA DRUGIM WYKRESIE CENTRA KLASTRÓW
ax[1].scatter(centra[:, 0], centra[:, 1], c='red', s=50)
plt.show()

# PRZEWIDZENIE KLAS SAMOCHODOW DLA ZBIORU TESTOWEGO
y_pred = kmn.predict(X_test)

# NOWE KLASY SAMOCHODOW PRZEWIDZIANE PRZEZ KLASTROWANIE
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, s=20)

# CENTRA KLASTROW
plt.scatter(centra[:, 0], centra[:, 1], c='red', s=50)
plt.show()


# ZADANIE 2
# W POSZCZEGOLNYCH KLASTRACH ZNALAZLY SIE SAMOCHODY O ROZNEJ CHARAKTERYSTYCE
# NAJNIZEJ POLOZONY KLASTER ZAWIERA SAMOCHODY O DUZYM ZASIEGU I MALEJ MOCY, NATOMIAST NAJWYZSZY KLASTER MA SAMOCHODY O DUZEJ MOCY I MALYM ZASIEGU