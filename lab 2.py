"""
Created on Tue Dec  3 17:24:20 2019

@author: k.kaminski
"""

from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

liczba_sasiadow_opis = 'LICZBA SASIADOW'
dokladnosci_scoringu_opis = 'DOKLADNOSC SCORINGU'

# PRZYKŁADOWY ZBIOR DANYCH

iris = datasets.load_iris()

# ZADANIE 1

print('Opis irysów w zbiorze to: ', iris['DESCR'])

# OPIS ZBIORU DANYCH:
# CHARAKTERYSTYKA TRZECH KLAS IRYSOW
# WYSOKOSC I SZEROKOSC
# KLASYFIKACJA
# BIBLIOGRAFIA


# ZADANIE 2

X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

liczby_sasiadow = [1, 2, 3, 4, 5 , 6, 7, 8, 9, 10, 11, 12, 13, 14]
dokladnosci_scoringu = []

for neighbours_count in liczby_sasiadow:
    # TWORZENIE KLASYFIKATORA
    knn = KNeighborsClassifier(neighbours_count)

    # NAUKA NA DANYCH
    knn.fit(X_train, y_train)

    # ZAPIS WYNIKU SCORINGU NA DANYCH TESTOWYCH DO OSOBNEJ LISTY
    dokladnosci_scoringu.append(knn.score(X_test, y_test))

print("LICZBA SASIADOW, DOKLADNOSC SCORINGU")

for liczba_sasiadow, dokladnosc_scoringu in zip(liczby_sasiadow, dokladnosci_scoringu):
    print([liczba_sasiadow, dokladnosc_scoringu])

# ZAKRES ZALEZNOSCI MIEDZY LICZBA SASIADOW A DOKLADNOSCIA
plt.plot(liczby_sasiadow, dokladnosci_scoringu)
plt.title('SCORING DLA ZMIENNEJ LICZBY SASIADOW ALGORYTMU KNN')
plt.xlabel(liczba_sasiadow_opis)
plt.ylabel(dokladnosci_scoringu_opis)

# ZADANIE 3

wines = datasets.load_wine()

# PODGLAD DANYCH W ZBIORZE
print('ELEMENTY ZBIORU WIN: ', list(wines.keys()))
# CECHY WIN
print('CECHY WIN W ZBIORZE TO: ', wines['feature_names'])

# KONWERSJA NA pandas.DataFrame
wines_df = pd.DataFrame(wines['data'], columns=wines['feature_names'])

# ZMIANA WARTOSCI NA PELNY OPIS TEKSTOWY DLA GATUNKU
targets = map(lambda x: wines['target_names'][x], wines['target'])

# DOKLEJANIE INFORMACJI O GATUNKU DO RESZTY DATAFRAME
wines_df['species'] = np.array(list(targets))

X = wines.data
y = wines.target

# PODZIAŁ NA ZBIOR UCZACY I TESTOWY
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8)

# TWORZENIE KLASYFIKATORA KNN UZYWAJAC PARAMTERU 5 SASIADOW
knn = KNeighborsClassifier(n_neighbors = 5)

# WYTRENUJ KLASYFIKATOR KNN
knn.fit(X_train, y_train)

# PREDYKCJA NA ZBIORZE TESTOWYM
y_pred = knn.predict(X_test)

# SPRAWDZENIE WARTOSCI PRZEWIDZIANYCH
print(["WARTOSCI PRZEWIDZIANE: ", y_pred[:5]])

# SPRAWDZENIE DOKLADNOSCI KLASYFIKATORA
print(["DOKLADNOSC KLASYFIKATORA: ", knn.score(X_test, y_test)])

# RAPORT Z UCZENIA confusion_matrix ORAZ classification_report
print()
print("RAPORT Z UCZENIA - classification_report")
print(classification_report(y_test, y_pred))

print()
print("RAPORT Z UCZENIA - confusion_matrix")
print("X - AKTUALNA CLASS")
print("Y - PRZEWIDZIANA CLASS")
print('KLASY WIN W ZBIORZE: ', wines['target_names'])
print(confusion_matrix(y_test, y_pred))

print("RAPORTY POKRYWAJA SIE Z RZECZYWISTOSCIA DLA class_0")
print("RAPORTY NIE POKRYWAJA SIE Z RZECZYWISTOSCIA DLA class_1, class_2")