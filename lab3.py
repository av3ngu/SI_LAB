# ZADANIE 1
import matplotlib.pyplot as plt
import numpy as np
 
from sklearn.datasets import load_boston

# WCZYTANIE ZBIORU CECH NIERUCHOMOSCI ORAZ ICH CEN
boston_nieruchomosci = load_boston()

tax = boston_nieruchomosci['data'][:, np.newaxis, 9]
plt.scatter(tax, boston_nieruchomosci['target'])
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# STWORZENIE REGRESORA LINIOWEGO
linreg = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(tax, boston_nieruchomosci['target'], test_size = 0.3)

linreg.fit(X_train, y_train)

# PRZEWIDYWANIE CENY
y_pred = linreg.predict(X_test)

# METRYKA DOMYSLNA
print('METRYKA DOMYSLNA: ', linreg.score(X_test, y_test))

# WSPOLCZYNNIK REGRESJI
print('WSPOLCZYNNIK REGRESJI:\n', linreg.coef_)

# WYKRES REGRESJI - PRZEWIDYWANIE CEN MIESZKAN
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=2)
plt.show()

from sklearn.model_selection import cross_val_score
print("WALIDACJA KRZYZOWA")
cv_score_mse = cross_val_score(linreg, tax, boston_nieruchomosci.target, cv=5, scoring='neg_mean_squared_error')
print(cv_score_mse)