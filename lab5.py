# ZADANIE 1
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

samochody = fetch_openml('cars1')

# PODZIAŁ ZBIORU NA CECHY I ETYKIETY
X = samochody.data
y = samochody.target

# INICJALIZACJA
pca = PCA()
pca.fit(X)

# DEKOMPOZYCJA PCA TWORZY N NOWYCH SZTUCZNYCH CECH, 
# KTORE PROBUJA NAJLEPIEJ ODZWIERCIEDLIC ZMIENNOSC ORYGINALNEGO ZBIORU
print("Liczba komponentów: ", pca.n_components_)

# WPŁYW ORYGINALNYCH CECH NA NOWE WYWNIOSKOWANE
print("Skład nowych cech:")
print(pca.components_)

# KTORE WYWNIOSKOWANE CECHY MAJA NAJWIEKSZY WPLYW NA OGOLNA ZMIENNOSC ZBIORU
print(pca.explained_variance_ratio_)

# TWORZENIE WYKRESU PRZY POMOCY SEABORN
import seaborn as sns

# KONWERSJA
samochody_df = pd.DataFrame(samochody['data'], columns=samochody['feature_names'])

# DOKLEJENIE INFORMACJI O GATUNKU DO RESZTY DATAFRAME
samochody_df['target'] = np.array(list(samochody['target']))

# WYKRES
sns.pairplot(samochody_df, hue='target')
plt.show()

# REDUKCJA ZBIORU CECH DO NAJLEPSZEJ
pca_limit = PCA(n_components = 1)

X_new = pca_limit.fit_transform(X)
X_new[:5]

# CECHY
print("LICZBA KOMPONENTOW: ", print(pca_limit.n_components_))

# WPLYW ORYGINALNYCH CECH NA WYWNIOSKOWANA CECHE
print("Skład nowej cechy:")
print(pca_limit.components_)

# BARDZO WYSOKA WYTLUMACZALNOSC NOWEJ CECHY
print(pca_limit.explained_variance_ratio_)
plt.scatter(X_new, y)
plt.show()