"""
Created on Tue Dec  3 11:05:40 2019

@author: k.kaminski
"""

# ZADANIE 1
import pandas as pd
import requests

# DICTIONARY OF LIST
def data_download(start_date, end_date, currency):
    request_url = 'http://api.nbp.pl/api/exchangerates/rates/A/' + currency + '/' + start_date + '/' + end_date + '/'
    currency_req = requests.get(request_url)
    currency_data = currency_req.json()
    return currency_data['rates'] 

# POBIERANIE DANYCH JAKO DATAFRAME
def data_download_as_dataframe(start_date, end_date, currency):
    currency_data = data_download(start_date, end_date, currency)
    return pd.DataFrame.from_dict(currency_data)

# ZADANIE 2
# POBIERANIE DANYCH O DOLARZE Z LISTOPADA
usd = data_download_as_dataframe('2019-11-01', '2019-11-30', 'USD')
# POBIERNIE DANYCH O EURO Z LISTOPADA
eur = data_download_as_dataframe('2019-11-01', '2019-11-30', 'EUR')

# ZADANIE 3
# DOLAR
print("ZAWARTOSC DATAFRAME")
print(usd.head()) # PODGLAD DATAFRAME
print("TYP DANYCH")
print(usd.dtypes) # TYP POBRANYCH DANYCH
print("ZMIANA NA DATATIME")
usd['effectiveDate'] = pd.to_datetime(usd['effectiveDate']) # ZMIANA DANYCH NA TYP DATATIME
print("SPRAWDZENIE NOWEGO TYPU DANYCH")
print(usd['effectiveDate'].dtypes) # SPRAWDZENIE TYPU DANYCH PO ZMIANIE
usd = usd.set_index("effectiveDate").drop(columns='no') # ZMIANA INDEKSU NA DATĘ I USUNIĘCIE KOLUMNY "no"
print(usd.head())

# EURO
print("ZAWARTOSC DATAFRAME")
print(eur.head()) # PODGLAD DATAFRAME
print("TYP DANYCH")
print(eur.dtypes) # TYP POBRANYCH DANYCH
print("ZMIANA NA DATATIME")
eur['effectiveDate'] = pd.to_datetime(eur['effectiveDate']) # ZMIANA DANYCH NA TYP DATATIME
print("SPRAWDZENIE NOWEGO TYPU DANYCH")
print(eur['effectiveDate'].dtypes) # SPRAWDZENIE TYPU DANYCH PO ZMIANIE
eur = eur.set_index("effectiveDate").drop(columns='no') # ZMIANA INDEKSU NA DATĘ I USUNIĘCIE KOLUMNY "no"
print(eur.head())

# ZADANIE 4
from numpy import corrcoef, array

usd = data_download('2019-11-01', '2019-11-30', 'USD')  
eur = data_download('2019-11-01', '2019-11-30', 'EUR')

# LISTA WARTOSCI W STOSTUNKU DO ZLOTOWKI 
usd2 = []
for tmp in usd:
    usd2.append(tmp['mid'])

eur2 = []
for tmp in eur:
    eur2.append(tmp['mid'])

# TABLICA KORELACJI
corrcoef(array(usd2), array(eur2))

# ZADANIE 5
import matplotlib.pyplot as plt

dane_wykres_usd = data_download_as_dataframe('2019-11-01', '2019-11-30', 'USD').set_index(['effectiveDate'])['mid']
dane_wykres_eur = data_download_as_dataframe('2019-11-01', '2019-11-30', 'EUR').set_index(['effectiveDate'])['mid']

fig, axs = plt.subplots(1,2, sharex=True, sharey=True) 
axs[0].plot(dane_wykres_usd) # PRZYPISANIE DANYCH DOLARA DO WYKRESU PIERWSZEGO
axs[1].plot(dane_wykres_eur) # PRZYPISANIE DANYCH EURO DO WYKRESU DRUGIEGO

# ZWIĘKSZENIE CZYTELNOCI WYKRESOW
fig.autofmt_xdate() # USTAWIENIE OPISOW OSI X
tmp_x=9 # ILOSC DANYCH WYSWIETLANYCH NA OSI X
axs[0].xaxis.set_major_locator(plt.MaxNLocator(tmp_x)) # OGRANICZENIE WYSWIETLANYCH DANYCH
plt.show()