import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from scipy import constants, fft
import scipy.optimize as opt

tab=pd.read_csv('/home/marco/Scaricati/get-mcf-data/copernicus_PG_selected.csv')

#due colonne del tempo prendo quella con numeri
tNum=(tab['date'].values)-58111
meanCo=tab['mean_co_ug/m3'].values
meanNh3=tab['mean_nh3_ug/m3'].values
meanNo2=tab['mean_no2_ug/m3'].values
meanO3=tab['mean_o3_ug/m3'].values
meanPm10=tab['mean_pm10_ug/m3'].values
meanPm2p5=tab['mean_pm2p5_ug/m3'].values
meanSo2=tab['mean_so2_ug/m3'].values

#Grafico con tutti gli inquinanti al variare del tempo
plt.plot(tNum, meanCo , color='green', label='CO' )
plt.plot(tNum, meanNh3 , color='orange', label='NH3')
plt.plot(tNum, meanNo2 , color='cyan', label='NO2' )
plt.plot(tNum, meanO3 , color='pink' , label='O3')
plt.plot(tNum, meanPm10 , color='black' , label='Pm10')
plt.plot(tNum, meanPm2p5 , color='violet' , label='Pm2P5')
plt.plot(tNum, meanSo2 , color='blue' , label='SO2')
plt.legend(loc='upper left')
plt.xlabel('Giorni')
plt.ylabel('Inquinanti')
plt.show()

#trasformata fourier CO e recupero frequenze

coFft=fft.fft(meanCo)
snyquist = 0.5
cof = snyquist*fft.fftfreq(len(coFft), d=1)
# Grafico Spettro di Potenza senza escludere il termine c(0) per f=0 e le frequenze negative il cui coefficiente c(-f)
    #  è il complesso coniugato del coefficiente c(f): c(-f) = c(f)*

    #  le frequenze sono ordinate secondo l'ordine [0-->fmax, -fmax, 0[
    #  per produrre un grafico corretto si possono riordinare le frequenze con fft.fftshift
# Grafico spetto di potenza in funzione delle frequenze

plt.plot(cof[:int(coFft.size/2)], np.absolute(coFft[:int(coFft.size/2)])**2, 'o', markersize=4)
plt.xlabel('Frequenza')
plt.ylabel('Spettro potenze in funzione di freq')
plt.xscale('log')
plt.yscale('log')
plt.show()

# Grafico spettro di potenza in funzione del periodo (1/freq)
plt.plot(1/cof[1:int(coFft.size/2)], np.absolute(coFft[1:int(coFft.size/2)])**2, 'o', markersize=4)
plt.xlabel('Periodo ')
plt.ylabel('Spettro')
plt.xscale('log')
plt.yscale('log')
plt.show()


# Applico maskera per filtrare frequenze meno imporattanti sulla base dei dati ovvero quelle con frequenze superiori a fcut in valore assoluto 
fftmask1 = np.absolute(cof)**2< 1e-2

filteredCoFft1 = coFft.copy()
filteredCoFft1[fftmask1] = 0

fftmask2 = np.absolute(cof)**2< 2e-2

filteredCoFft2 = coFft.copy()
filteredCoFft2[fftmask2] = 0



# Trasformata FFT inversa con coefficienti filtrati 
filteredCoFft1Inv = fft.ifft(filteredCoFft1, n=len(meanCo))
filteredCoFft2Inv = fft.ifft(filteredCoFft2, n=len(meanCo))

#controlla problemi sulle lunghezze e guarda meglio il filtro da mettere

print(len(meanCo), coFft.size, filteredCoFft1.size)

# Grafico dati originali e filtrati
plt.subplots(figsize=(11,7))
plt.plot(tNum, meanCo , color='gold',      label='Dati Originali')
plt.plot(tNum,filteredCoFft1Inv ,     color='magenta',   label='Filtro $P>1\cdot 10^7$')
plt.plot(tNum, filteredCoFft2Inv ,     color='blue',   label='Filtro $P>1\cdot 10^7$')
plt.legend(fontsize=13)
plt.xlabel('Days')
plt.ylabel('Densità CO')
plt.show()
