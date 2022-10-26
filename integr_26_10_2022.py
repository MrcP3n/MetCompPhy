import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
from scipy import optimize
import math

tab_v_t= pd.read_csv('/home/mp000058/Scaricati/metodi-computazionali/esercitazione4/vel_vs_time.csv')

ax=tab_v_t['t']
ay=tab_v_t['v']
plt.plot(ax,ay,color='midnightblue')
plt.xlabel('Tempo')
plt.ylabel('Velocita')

print('Distanza in funzione del tempo', integrate.simpson(ay,ax))

plt.show()

x=np.ones(len(ax))
for i in range (1,len(ax)):
    x[i]=integrate.simpson(ay[:i],ax[:i])

print(i,x)

plt.plot(ax,x,color='pink')
plt.xlabel('Tempo')
plt.ylabel('Posizione')

plt.show()




