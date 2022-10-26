import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
from scipy import optimize
import math


def potenziale(k,x):
    V=k*x**6
    return V

def periodo(m,x0,k):
    x=np.arange(0,x0,0.01)
    fv=1/(np.sqrt(potenziale(k,x0)-potenziale(k,x)))
    T=integrate.simpson(fv,x)*np.sqrt(8*m)
    return T

x0=np.arange(1,26,1)
Per=periodo(1.5,10,120)
Periodo=np.ones(len(x0))
j=0
for i in range(len(x0)):
    Periodo[j]=periodo(1.5,x0[j],120)
    j=j+1

    
print('\i',Per)
print(Periodo)

plt.plot(x0,Periodo,color='green')
plt.xlabel('X0')
plt.ylabel('Periodo')

plt.show()
