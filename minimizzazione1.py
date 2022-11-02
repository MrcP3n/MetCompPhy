import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import pandas as pd
import math
from numpy import mean
df=pd.read_csv('/home/mp000058/Scaricati/metodi-computazionali/esercitazione4/fit_data.csv')

array_y=df['y']
array_x=df['x']
err_y=np.sqrt(array_y)

def gaussiana_lognorm(x,med,sigma,normalizzaz): # med,sigma,normalizz parametri liberi da dare nel fit
    log_x=np.zeros(len(x))
    for i in range(len(x)):
        log_x[i]=math.log(x[i])
    #med=np.mean(x) mi serve la funzione da fittare quindi non va bene che valori in coda vanno bene anzi non devono pesare uguali senno non distribuzione poissoniana inoltre non dipende dalle frequenze
    #sigma=np.std(x)
    gauss=np.zeros(len(x))
    for i in range(len(x)):
       # a=1/(x[i]*(sigma*math.sqrt(2*math.pi)))
        gauss[i]=(1/x[i])*normalizzaz*math.exp(-0.5*((log_x[i]-med)/sigma)**2)
    return gauss



plt.errorbar(array_x,array_y,yerr=err_y,color='midnightblue', fmt='o-')
plt.xlabel('Numero misure')
plt.ylabel('Misure')
plt.show()


#Calcolo e prelevo i parametri liberi
params,params_covariance = optimize.curve_fit(gaussiana_lognorm, array_x, array_y, sigma=err_y, absolute_sigma=True)

print('media_fit, sigma_fit, coefficente di parametrizzazione', params )
print('params_covariance /n', params_covariance)
print('errori params_covariance sono su diagonale',np.sqrt(params_covariance.diagonal()))


ytest = gaussiana_lognorm(array_x, params[0], params[1],params[2])

plt.plot(np.log(array_x), array_y,'o')
plt.plot(np.log(array_x), ytest, color='violet')

plt.show()

# Calcolo Chi quadrato
# Valore funzine fit ottimizzata in corrispondneza dei tempi dei dati


# chi2
chi2 =  np.sum( (ytest - array_y)**2 /ydata ) 

# gradi di libertà
ndof = len(array_x)-len(params)

print('Chi2', chi2, 'chi2 ridotto' , chi2/ndof )
