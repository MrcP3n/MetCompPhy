import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import pandas as pd

def EqVout(Vout,t):
    vin=(int((t+1))%2*2)-1
    imp=np.array([1,0.1,0.01])
    return (vin-Vout)*(1/imp[0])

def EqVout1(Vout,t):
    vin=(int((t+1))%2*2)-1
    imp=np.array([1,0.1,0.01])
    return (vin-Vout)*(1/imp[1])

def EqVout2(Vout,t):
    vin=(int((t+1))%2*2)-1
    imp=np.array([1,0.1,0.01])
    return (vin-Vout)*(1/imp[2])

V0=0
a=0
b=10
num=10000
h = (b-a)/num 
tt = np.arange(a,b,h)
vin=((tt.astype(int)+1)%2)*2-1
vv0 = np.empty(0)
V = V0
 # Calcolo solzuine tramite scipy.integrate.odeint
vv0 = integrate.odeint(EqVout, y0=V0 , t=tt) #args=(vin)

plt.title('scipy.integrate.odeint ', color='slategray', fontsize=14)
plt.plot(tt,vin,color='blue')
plt.plot(tt,vv0,color='salmon')
plt.xlabel('t')
plt.ylabel('V')
plt.savefig('Circuito_eqdiff0.png')
plt.show()

vv1= np.empty(0)
vv1 = integrate.odeint(EqVout1, y0=V0 , t=tt) #args=(vin)

plt.title('scipy.integrate.odeint ', color='slategray', fontsize=14)
plt.plot(tt,vin,color='blue')
plt.plot(tt,vv1,color='salmon')
plt.xlabel('t')
plt.ylabel('V')
plt.savefig('Circuito_eqdiff1.png')
plt.show()

vv2 = np.empty(0)
V = V0
 # Calcolo solzuine tramite scipy.integrate.odeint
vv2 = integrate.odeint(EqVout2, y0=V0 , t=tt) #args=(vin)

plt.title('scipy.integrate.odeint ', color='slategray', fontsize=14)
plt.plot(tt,vin,color='blue')
plt.plot(tt,vv2,color='salmon')
plt.xlabel('t')
plt.ylabel('V')
plt.savefig('Circuito_eqdiff2.png')
plt.show()

mydf=pd.DataFrame(columns=['Tempi', 'Vout (RC=1)', 'Vout (RC=0.1) ','Vout (RC=0.01)'])
mydf['Tempi']=tt
mydf['Vout (RC=1)']=vv0
mydf['Vout (RC=0.1)']=vv1
mydf['Vout (RC=0.01)']=vv2
print(mydf)



mydf.to_csv('Circuito_eqdiff.csv')


