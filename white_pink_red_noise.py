import numpy as np
import pandas as pd
from scipy import constants, fft
import matplotlib.pyplot as plt
import math
from scipy import optimize


mybomber1=pd.read_csv('/home/mp000058/Scaricati/get-mcf-data/get_data.py/data_sample1.csv')
mybomber2=pd.read_csv('/home/mp000058/Scaricati/get-mcf-data/get_data.py/data_sample2.csv')
mybomber3=pd.read_csv('/home/mp000058/Scaricati/get-mcf-data/get_data.py/data_sample3.csv')


mybomber1_time=mybomber1['time'].values
mybomber1_mis=mybomber1['meas'].values
mybomber2_time=mybomber2['time'].values
mybomber2_mis=mybomber2['meas'].values
mybomber3_time=mybomber3['time'].values
mybomber3_mis=mybomber3['meas'].values

plt.plot(mybomber1_time , mybomber1_mis , color='beige')
plt.title('Data sample 1', color='violet')
plt.xlabel('Time')
plt.ylabel('Misure')


plt.plot(mybomber2_time , mybomber2_mis , color='pink')
plt.title('Data sample 2', color='violet')
plt.xlabel('Time')
plt.ylabel('Misure')


plt.plot(mybomber3_time , mybomber3_mis , color='red')
plt.title('Data sample 3', color='violet')
plt.xlabel('Time')
plt.ylabel('Misure')
plt.show()

fftmis1=fft.fft(mybomber1_mis)
fftmis2=fft.fft(mybomber2_mis)
fftmis3=fft.fft(mybomber3_mis)

#[:array.size//2] fai fino a meta per prendere quei contributi che non si ripetono
plt.plot(np.absolute(fftmis1[:fftmis1.size//2])**2,'o',color='beige',markersize=3)

plt.plot(np.absolute(fftmis2[:fftmis2.size//2])**2,'o',color='pink',markersize=3)

plt.plot(np.absolute(fftmis3[:fftmis3.size//2])**2,'o',color='red',markersize=3)
plt.xlabel('Indice')
plt.ylabel('$|c_k|^2$')
plt.xscale('log')
plt.yscale('log')
plt.show()

snyq=0.5
d=1
noisef1= snyq*fft.fftfreq(fftmis1.size,d)
noisef2= snyq*fft.fftfreq(fftmis2.size,d)
noisef3= snyq*fft.fftfreq(fftmis3.size,d)

'''plt.plot(noisef1[:int(fftmis1.size/2)],np.absolute(fftmis1[:fftmis1.size//2])**2,'o',color='beige',markersize=3)
plt.plot(noisef2[:int(fftmis2.size/2)],np.absolute(fftmis2[:fftmis2.size//2])**2,'o',color='pink',markersize=3)
plt.plot(noisef3[:int(fftmis3.size/2)],np.absolute(fftmis3[:fftmis3.size//2])**2,'o',color='red',markersize=3)
plt.xlabel('Frequenza')
plt.ylabel('$|c_k|^2$')
plt.xscale('log')
plt.yscale('log')
plt.show()'''

def noise(f,a,B):
    lip=a*(1/f**B)
    return lip
pstart=np.array([0,1])
params1,params1_covariance = optimize.curve_fit(noise,noisef1[1:int(fftmis1.size/2)] , np.absolute(fftmis1[1:fftmis1.size//2])**2,p0=[pstart] )
noiseffit1=noise(noisef1[1:int(fftmis1.size/2)],params1[0],params1[1])
params2,params2_covariance = optimize.curve_fit(noise,noisef1[1:int(fftmis2.size/2)] , np.absolute(fftmis2[1:fftmis2.size//2])**2 ,p0=[pstart])
noiseffit2=noise(noisef2[1:int(fftmis2.size/2)],params2[0],params2[1])
params3,params3_covariance = optimize.curve_fit(noise,noisef1[5:int(fftmis3.size/2)] , np.absolute(fftmis3[5:fftmis3.size//2])**2 , p0=[pstart])
noiseffit3=noise(noisef3[5:int(fftmis3.size/2)],params3[0],params3[1])

plt.plot(noisef1[1:int(fftmis1.size/2)],np.absolute(fftmis1[1:fftmis1.size//2])**2,'o',color='beige',markersize=3)
plt.plot(noisef2[1:int(fftmis2.size/2)],np.absolute(fftmis2[1:fftmis2.size//2])**2,'o',color='pink',markersize=3)
plt.plot(noisef3[5:int(fftmis3.size/2)],np.absolute(fftmis3[5:fftmis3.size//2])**2,'o',color='red',markersize=3)
plt.plot(noisef1[1:int(fftmis1.size/2)],noiseffit1,'-',label='whitenoise',color='beige')
plt.plot(noisef2[1:int(fftmis2.size/2)],noiseffit2,'-',label='pinknoise',color='violet')
plt.plot(noisef3[5:int(fftmis3.size/2)],noiseffit3,'-',label='rednoise',color='darkred')
plt.title('Spettro di potenze e relativo Fit di frequenze dei vari rumori')
plt.xscale('log')
plt.yscale('log')
plt.savefig('white_pink_red_noise.png')
plt.show()



    









