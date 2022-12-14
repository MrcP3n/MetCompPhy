import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import reco

def array_hit(tab):
    aa = np.empty(0)
    time = tab['hit_time'].values
    mod = tab['mod_id'].values
    sens = tab['det_id'].values
    for i in range(len(time)):
        aa = np.append(aa, reco.Hit(mod[i], sens[i], time[i]))
    return aa



### Eventi del modulo0
tab = pd.read_csv('/home/mp000058/get-mcf-data/hit_times_M0.csv')
#print(tab)

time = tab['hit_time'].values
plt.hist(time, bins=80, color='darkgreen')
plt.show()
time_diff = np.diff(time)
mask = time_diff > 0
plt.hist(np.log10(time_diff[mask]), bins=70, color='darkred')
plt.title('hit provenienti dal modulo 0')
plt.savefig('modulo0eventi.png')
#plt.show()

  
### Eventi di tutti i moduli

a0 = array_hit(tab) #array_hit(pd.read_csv('hit_times_M0.csv'))
a1 = array_hit(pd.read_csv('/home/mp000058/get-mcf-data/hit_times_M1.csv'))
a2 = array_hit(pd.read_csv('/home/mp000058/get-mcf-data/hit_times_M2.csv'))
a3 = array_hit(pd.read_csv('/home/mp000058/get-mcf-data/hit_times_M3.csv'))

aaa = np.empty(0)
aaa = np.concatenate((a0, a1, a2, a3))
#print(aaa.shape)
aaa = np.sort(aaa)
aaa_tdiff = np.diff(aaa)

mask = aaa_tdiff > 0
aaa_dtime = np.zeros(len(aaa_tdiff[mask]))

for i in range(len(aaa_tdiff[mask])):
    aaa_dtime[i] = np.log10((aaa_tdiff[mask])[i])

plt.hist(aaa_dtime, bins=30, color='darkcyan')
plt.title('hit provenienti da tutti i moduli')
plt.savefig('Modulo0123Eventi.png')
#plt.show()

#differenze temporali tra hit determinano in che evento sono
#se ho piccolissimo deltat sono in stesso evento senno' sono passato ad un altro evento

'''def array_hit(tab):
    aa = np.empty(0)
    time = tab['hit_time'].values
    mod = tab['mod_id'].values
    sens = tab['det_id'].values
    for i in range(len(time)):
        aa = np.append(aa, reco.Hit(mod[i], sens[i], time[i]))
    return aa
'''

#treshold quanto deve essere grande per far funzionare?
#treshold necessario per capire quando cambiare evento

def arrayEvent(arrHit,treshold):
    #usa funzione che aggiunge hit
    #costruttore basta per creare evento
    #devo decidere quando cambiare evento ma l' aggiunta di hit ci deve essere
    #quindi devo implementare una condizione e come aggiungo hit al giusto evento e passare a creazione e riempimento di evento

    #aggiusta nomi capendo meglio concetto del codice senza speranza
    
    events=np.empty(0)  #arrayvuoto
    for hit in arrayHit:
        if(tempoHit)-(tempoHitPrecedente)>treshold):
            events=np.append(events,)
            
        events[-1].addHit(hit)
        tempo hit precedente)=hit.time

     return events   

 #un solo hit evento allora evento ha durata temporale di  =0
 #numero di hit per eventi si accumula a 20 perche' ci sarebbe la coda(se avessi piu' di 20 sensori potrebbe continuare)








