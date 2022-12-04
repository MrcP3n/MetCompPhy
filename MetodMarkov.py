import numpy as np
from scipy.optimize import minimize
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import emcee
import corner
import time
import tkinter

#b dove sbatte retta ovvero tipo la q , m pendenza , alpha altezza campana , sigma la larghezza , mu media 

params=(-0.18,10,-5,5,0.8)
def funz(p, E):
    m, b, alpha, mu, sigma = p
    return ((m*E)+b+(alpha*np.exp(-((E-mu)**2)/(2*sigma**2) ) ))

### Tab
#tab = pd.read_csv('/home/mp???/Scaricati/get-mcf-data/get_data.py/absorption_line.csv')
#tab = pd.read_csv('/home/ubuntu/absorption_line.csv')
tab = pd.read_csv('/home/marco/Scaricati/get-mcf-data/absorption_line.csv')
print(tab)

energy = tab['E'].values
flow = tab['f'].values
ferr = tab['ferr'].values

plt.errorbar(energy, flow, yerr=ferr, color='salmon')
#plt.legend(fontsize=13)
plt.xlabel('Energia')
plt.ylabel('Flusso')
plt.title('dati csv', fontsize = 12)
#plt.xscale('log')
#plt.yscale('log')
#plt.savefig('daticsv.png')
plt.show()

def lnlike_acc(p, E, f, ferr):
    return -0.5 * np.sum(((f - funz(p,E)  )/ferr) ** 2) 

def lnprior_acc(p):
    m, b, alpha, mu, sigma = p
    if ( -1 < m < 1 and  5 < b < 15 and -10 <alpha<0 and 0 <mu< 10 and 0<sigma<10 ):
        return 0.0
    return -np.inf


# logaritmo della distribuzione di probabilità totale 
# log(prob) = log(prior) + log(likelihood)
def lnprob_acc(p, E, f, ferr):
    lp = lnprior_acc(p)
    
    if np.isfinite(lp):
        return lp + lnlike_acc(p, E, f, ferr) 
    
    return -np.inf


# numero di walker
nw = 30

# condizioni iniziali 
initial_acc = params
ndim_acc = len(initial_acc)

# definisco parametri iniziali per i walker come piccola variazione random attorno
#  ai paramtri iniziali stabiliti 
p0 = np.array(initial_acc)  +0.1*np.random.randn(nw, ndim_acc)

# definisco il sampler di emcee
sampler_acc = emcee.EnsembleSampler(nw, ndim_acc, lnprob_acc, args=(energy, flow, ferr))

# Lancio campionamento per 2000 passi
print("Running production...")
sampler_acc.run_mcmc(p0, 2000, progress=True);


fig, axes = plt.subplots(ndim_acc, figsize=(10, 9), sharex=True)
samples_acc = sampler_acc.get_chain()

labels = ['m', 'b', 'alpha', 'mu', 'sigma' ]
for i in range(ndim_acc):
    ax = axes[i]
    ax.plot(samples_acc[:, :, i], color='red', alpha=0.3)
    ax.set_xlim(0, len(samples_acc))
    ax.set_ylabel(labels[i])
    #ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("numero passi");

plt.savefig('SampleParams.png')
plt.show()

# Escludo i primi 200 passi dalla valutazione 
flat_samples_acc = sampler_acc.get_chain(discard=200, thin=15, flat=True)


# Grafico dei dati e dei campionamenti escludendo i primi 200 passi
plt.errorbar(energy, flow,yerr=ferr, fmt="ok", capsize=0)
plt.xlabel('Energia ')
plt.ylabel('flusso')

# Plot 50 posterior samples.
for s in flat_samples_acc[np.random.randint( len(flat_samples_acc), size=50)]:
    plt.plot(energy, funz(s,energy), color="orange", alpha=0.3)
plt.show()


fig = corner.corner( flat_samples_acc, labels=labels , show_titles=True, color='orange');
plt.show()

