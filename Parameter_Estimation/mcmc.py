import emcee
import corner
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "..\Interpolation")

from interpolation import spec_interpolate
np.random.seed(20)

import interpolation
spectra_df = interpolation.spectra_df

#########################################
#INPUTS
#########################################

observed_file = 'norm_RVcorr_LHS73.txt'

gaprange = [8200,8390]
telluric_ranges = [[6860, 6960],[7550, 7650],[8200, 8430]] 

#########################################

def snr_estimate(flux):
    flux = np.array(flux)
    flux = np.array(flux[np.where(flux != 0.0)])
    n = len(flux)      

    if (n>4):
        signal = np.median(flux)
        noise  = 0.6052697 * np.median(abs(2.0 * flux[2:n-2] - flux[0:n-4] - flux[4:n]))
        return float(signal / noise)  
    else:
        return 0.0
spec = pd.read_csv(f'../Data/OBSERVED/Processed/{observed_file}', names=['wave','flux'], delim_whitespace=True)

spec = spec[(spec['wave']<gaprange[0]) | (spec['wave']>gaprange[1])]

for i in range(len(telluric_ranges)):
    spec = spec[(spec['wave']<telluric_ranges[i][0]) | (spec['wave']>telluric_ranges[i][1])]
spec.reset_index(inplace = True,drop=True)

wave = np.array(spec['wave'])
flux = np.array(spec['flux'])

SNR = snr_estimate(flux)

def log_likelihood(theta, gaprange, telluric_ranges): #theta = [teff,logg, metallicity]
    syn_spec = spec_interpolate(theta[0], theta[1], theta[2], gaprange, telluric_ranges)
    syn_wave = np.array(syn_spec['wave'])
    syn_flux = np.array(syn_spec['flux'])
    isyn_flux = np.interp(wave,syn_wave,syn_flux)
    fluxerr = isyn_flux/SNR
    return -0.5 * np.sum(np.log(2*np.pi*fluxerr**2) + (flux - isyn_flux)**2/fluxerr**2)

def log_prior(theta):
    if np.any(theta < theta_min) or np.any(theta > theta_max):
        return -np.inf
    else:
        return 0

def log_posterior(theta, gaprange, telluric_ranges):
    if np.isinf(log_prior(theta)):
        return log_prior(theta)
    else:
        return log_likelihood(theta, gaprange, telluric_ranges)+log_prior(theta)
    
theta_min = [spectra_df['Teff'].min(),spectra_df['Logg'].min(),spectra_df['Metal'].min()]
theta_max = [spectra_df['Teff'].max(),spectra_df['Logg'].max(),spectra_df['Metal'].max()]
     
ndim = 3
nsteps = 500
 
backend = emcee.backends.HDFBackend(f"logfile_{observed_file}.h5")

#For the 1st run 
#***************************************** 
  
starting_guesses = []
for teff in np.arange(theta_min[0]+100,theta_max[0],100):
    for logg in np.arange(theta_min[1]+0.5,theta_max[1],0.5):
        for metal in np.arange(theta_min[2]+0.5,theta_max[2],0.5):
           starting_guesses.append([teff,round(logg,4),round(metal,4)])
           
backend.reset(len(starting_guesses),ndim)

#*****************************************  
#For subsequent runs (when logfile is present)
#*****************************************  
'''
starting_guesses = backend.get_chain()[-1]
'''
#*****************************************  

nwalkers = len(starting_guesses)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(gaprange, telluric_ranges), backend=backend)
coords, prob, state = sampler.run_mcmc(starting_guesses, nsteps, progress=True)

fig, ax = plt.subplots(3, sharex=True,figsize=(10,8))

for i in range(3):
    ax[i].plot(sampler.get_chain()[:, :, i],'-k', alpha=0.5)
    
ax[0].set_ylabel('Teff',fontsize=15)
ax[1].set_ylabel('log g',fontsize=15)
ax[2].set_ylabel('[M/H]',fontsize=15)
fig.savefig(f'Chains_{observed_file}_final.png',dpi=500)
    
figure = corner.corner(sampler.get_chain(flat=True),labels=['Teff', 'log g', '[M/H]'],quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12}, range=[(3800,4200),(4,5.5),(-2.5,0)])
plt.savefig(f'Corner_{observed_file}.png',dpi=500)