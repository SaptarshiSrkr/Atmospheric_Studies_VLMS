import emcee
import corner
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "..\Interpolation")
from interpolation_abun import abun_interpolate
np.random.seed(22)

observed_file = 'norm_RVcorr_LHS72.txt' 
logg = 4.7
ion = 'Ca'

#Logg = 4.7 for LHS72 and 4.8 for LHS73

abun_min = 6.00
abun_max = 6.64

gaprange = [8200,8390]
telluric_ranges = [[6860, 6960],[7550, 7650],[8200, 8430]] 

spec = pd.read_csv(f'../Data/OBSERVED/Processed/{observed_file}', names=['wave','flux'], delim_whitespace=True)

wave = np.array(spec['wave'])
flux = np.array(spec['flux'])

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
    
SNR = snr_estimate(flux)

def log_likelihood(theta, logg, ion, gaprange, telluric_ranges):
    abundance=theta[0]
    syn_spec = abun_interpolate(logg,ion,abundance, gaprange, telluric_ranges)
    syn_wave = np.array(syn_spec['wave'])
    syn_flux = np.array(syn_spec['flux'])
    isyn_flux = np.interp(wave,syn_wave,syn_flux)
    fluxerr = isyn_flux/SNR
    return -0.5 * np.sum(np.log(2*np.pi*fluxerr**2) + (flux - isyn_flux)**2/fluxerr**2)

def log_prior(theta):
    abundance=theta[0]
    if abundance < abun_min or abundance > abun_max:
        return -np.inf
    else:
        return 0

def log_posterior(theta, logg, ion, gaprange, telluric_ranges):
    if np.isinf(log_prior(theta)):
        return log_prior(theta)
    else:
        return (log_likelihood(theta, logg, ion, gaprange, telluric_ranges) + log_prior(theta))
 
starting_guesses = []
for abun in np.arange(abun_min,abun_max,0.025):
    starting_guesses.append([abun])
    
ndim = 1
nsteps = 500
nwalkers = len(starting_guesses)                                      

backend = emcee.backends.HDFBackend(f"logfile_{observed_file}.h5")
backend.reset(len(starting_guesses),ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,args=(logg, ion, gaprange, telluric_ranges), backend=backend)
coords, prob, state = sampler.run_mcmc(starting_guesses, nsteps, progress=True)

plt.figure(figsize=(10,6))
plt.plot(sampler.get_chain()[:, :, 0],'-k', alpha=0.5)
plt.ylabel(f'{ion} Abundance',fontsize=15)
plt.savefig(f'Chains_{observed_file}_final.png',dpi=500)
    
figure = corner.corner(sampler.get_chain(flat=True),labels=[f'{ion} Abundance'],quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12}, range=[(abun_min,abun_max)])
plt.savefig(f'Corner_{observed_file}.png',dpi=500)