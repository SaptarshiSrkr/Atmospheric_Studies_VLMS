import pandas as pd
import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "..\Interpolation")
sys.path.insert(0, "..\Equivalent_Widths")

from equivalent_width_synth import ew_synth
from interpolation_abun import abun_interpolate
import interpolation_abun
spectra_df = interpolation_abun.spectra_df

np.random.seed(20)

#########################################
#INPUTS
#########################################

observed_file = 'norm_RVcorr_LHS73.txt'
logg = 4.8 #4.7 for LHS72, 4.8 for LHS73

ions_list = ['Ca','Ti','Fe','Na']

gaprange = [8200,8390]
telluric_ranges = [[6860, 6960],[7550, 7650],[8200, 8430]]
 
#########################################

obs_lines = pd.read_excel(f'../Equivalent_Widths/{observed_file}_EWs.xlsx')

def log_likelihood(theta, logg, ion, gaprange, telluric_ranges, obs_lines):
    abundance=theta[0]
    syn_spec = abun_interpolate(logg, ion, abundance, gaprange, telluric_ranges)
    obs_lines = obs_lines[obs_lines['Corresponding_Ion'].str.contains(f"{ion}")]
    obs_lines.reset_index(inplace=True,drop=True)
    ew_synth_df = ew_synth(syn_spec,obs_lines)
    syn_ews = np.array(ew_synth_df['EW'])
    obs_ews = np.array(obs_lines['EW'])
    err = np.array(obs_lines['EW Error'])
    return -0.5 * np.sum(np.log(2*np.pi*err**2) + (obs_ews - syn_ews)**2/err**2)

def log_prior(theta):
    abundance=theta[0]
    if abundance < abun_min or abundance > abun_max:
        return -np.inf
    else:
        return 0
    
def log_posterior(theta, logg, ion, gaprange, telluric_ranges, obs_lines):
    if np.isinf(log_prior(theta)):
        return log_prior(theta)
    else:
        return (log_likelihood(theta, logg, ion, gaprange, telluric_ranges, obs_lines) + log_prior(theta))
    
ndim = 1
nsteps = 200  
    
for ion in ions_list:

    abun_min = spectra_df[spectra_df.Ion == ion]['Abundance'].min()
    abun_max = spectra_df[spectra_df.Ion == ion]['Abundance'].max()
    
    starting_guesses = [[round(i,4)] for i in np.arange(abun_min,abun_max,0.025)]
    nwalkers = len(starting_guesses)                                      
    
    backend = emcee.backends.HDFBackend(f"logfile_{ion}_{observed_file}.h5")
    backend.reset(len(starting_guesses),ndim)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,args=(logg, ion, gaprange, telluric_ranges, obs_lines), backend=backend)
    coords, prob, state = sampler.run_mcmc(starting_guesses, nsteps, progress=True)
    
    plt.figure(figsize=(10,6))
    plt.plot(sampler.get_chain()[:, :, 0],'-k', alpha=0.5)
    plt.ylabel(f'{ion} Abundance',fontsize=15)
    plt.savefig(f'Chains_{ion}_{observed_file}_final.png',dpi=500)
        
    figure = corner.corner(sampler.get_chain(flat=True,discard=50),labels=[f'{ion} Abundance'],quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig(f'Corner_{ion}_{observed_file}.png',dpi=500)