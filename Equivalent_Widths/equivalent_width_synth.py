import pandas as pd
import numpy as np

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from lmfit.models import GaussianModel, VoigtModel, LorentzianModel, ConstantModel

def equivalent_width(wave, flux):
    a = np.insert(wave, 0, 2*wave[0] - wave[1])
    b = np.append(wave, 2*wave[-1] - wave[-2])
    edges = (a + b) / 2
    
    dx = np.abs(np.diff(edges))
    ew = np.sum((1-flux)*dx)
    return ew

def ew_synth(spec_df,obs_lines):
    """
    Parameters
    ----------
    spec_df : pandas Dataframe
        Synthetic spectrum.
    obs_lines : pandas Datframe
        Lines of observed spectrum for which EWs are needed.

    Returns
    -------
    synth_ews : pandas Dataframe
        Equivalent widths of synthetic spectrum..

    """
    ew_list = []
    
    for i in range(len(obs_lines)):
        line = obs_lines['Closest_NIST_Wavelength_(A)'][i]
        blue_end = obs_lines['Blue_end'][i]
        red_end = obs_lines['Red_end'][i]
        
        df1 = spec_df[(spec_df.wave >= blue_end) & (spec_df.wave <= red_end)]
        df1.reset_index(inplace=True,drop=True)
            
        wave = np.array(df1['wave'])
        flux = np.array(df1['flux'])
        
        gflux = gaussian_filter1d(flux,2)
        peak_indices = find_peaks(-gflux,height=-0.8)[0]

        if len(peak_indices) == 0:
            #print(f'{line} line not detected in spectrum.')
            ew_list.append(100)
            continue
    
        model = ConstantModel()
        pars = model.make_params()
        pars['c'].set(value=1)
    
        if obs_lines['Profile'][i] == 'Gaussian':
            profile = GaussianModel
        elif obs_lines['Profile'][i] == 'Voigt':
            profile = VoigtModel
        elif obs_lines['Profile'][i] == 'Lorentzian':
            profile = LorentzianModel
        
        for peak_index in range(len(peak_indices)):
            prefix = chr(ord('a') + peak_index)
            model_add = profile(prefix=f'{prefix}_')
            model += model_add
            pars.update(model_add.make_params())
            pars[f'{prefix}_center'].set(value=wave[peak_indices[peak_index]])
            pars[f'{prefix}_sigma'].set(value=0.1)  
            pars[f'{prefix}_amplitude'].set(value=0.3*(flux[peak_indices[peak_index]]-1))  
    
        out = model.fit(flux, pars, x=wave, max_nfev=1000)
        centers = []
        
        for i in out.best_values:
            if i[-6:] == 'center':
                centers.append(out.best_values[i])
    
        index = len(centers)-min(range(len(centers)), key=lambda i: abs(centers[i]-line))
        pfix = chr(ord('a') + index - 1)
        
        comps = out.eval_components(x=np.array(df1['wave']))
                    
        ew = equivalent_width(np.array(df1['wave']),1+comps[f'{pfix}_'])
        ew_list.append(ew)
        
    synth_ews = pd.DataFrame()
    synth_ews['Closest_NIST_Wavelength_(A)'] = obs_lines['Closest_NIST_Wavelength_(A)']
    synth_ews['EW'] = ew_list
        
    return synth_ews
    

