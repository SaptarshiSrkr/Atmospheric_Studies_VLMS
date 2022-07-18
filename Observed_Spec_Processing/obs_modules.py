import pandas as pd
import numpy as np
from numpy import array, where, median, abs
from scipy.optimize import fmin
from scipy.ndimage import maximum_filter1d, median_filter, gaussian_filter1d

def spec_normalize(spec_df):
    '''
    Parameters
    ----------
    spec_df : Pandas DataFrame
        Pandas DataFrame containing the unnormalized spectrum. Must have columns 'wave' and 'flux'

    Returns
    -------
    norm_spec_df : Pandas DataFrame
        Pandas DataFrame containing the normalized spectrum.

    '''
    spec_df.drop_duplicates('wave',inplace=True,ignore_index=True)
    flux = np.array(spec_df['flux'])
    wave = np.array(spec_df['wave'])
    flux = flux/flux.mean() #Dividing all the flux values by the mean flux value for better accuracy.
    
    smooth_flux = flux
    for i in range(20): #Iterating to get a good continuum fit
        smooth_flux=maximum_filter1d(smooth_flux,int(30/(i+1)))
        smooth_flux=median_filter(smooth_flux,int(1000/(i+1))) #Required to conserve emission lines
        
    norm_spec_df = pd.DataFrame()
    norm_spec_df['wave'] = wave
    norm_spec_df['flux'] = flux/smooth_flux
    norm_spec_df['flux'].fillna(0,inplace=True)
    
    return norm_spec_df
    
    
def rv_corr(spec_df):
    template = pd.read_csv('conv1.34_norm_lte041-4.50-1.5a+0.4.BT-Settl.CIFIST2011_2017.spec.7.txt',names=['wave','flux'],delim_whitespace=True)
    gaprange = [8200,8390]
    telluric_ranges = [[6860, 6960],[7550, 7650],[8200, 8430]] 

    template = template[(template['wave']<gaprange[0]) | (template['wave']>gaprange[1])]

    for i in range(len(telluric_ranges)):
        template = template[(template['wave']<telluric_ranges[i][0]) | (template['wave']>telluric_ranges[i][1])]
    template.reset_index(inplace = True,drop=True)
    
    def minimize(v):
        temp_wave = np.array(template['wave'])
        temp_flux = np.array(template['flux'])

        spec_wave = np.array(spec_df['wave'])*np.sqrt((1-v)/(1+v))
        spec_flux1 = np.array(spec_df['flux'])
        spec_flux2 = np.interp(temp_wave,spec_wave,spec_flux1)
        spec_flux = gaussian_filter1d(spec_flux2,3)

        return np.sum((spec_flux-temp_flux)**2)

    rv = fmin(minimize,0)[0]
    c = 299792.458
    print(f'The radial velocity is {rv*c} km/s towards Earth.')
    spec_df['wave'] = np.array(spec_df['wave'])*np.sqrt((1-rv)/(1+rv))
       
    return spec_df
    