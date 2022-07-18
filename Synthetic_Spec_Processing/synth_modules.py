import pandas as pd
import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter

def spec_normalize(spec_df):
    '''
    Parameters
    ----------
    spec_df : Pandas DataFrame
        Pandas DataFrame containing the unnormalised spectrum. Must have columns 'wave' and 'flux'

    Returns
    -------
    norm_spec_df : Pandas DataFrame
        Pandas DataFrame containing the normalised spectrum.

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
    
