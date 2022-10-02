import os
import numpy as np
import pandas as pd

conv_synth_dir = '../Data/SYNTHETIC/Convolved_Abun'

spectra_df = pd.DataFrame()
spectra_df['Spectrum'] = []
spectra_df['Logg'] = []
spectra_df['Ion'] = []
spectra_df['Abundance'] = []

for file in os.listdir(f'{conv_synth_dir}'):
     synthetic_file = os.fsdecode(file)
     
     if synthetic_file.endswith(".txt"): 
         
        df = pd.DataFrame()
        df['Spectrum'] = [synthetic_file]
        df['Logg'] = float(synthetic_file.split('_')[4].partition('lte')[2].split('-')[1])
        df['Ion'] = synthetic_file.split('_')[2]
        df['Abundance'] = float(synthetic_file.split('_')[3])/100
        spectra_df = pd.concat([spectra_df,df],ignore_index=True)
        
def abun_interpolate(logg,ion,abundance,gaprange = False, telluric_ranges = False):
    '''

    Parameters
    ----------
    logg : float
        Log of surface gravity.
    ion : string
        Name of ion.
    abundance : float
        Abundance.
    gaprange : list, optional
        The gaprange of the observed spectrum. The default is False.
    telluric_ranges : list, optional
        Ranges of Telluric lines. The default is False.

    Raises
    ------
    ValueError
        If abundance does not belong to grid.

    Returns
    -------
    pandas DataFrame
        Returns interpolated spectrum.

    '''
    df = spectra_df[(spectra_df.Ion == ion) & (spectra_df.Logg == logg)]
    df.reset_index(inplace=True,drop=True)
    
    if df[df.Abundance <= abundance].empty or df[df.Abundance >= abundance].empty:
        la = df['Abundance'].min()
        ua = df['Abundance'].max()
        raise ValueError(f'\nAbundance {abundance} is not inside the grid. Please enter a value between {la} and {ua}.')
    
    if abundance in np.array(df['Abundance']):
        file_available = df[df.Abundance == abundance].reset_index(drop=True)['Spectrum'][0]
        df_available = pd.read_csv(f'{conv_synth_dir}/{file_available}',names=['wave','flux'],delim_whitespace=True)
        
        if bool(gaprange):
            df_available = df_available[(df_available['wave']<gaprange[0]) | (df_available['wave']>gaprange[1])]
            df_available.reset_index(inplace = True,drop=True)

        if bool(telluric_ranges):
            for i in range(len(telluric_ranges)):
                df_available = df_available[(df_available['wave']<telluric_ranges[i][0]) | (df_available['wave']>telluric_ranges[i][1])]
            df_available.reset_index(inplace = True,drop=True)
                                   
        return df_available
    
    lower_abundance = df[df.Abundance <= abundance].sort_values('Abundance',ascending=False).reset_index(drop=True)['Abundance'][0]
    upper_abundance = df[df.Abundance >= abundance].sort_values('Abundance').reset_index(drop=True)['Abundance'][0]
    
    lower_file = df[df.Abundance == lower_abundance].reset_index(drop=True)['Spectrum'][0]
    upper_file = df[df.Abundance == upper_abundance].reset_index(drop=True)['Spectrum'][0]
    
    lower_df = pd.read_csv(f'{conv_synth_dir}/{lower_file}',names=['wave','flux'],delim_whitespace=True)
    
    if bool(gaprange):
        lower_df = lower_df[(lower_df['wave']<gaprange[0]) | (lower_df['wave']>gaprange[1])]
        lower_df.reset_index(inplace = True,drop=True)

    if bool(telluric_ranges):
        for i in range(len(telluric_ranges)):
            lower_df = lower_df[(lower_df['wave']<telluric_ranges[i][0]) | (lower_df['wave']>telluric_ranges[i][1])]
        lower_df.reset_index(inplace = True,drop=True)  
    
    wave = np.array(lower_df['wave'])
    
    pre_upper_df = pd.read_csv(f'{conv_synth_dir}/{upper_file}',names=['wave','flux'],delim_whitespace=True)
    upper_df = pd.DataFrame()
    upper_df['wave'] = wave
    upper_df['flux'] = np.interp(wave,pre_upper_df['wave'],pre_upper_df['flux'])
    
    flux_list = []
    for i in range(len(wave)):
        flux_lower = lower_df['flux'][i]
        flux_upper = upper_df['flux'][i]
        flux = np.interp(abundance,[lower_abundance,upper_abundance],[flux_lower,flux_upper])
        flux_list.append(flux)
        
    spec_df = pd.DataFrame()
    spec_df['wave'] = wave
    spec_df['flux'] = flux_list
    
    return spec_df                               
