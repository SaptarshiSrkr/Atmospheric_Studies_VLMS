import warnings
import re
import pandas as pd
import numpy as np

import astropy
from astropy import units as u
from astroquery.nist import Nist

from specutils import Spectrum1D
from specutils.fitting import find_lines_derivative

telluric_ranges = [[6860, 6960],[7550, 7650],[8200, 8430]]

'''
ions_list=['V','H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al',
           'Si','P','S','Cl','Ar','K','Ca','Sc','Ti','Cr','Mn','Fe','Co',
           'Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr',
           'Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','Xe',
           'Cs','Ba']
'''

ions_list=['Ca']

def line_identification(spec_df,flux_threshold=0.2,wavelength_type='vacuum',line_range=False,synthetic_spectrum=False):
    '''

    Parameters
    ----------
    spec_df : Pandas DataFrame
        The spectrum for which line identification has to be done. Must have columns 'wave' and 'flux'.
    flux_threshold : float, optional
        The threshold flux for identification. The default is 0.2.
    wavelength_type : either 'air+vacuum' or 'vacuum', optional
        The default is 'vacuum'.
    line_range : list, optional
        Range of wavelengths for which identification has to be done. The default is the entire spectrum.
    synthetic_spectrum : bool, optional
        Either 'True' or 'False'. Identification for telluric range is not done if False. The default is False.

    Returns
    -------
    df : Pandas DataFrame
        The line identification information.

    '''
    
    wave = np.array(spec_df['wave'])
    flux = np.array(spec_df['flux'])
    
    spec1d = Spectrum1D(spectral_axis=wave*u.AA , flux=flux*u.lm-1*u.lm)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lines = find_lines_derivative(spec1d, flux_threshold=flux_threshold)
    
    lines_list=[]
    for i in range(len(lines[lines['line_type'] == 'absorption'])):
        line=lines[lines['line_type'] == 'absorption'][i][0]
        lines_list.append(line.value)
        
    if bool(line_range) == True: 
        lines_list = [line for line in lines_list if line_range[0] < line < line_range[1]]
    
    if synthetic_spectrum == False:   
        for i in range(len(telluric_ranges)):
            lines_list = [line for line in lines_list if not telluric_ranges[i][0] < line < telluric_ranges[i][1]]  
            
    linename = ions_list[0]
    for ion in ions_list[1:]:
        linename = linename + " " + ion
    
    ions = []
    ion_lines = []

    for wl in lines_list:
    
        try:
    
            table = Nist.query((np.around(wl,2)-0.05) * u.AA, (np.around(wl,2)+0.05) * u.AA, linename=linename ,wavelength_type=wavelength_type)
    
            prob_ions = table['Spectrum']
    
            ritz = table['Ritz']
    
            if type(ritz)==astropy.table.column.MaskedColumn:
                ritz = ritz.filled(0)
    
            if type(ritz[0])==np.str_:
                ritz = [float(re.sub('[^\d\.]', '', wl)) for wl in ritz]
    
            observed = table['Observed']
    
            if type(observed)==astropy.table.column.MaskedColumn:
                observed = observed.filled(0)
    
            if type(observed[0])==np.str_:
                observed = [float(re.sub('[^\d\.]', '', wl)) for wl in observed]
    
            prob_lines = ritz
    
            for i in range(len(prob_lines)):
                if prob_lines[i]==0:
                    prob_lines[i] = observed[i]
    
            i_closest = min(range(len(prob_lines)), key = lambda i: abs(prob_lines[i]-wl))

            print(f'Line at {prob_lines[i_closest]} corresponds to {prob_ions[i_closest]} ion.')

            ion_lines.append(prob_lines[i_closest])
            ions.append(prob_ions[i_closest])

        except Exception as e: 
            
            #print(e)
            ion_lines.append(np.nan)
            ions.append(np.nan) 
            
    df = pd.DataFrame()        
    df['Absorption Lines (A)'] = lines_list       
    df['Closest NIST Wavelength (A)'] = ion_lines
    df['Corresponding Ion'] = ions
    
    return df