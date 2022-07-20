import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp2d, RegularGridInterpolator

conv_synth_dir = '../Data/SYNTHETIC/Convolved'

#Creating a dataframe of available spectra to use for interpolation

spectra_df = pd.DataFrame()
spectra_df['Spectrum'] = []
spectra_df['Teff'] = []
spectra_df['Logg'] = []
spectra_df['Metal'] = []

for file in os.listdir(f'{conv_synth_dir}'):
    synthetic_file = os.fsdecode(file)

    if synthetic_file.endswith(".txt"): 
         
        df = pd.DataFrame()
        df['Spectrum'] = [synthetic_file]
        vals_string = synthetic_file.partition('lte')[2]. partition ('a')[0]
        df['Teff'] = [float(vals_string[:-9])*100]
        df['Logg'] = [float(vals_string[-8:-4])]
        df['Metal'] = [float(vals_string[-4:])]
        spectra_df = pd.concat([spectra_df,df],ignore_index=True)
         
#Function that returns the required spectrum as a dataframe by interpolating from grid

def spec_interpolate(teff, logg, metal, gaprange = False, telluric_ranges = False):
    '''
    Parameters
    ----------
    teff : float
        Effective temperature in K.
    logg : float
        Log of surface gravity in cm/s^2.
    metal : float
        Metallicity in dex.
    gaprange : list, optional
        The gaprange of the observed spectrum. The default is False.
    telluric_ranges : list, optional
        Ranges of Telluric lines. The default is False.

    Raises
    ------
    ValueError
        If the and of values of the 3 parameters fall outside the available grid range.

    Returns
    -------
    TYPE
        The interpolated spectrum as a pandas DataFrame.

    '''
    
    #Check if the parameters are inside the grid. Raise a ValueError if not.
    
    if spectra_df[spectra_df.Teff <= teff].empty or spectra_df[(spectra_df.Teff >= teff)].empty:
        raise ValueError(f'\nTeff {teff} is not inside the grid.')
    if spectra_df[spectra_df.Metal <= metal].empty or spectra_df[spectra_df.Metal >= metal].empty:
        raise ValueError(f'\nMetallicity {metal} is not inside the grid.')
    if spectra_df[spectra_df.Logg <= logg].empty or spectra_df[spectra_df.Logg >= logg].empty:
        raise ValueError(f'\nLogg {logg} is not inside the grid.')
    
    #Check if the spectrum for the given parameters is already available in the grid.
    
    if teff in np.array(spectra_df['Teff']):
        if logg in np.array(spectra_df['Logg']):
            if metal in np.array(spectra_df['Metal']):
                file_available = spectra_df[(spectra_df.Teff == teff) & (spectra_df.Logg == logg) & (spectra_df.Metal == metal)].reset_index(drop=True)['Spectrum'][0]
                df_available = pd.read_csv(f'{conv_synth_dir}/{file_available}', names=['wave','flux'], delim_whitespace=True)
                
                #Remove unnecessary wavelenghts
                if bool(gaprange):
                    df_available = df_available[(df_available['wave']<gaprange[0]) | (df_available['wave']>gaprange[1])]
                    df_available.reset_index(inplace = True,drop=True)

                if bool(telluric_ranges):
                    for i in range(len(telluric_ranges)):
                        df_available = df_available[(df_available['wave']<telluric_ranges[i][0]) | (df_available['wave']>telluric_ranges[i][1])]
                    df_available.reset_index(inplace = True,drop=True)
                
                return df_available
                        
    #Find the parameters just below and above the given parameters.
    
    lower_teff = spectra_df[spectra_df.Teff <= teff].sort_values('Teff',ascending=False).reset_index(drop=True)['Teff'][0]
    upper_teff = spectra_df[spectra_df.Teff >= teff].sort_values('Teff').reset_index(drop=True)['Teff'][0]
    
    lower_logg = spectra_df[spectra_df.Logg <= logg].sort_values('Logg',ascending=False).reset_index(drop=True)['Logg'][0]
    upper_logg = spectra_df[spectra_df.Logg >= logg].sort_values('Logg').reset_index(drop=True)['Logg'][0]
    
    lower_metal = spectra_df[spectra_df.Metal <= metal].sort_values('Metal',ascending=False).reset_index(drop=True)['Metal'][0]
    upper_metal = spectra_df[spectra_df.Metal >= metal].sort_values('Metal').reset_index(drop=True)['Metal'][0]
       
    file_lll = spectra_df[(spectra_df.Teff == lower_teff) & (spectra_df.Logg == lower_logg) & (spectra_df.Metal == lower_metal)].reset_index(drop=True)['Spectrum'][0]
       
    df_lll = pd.read_csv(f'{conv_synth_dir}/{file_lll}', names=['wave','flux'], delim_whitespace=True)
     
    if bool(gaprange):
        df_lll = df_lll[(df_lll['wave']<gaprange[0]) | (df_lll['wave']>gaprange[1])]
        df_lll.reset_index(inplace = True,drop=True)

    if bool(telluric_ranges):
        for i in range(len(telluric_ranges)):
            df_lll = df_lll[(df_lll['wave']<telluric_ranges[i][0]) | (df_lll['wave']>telluric_ranges[i][1])]
        df_lll.reset_index(inplace = True,drop=True)        

    wave=np.array(df_lll['wave'])

    file_uuu = spectra_df[(spectra_df.Teff == upper_teff) & (spectra_df.Logg == upper_logg) & (spectra_df.Metal == upper_metal)].reset_index(drop=True)['Spectrum'][0]
    
    df = pd.read_csv(f'{conv_synth_dir}/{file_uuu}', names=['wave','flux'], delim_whitespace=True)
    df_uuu = pd.DataFrame()
    df_uuu['flux'] = np.interp(wave,np.array(df['wave']),np.array(df['flux']))
    df_uuu['wave'] = wave
                
    if lower_teff == upper_teff:
        if lower_logg == upper_logg:
            #Doing a 1D Linear Interpolation using [M/H] to generate the spectrum.
            flux_list = []
            for i in range(len(wave)):
                flux_lower = df_lll['flux'][i]
                flux_upper = df_uuu['flux'][i]
                flux = np.interp(metal,np.array([lower_metal,upper_metal]),np.array([flux_lower,flux_upper]))
                flux_list.append(flux)
                
            spec_df = pd.DataFrame()
            spec_df['wave'] = wave
            spec_df['flux'] = flux_list
            
            return spec_df
          
        elif lower_metal == upper_metal:
            #Doing a 1D Linear Interpolation using Logg to generate the spectrum.
            flux_list = []
            for i in range(len(wave)):
                flux_lower = df_lll['flux'][i]
                flux_upper = df_uuu['flux'][i]
                flux = np.interp(logg,np.array([lower_logg,upper_logg]),np.array([flux_lower,flux_upper]))
                flux_list.append(flux)
                
            spec_df = pd.DataFrame()
            spec_df['wave'] = wave
            spec_df['flux'] = flux_list
            
            return spec_df
        
    if lower_logg == upper_logg and lower_metal == upper_metal:    
        #Doing a 1D Linear Interpolation using Teff to generate the spectrum.
        flux_list = []
        for i in range(len(wave)):
            flux_lower = df_lll['flux'][i]
            flux_upper = df_uuu['flux'][i]
            flux = np.interp(teff,np.array([lower_teff,upper_teff]),np.array([flux_lower,flux_upper]))
            flux_list.append(flux)
            
        spec_df = pd.DataFrame()
        spec_df['wave'] = wave
        spec_df['flux'] = flux_list
        
        return spec_df
        
    file_llu = spectra_df[(spectra_df.Teff == lower_teff) & (spectra_df.Logg == lower_logg) & (spectra_df.Metal == upper_metal)].reset_index(drop=True)['Spectrum'][0]
    file_lul = spectra_df[(spectra_df.Teff == lower_teff) & (spectra_df.Logg == upper_logg) & (spectra_df.Metal == lower_metal)].reset_index(drop=True)['Spectrum'][0]
    file_luu = spectra_df[(spectra_df.Teff == lower_teff) & (spectra_df.Logg == upper_logg) & (spectra_df.Metal == upper_metal)].reset_index(drop=True)['Spectrum'][0]
    file_ull = spectra_df[(spectra_df.Teff == upper_teff) & (spectra_df.Logg == lower_logg) & (spectra_df.Metal == lower_metal)].reset_index(drop=True)['Spectrum'][0]
    file_ulu = spectra_df[(spectra_df.Teff == upper_teff) & (spectra_df.Logg == lower_logg) & (spectra_df.Metal == upper_metal)].reset_index(drop=True)['Spectrum'][0]
    file_uul = spectra_df[(spectra_df.Teff == upper_teff) & (spectra_df.Logg == upper_logg) & (spectra_df.Metal == lower_metal)].reset_index(drop=True)['Spectrum'][0]
    
    df = pd.read_csv(f'{conv_synth_dir}/{file_llu}', names=['wave','flux'], delim_whitespace=True)
    df_llu = pd.DataFrame()
    df_llu['flux'] = np.interp(wave,np.array(df['wave']),np.array(df['flux']))
    df_llu['wave'] = wave

    df = pd.read_csv(f'{conv_synth_dir}/{file_lul}', names=['wave','flux'], delim_whitespace=True)
    df_lul = pd.DataFrame()
    df_lul['flux'] = np.interp(wave,np.array(df['wave']),np.array(df['flux']))
    df_lul['wave'] = wave

    df = pd.read_csv(f'{conv_synth_dir}/{file_luu}', names=['wave','flux'], delim_whitespace=True)
    df_luu = pd.DataFrame()
    df_luu['flux'] = np.interp(wave,np.array(df['wave']),np.array(df['flux']))
    df_luu['wave'] = wave

    df = pd.read_csv(f'{conv_synth_dir}/{file_ull}', names=['wave','flux'], delim_whitespace=True)
    df_ull = pd.DataFrame()
    df_ull['flux'] = np.interp(wave,np.array(df['wave']),np.array(df['flux']))
    df_ull['wave'] = wave

    df = pd.read_csv(f'{conv_synth_dir}/{file_ulu}', names=['wave','flux'], delim_whitespace=True)
    df_ulu = pd.DataFrame()
    df_ulu['flux'] = np.interp(wave,np.array(df['wave']),np.array(df['flux']))
    df_ulu['wave'] = wave

    df = pd.read_csv(f'{conv_synth_dir}/{file_uul}', names=['wave','flux'], delim_whitespace=True)
    df_uul = pd.DataFrame()
    df_uul['flux'] = np.interp(wave,np.array(df['wave']),np.array(df['flux']))
    df_uul['wave'] = wave

    if lower_teff == upper_teff:
        #Doing a 2D Bilinear Interpolation using Logg and [M/H] to generate the spectrum.
        flux_list = []
        for i in range(len(wave)):
            flux_ll = df_lll['flux'][i]
            flux_lu = df_llu['flux'][i]
            flux_ul = df_lul['flux'][i]
            flux_uu = df_luu['flux'][i]
            
            x = [lower_logg,upper_logg] 
            y = [lower_metal,upper_metal]

            z = [[flux_ll,flux_ul], #z needs to have 'xy' type indexing 
                 [flux_lu,flux_uu]] #for correct interpolation.

            interpolator = interp2d(x,y,z,bounds_error=True)
            flux = float(interpolator(logg,metal))
            flux_list.append(flux)
            
        spec_df = pd.DataFrame()
        spec_df['wave'] = wave
        spec_df['flux'] = flux_list
        
        return spec_df
    
    if lower_logg == upper_logg:
        #Doing a 2D Bilinear Interpolation using Teff and [M/H] to generate the spectrum.
        flux_list = []
        for i in range(len(wave)):
            flux_ll = df_lll['flux'][i]
            flux_lu = df_llu['flux'][i]
            flux_ul = df_ull['flux'][i]
            flux_uu = df_ulu['flux'][i]
            
            x = [lower_teff,upper_teff]
            y = [lower_metal,upper_metal]
                       
            z = [[flux_ll,flux_ul], 
                 [flux_lu,flux_uu]] 

            interpolator = interp2d(x,y,z,bounds_error=True)
            flux = float(interpolator(teff,metal))
            flux_list.append(flux)
            
        spec_df = pd.DataFrame()
        spec_df['wave'] = wave
        spec_df['flux'] = flux_list
        
        return spec_df
    
    if lower_metal == upper_metal:
        #Doing a 2D Bilinear Interpolation using Teff and Logg to generate the spectrum.
        flux_list = []
        for i in range(len(wave)):
            flux_ll = df_lll['flux'][i]
            flux_lu = df_lul['flux'][i]
            flux_ul = df_ull['flux'][i]
            flux_uu = df_uul['flux'][i]
            x = [lower_teff,upper_teff]
            y = [lower_logg,upper_logg]

            z = [[flux_ll,flux_ul],
                 [flux_lu,flux_uu]]

            interpolator = interp2d(x,y,z,bounds_error=True)
            flux = float(interpolator(teff,logg))
            flux_list.append(flux)
            
        spec_df = pd.DataFrame()
        spec_df['wave'] = wave
        spec_df['flux'] = flux_list
        
        return spec_df
    
    #Doing a 3D Trilinear Interpolation to generate the spectrum.
    flux_list = []
    for i in range(len(wave)):
        flux_lll = df_lll['flux'][i]
        flux_llu = df_llu['flux'][i]
        flux_lul = df_lul['flux'][i]
        flux_luu = df_luu['flux'][i]
        flux_ull = df_ull['flux'][i]
        flux_ulu = df_ulu['flux'][i]
        flux_uul = df_uul['flux'][i]
        flux_uuu = df_uuu['flux'][i]
        
        x = np.array([lower_teff,upper_teff])
        y = np.array([lower_logg,upper_logg])
        z = np.array([lower_metal,upper_metal])
        points = (x,y,z)
        
        '''
        indexing must be of 'ij' kind for RegularGridInterpolator:

        array([[[(xl,yl,zl), (xl,yl,zu)],
                [(xl,yu,zl), (xl,yu,zu)]],
               [[(xu,yl,zl), (xu,yl,zu)],
                [(xu,yu,zl), (xu,yu,zu)]]])
        
        where l represents the lower value and u represents the upper value.
        '''
        values = np.array([[[flux_lll, flux_llu],
                            [flux_lul, flux_luu]],
                           [[flux_ull, flux_ulu],
                            [flux_uul, flux_uuu]]])
        
        point = np.array([teff,logg,metal])
        interpolator = RegularGridInterpolator(points,values,bounds_error=True)
        flux = float(interpolator(point))
        flux_list.append(flux)
        
    spec_df = pd.DataFrame()
    spec_df['wave'] = wave
    spec_df['flux'] = flux_list
    
    return spec_df
