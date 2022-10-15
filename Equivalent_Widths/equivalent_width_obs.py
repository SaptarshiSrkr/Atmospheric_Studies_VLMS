import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from lmfit.models import GaussianModel, VoigtModel, LorentzianModel, ConstantModel

spec_file = 'norm_RVcorr_LHS73.txt'
lines_file = 'norm_RVcorr_LHS73.txt_linelist.csv'

spec_df = pd.read_csv(f'../Data/OBSERVED/Processed/{spec_file}',names=['wave','flux'],delim_whitespace=True)
lines_df = pd.read_csv(f'{lines_file}')
lines_EWs = lines_df.copy(deep=True)

wave = np.array(spec_df['wave'])
flux = np.array(spec_df['flux'])

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

snr = snr_estimate(flux)

def equivalent_width(wave, flux, cont, dcont, snr):
    a = np.insert(wave, 0, 2*wave[0] - wave[1])
    b = np.append(wave, 2*wave[-1] - wave[-2])
    edges = (a + b) / 2
    
    dx = np.abs(np.diff(edges))
    ew = np.sum((1-flux)*dx)
    
    dwave = dx[0]
    N = len(wave)
    ew_err = ew*dwave*(N*dcont/cont+N/snr)
    return ew, ew_err

ew_list = []
ew_err_list = []

for i in range(len(lines_df)):

    line = lines_df['Closest_NIST_Wavelength_(A)'][i]
    blue_end = lines_df['Blue_end'][i]
    red_end = lines_df['Red_end'][i]
    
    df1 = spec_df[(spec_df.wave >= blue_end) & (spec_df.wave <= red_end)]
    df1.reset_index(inplace=True,drop=True)
        
    wave = np.array(df1['wave'])
    flux = np.array(df1['flux'])
    
    gflux = gaussian_filter1d(flux,2)
    peak_indices = find_peaks(-gflux,height=-0.8)[0]

    if len(peak_indices) == 0:
        print(f'{line} line not detected in spectrum.')
        ew_list.append(np.nan)
        ew_err_list.append(np.nan)
        continue

    model = ConstantModel()
    pars = model.make_params()
    pars['c'].set(value=1)

    if lines_df['Profile'][i] == 'Gaussian':
        profile = GaussianModel
    elif lines_df['Profile'][i] == 'Voigt':
        profile = VoigtModel
    elif lines_df['Profile'][i] == 'Lorentzian':
        profile = LorentzianModel
    
    profilestr = lines_df['Profile'][i]
    
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
    
    cont = out.result.params['c'].value
    dcont = out.result.params['c'].stderr
    if dcont==None:
        dcont=0
        
    ew, ew_err = equivalent_width(np.array(df1['wave']),1+comps[f'{pfix}_'], cont, dcont, snr)
    
    center = out.best_values[f'{pfix}_center']
    
    ew_list.append(ew)
    ew_err_list.append(ew_err)
            
    plt.figure(figsize=(15,10))
    plt.fill([center-ew/2,center+ew/2,center+ew/2,center-ew/2],[0,0,1,1],'orange',alpha=0.2,label='EW Box')
    plt.plot(df1['wave'], df1['flux'], c='r', alpha=0.2,label='Spectrum')
    plt.scatter(df1['wave'], df1['flux'], c='r', alpha=0.2)
    plt.plot(df1['wave'], comps[f'{pfix}_']+comps['constant'],'k--', label=f'Component {pfix}: Best Fit', alpha=1)
    plt.scatter(wave, flux, c='r',label='Points to fit', alpha=1)
    #plt.scatter(wave[peak_indices],flux[peak_indices])
    #plt.plot(wave,out.init_fit,label='Initial Fit',alpha=0.2)
    #plt.plot(wave,np.ones_like(wave)*comps['constant'], label='Constant')

    for i in range(len(peak_indices)): 
        pfix_comp = chr(ord('a') + i)
        if pfix_comp != pfix:
            plt.plot(df1['wave'],comps[f'{pfix_comp}_'] + comps['constant'],'--',label=f'Component {pfix_comp}')
    
    plt.legend()
    plt.savefig(f'Fits/{line}_{spec_file}_{profilestr}.png',dpi=300,facecolor='white')
    plt.close()
  
lines_EWs['EW'] = ew_list
lines_EWs['EW Error'] = ew_err_list
lines_EWs.to_excel(f'{spec_file}_EWs.xlsx',index=False)