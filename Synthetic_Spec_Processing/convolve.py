import os
import pandas as pd
from scipy.ndimage import gaussian_filter1d

std_dev = 1.34
print('\nStarting to convolve spectra.')

for file in os.listdir(f'../Data/SYNTHETIC/Normalized'):
    filename = os.fsdecode(file)
     
    if filename.endswith(".txt"): 
        spec = pd.read_csv(f'../Data/SYNTHETIC/Normalized/{filename}', names=['wave','flux'], delim_whitespace=True)
        spec['flux'] = gaussian_filter1d(spec['flux'],std_dev)
        
        spec.to_csv(f'../Data/SYNTHETIC/Convolved/conv{std_dev:.2f}_{filename}',sep='\t',header=False,index=False)
        
        print(f'Convolved {filename} with a gaussian of stdev {std_dev:.2f}.')

print('\nConvolution completed.')
