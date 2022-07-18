import os
import pandas as pd
from obs_modules import spec_normalize, rv_corr

waverange = [6460,9000]
gaprange = [8200,8390]

for file in os.listdir('../Data/OBSERVED/Raw/'):
    filename = os.fsdecode(file)
     
    if filename.endswith(".dat"): 
        
        print(f'\nNormalizing and RV correcting {filename}\n')
        df = pd.read_csv(f'../Data/OBSERVED/Raw/{filename}',names=['wave','flux'],delim_whitespace=True)
        df=df[(df['wave']>=waverange[0]-10) & (df['wave']<=waverange[1]+10)]
        df=df[(df['wave']<=gaprange[0]) | (df['wave']>=gaprange[1])]
        df.reset_index(inplace = True, drop = True)
        
        norm_df = spec_normalize(df)
        norm_df['flux'].fillna(0,inplace=True)
        
        norm_RVcorr_df = rv_corr(norm_df)
        norm_RVcorr_df=norm_RVcorr_df[(norm_RVcorr_df['wave']>=waverange[0]) & (norm_RVcorr_df['wave']<=waverange[1])]
        
        txtfile =  filename.split('.')[0]+'.txt'
        norm_RVcorr_df.to_csv(f'../Data/OBSERVED/Processed/norm_RVcorr_{txtfile}',sep='\t',index=False,header=False)

print('\nNormalized and RV corrected all observed spectra.')