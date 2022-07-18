import pandas as pd
import os

waverange = [6460,9000]

print('\nStarting process to reduce raw synthetic spectrum files.')

for file in os.listdir('../Data/SYNTHETIC/Raw/'):
    filename = os.fsdecode(file)
     
    if filename.endswith(".7"): 
         
        df = pd.read_csv(f'../Data/SYNTHETIC/Raw/{filename}',usecols=[0, 1],names=['wave','flux'],sep='\s+',low_memory=False)
        
        if df.dtypes.wave == 'object':
            df['wave']=df['wave'].str.replace('D','e')
            df['wave']=df['wave'].astype(float)

        df=df[(df['wave']>=waverange[0]) & (df['wave']<=waverange[1])]
        df.reset_index(inplace = True, drop = True)

        if df.dtypes.flux == 'object':
            df['flux']=df['flux'].str.replace('D','e')
            df['flux']=df['flux'].astype(float)

        df['flux']=10**(df['flux']-8)      
        
        df.to_csv(f'../Data/SYNTHETIC/Reduced/{filename}.txt',sep='\t',index=False,header=False)
                
        print(f'Spectrum {filename} reduced.')
        continue
    else:
        continue

print('\nProcess has been completed.')