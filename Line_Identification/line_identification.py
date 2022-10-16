import pandas as pd
import numpy as np
from lineid_module import line_identification

#Get lines that are visible both in the observed and the synthetic spectrum

obsfile = "norm_RVcorr_LHS73.txt"
synhfile = "conv1.34_norm_lte041-4.50-1.5a+0.4.BT-Settl.CIFIST2011_2017.spec.7.txt"
ions_list = ['Ca','Fe','Ti','Na']

df_obs = pd.read_csv(f'../Data/OBSERVED/Processed/{obsfile}',names=['wave','flux'],delim_whitespace=True)
df_synth = pd.read_csv(f'../Data/SYNTHETIC/Convolved/{synhfile}',names=['wave','flux'],delim_whitespace=True)

dfobs_lines = line_identification(df_obs,ions_list,flux_threshold=0.2)
dfobs_lines.dropna(inplace=True)
dfobs_lines.reset_index(inplace=True,drop=True)

dfsynth_lines = line_identification(df_synth,ions_list,flux_threshold=0.2)
dfsynth_lines.dropna(inplace=True)
dfsynth_lines.reset_index(inplace=True,drop=True)

drop_indices = []
for i in range(len(dfobs_lines)):
    if dfobs_lines['Closest_NIST_Wavelength_(A)'][i] not in np.array(dfsynth_lines['Closest_NIST_Wavelength_(A)']):
        drop_indices.append(i)
        
dfobs_lines=dfobs_lines.drop(drop_indices)
dfobs_lines.to_csv(f'{obsfile}_linelist.csv',sep=',',index=None)