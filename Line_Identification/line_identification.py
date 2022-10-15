import pandas as pd
from lineid_module import line_identification

LHS72 = pd.read_csv('../Data/OBSERVED/Processed/norm_RVcorr_LHS72.txt',names=['wave','flux'],delim_whitespace=True)
LHS73 = pd.read_csv('../Data/OBSERVED/Processed/norm_RVcorr_LHS73.txt',names=['wave','flux'],delim_whitespace=True)

LHS72_lines = line_identification(LHS72,flux_threshold=0.2)
LHS72_lines.dropna(inplace=True)

LHS73_lines = line_identification(LHS73,flux_threshold=0.2)
LHS73_lines.dropna(inplace=True)

LHS72_lines.to_csv('LHS72_linelist.csv',sep=',',index=None)
LHS73_lines.to_csv('LHS73_linelist.csv',sep=',',index=None)