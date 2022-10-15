import pandas as pd
from line_Identification import line_identification

LHS72 = pd.read_csv('../Data/OBSERVED/norm_RVcorr_LHS72.txt',names=['wave','flux'],delim_whitespace=True)
LHS73 = pd.read_csv('../Data/OBSERVED/norm_RVcorr_LHS73.txt',names=['wave','flux'],delim_whitespace=True)