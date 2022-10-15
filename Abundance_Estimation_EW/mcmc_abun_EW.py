import pandas as pd
import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "..\Interpolation")

from interpolation_abun import abun_interpolate
import interpolation_abun
spectra_df = interpolation_abun.spectra_df

np.random.seed(20)

#########################################
#INPUTS
#########################################

obs_filename = 'norm_RVcorr_LHS72.txt'
logg = 4.7 #4.7 for LHS72, 4.8 for LHS73

ions_list = ['Ca','Fe','Ti','Na']

gaprange = [8200,8390]
telluric_ranges = [[6860, 6960],[7550, 7650],[8200, 8430]]
 
#########################################

obs_lines = pd.read_excel(f'../Equivalent_Widths/{obs_filename}_EWs.xlsx')

#Getting Equivalent Widths for Synthetic Spectrum

def equivalent_width_synth()