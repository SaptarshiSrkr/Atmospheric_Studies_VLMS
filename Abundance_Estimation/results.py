import emcee
import numpy as np

spec_name = 'LHS72'
ions_list = ['Ca','Fe','Ti','Na']
err_list = []
median_list = []

for ion in ions_list:
    backend = emcee.backends.HDFBackend(f"logfile_{ion}_norm_RVcorr_{spec_name}.txt.h5")
    abundances = backend.get_chain(flat=True)
    
    error = np.std(abundances)
    median = np.median(abundances)
    
    err_list.append(error)
    median_list.append(median)
    
text_file = open(f'Results_{spec_name}.txt', "w")

for i in range(len(ions_list)):
    text_file.write(f"Abundance of {ions_list[i]}: {median_list[i]:.2f} +/- {err_list[i]:.2f}\n")

text_file.close()