import pandas as pd
from interpolation import spec_interpolate
from interpolation_abun import abun_interpolate

'''
teff = 3961
logg = 4.82
metal = -1.39

df = spec_interpolate(teff,logg,metal)
filename = f'conv1.34_norm_lte{float(teff/100):.1f}-{logg:.2f}{metal:.1f}a+X.X.BT-Settl.CIFIST2011_2017.spec.7.txt'
df.to_csv(filename,sep='\t',header=False,index=False)

'''
ion = 'Ca'
logg = 4.70
abun = 6.38

df = abun_interpolate(logg, ion, abun)
filename = f'conv1.34_norm_{ion}_{int(abun*100)}_lte040-{logg:.2f}-1.Xa+0.4.BT-Settl.CIFIST2011_2017.spec.7.txt'
df.to_csv(filename,sep='\t',header=False,index=False)
