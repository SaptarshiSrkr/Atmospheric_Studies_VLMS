import pandas as pd
from interpolation import spec_interpolate

teff = 3904
logg = 4.80
metal = -2

df = spec_interpolate(teff,logg,metal)
filename = f'conv1.34_norm_lte{float(teff/100):.1f}-{logg:.2f}{metal:.1f}a+X.X.BT-Settl.CIFIST2011_2017.spec.7.txt'
df.to_csv(filename,sep='\t',header=False,index=False)