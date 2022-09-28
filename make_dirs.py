import os

print('Creating required folders.\n')

newpath = f'Data' 
if not os.path.exists(newpath):
    os.makedirs(newpath)
    print(f'Created directory: Data\n')

paths = ['SYNTHETIC','OBSERVED']

for path in paths:
    newpath = f'Data/{path}' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        print(f'Created directory: Data/{path}\n')

synth_paths = ['Raw','Normalized','Convolved','Raw_Abun','Normalized_Abun','Convolved_Abun']

for synth_path in synth_paths:
    newpath = f'Data/SYNTHETIC/{synth_path}' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        print(f'Created directory: Data/SYNTHETIC/{synth_path}\n')

obs_paths = ['Raw','Processed']

for obs_path in obs_paths:
    newpath = f'Data/OBSERVED/{obs_path}' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        print(f'Created directory: Data/OBSERVED/{obs_path}\n')