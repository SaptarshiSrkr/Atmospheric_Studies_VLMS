import os

cwd=os.getcwd()

paths = ['Normalized','Convolved']

print('\nCreating required folders.')

for path in paths:
    newpath = f'../Data/SYNTHETIC/{path}' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        print(f'Created folder: {path}')