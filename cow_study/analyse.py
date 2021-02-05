import os
import sys
import glob
sys.path.append("/Users/matt/Scratch/stats/sweights")
from utils import plot_pull
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read(fname):
  if not os.path.exists(fname):
    return np.array( [np.nan]*8 )
  with open(fname,'r') as f:
    lines = f.readlines()
    Ns  = float(lines[0].split()[1])
    Nse = float(lines[0].split()[3])
    Nb  = float(lines[1].split()[1])
    Nbe = float(lines[1].split()[3])
    lf  = float(lines[2].split()[1])
    lc  = float(lines[2].split()[3])
    lr  = float(lines[2].split()[5])
    ee  = float(lines[3].split()[1])

    return Ns, Nse, Nb, Nbe, lf, lc, lr, ee

fprefix = sys.argv[1]
files = glob.glob(fprefix+"*.log")
print('Found', len(files), 'files')

res = np.array( [ read(f) for f in files ] )
df  = pd.DataFrame(res, columns=['Ns','Nse','Nb','Nbe','lmbd','lmbdecor','lmbderaw','EquivEv'])

print(df)

fig, ax = plt.subplots(2,2,figsize=(12,8))
plot_pull( df['EquivEv'].to_numpy(), ax[0,0] )
plot_pull( df['lmbd'].to_numpy(), ax[0,1])
plot_pull( (df['lmbd'].to_numpy()-4)/df['lmbdecor'], ax[1,0])
plot_pull( (df['lmbd'].to_numpy()-4)/df['lmbderaw'], ax[1,1])
fig.tight_layout()
plt.show()


#print( read('res/cowan_eff_swc_n1000_t99.log') )
#Ns, Nb, lr, lc, ee = read('res/cowan_eff_swc_n1000_t99.log')
