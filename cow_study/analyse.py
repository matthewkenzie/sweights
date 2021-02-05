import os
import sys
sys.path.append("/Users/matt/Scratch/stats/sweights")
from utils import plot_pull
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import pickle

if len(sys.argv)==1:
  sys.exit('usage: python analyse.py -i [files]')

interactive = False
files = sys.argv[1:]
outf = sys.argv[1].split('_t')[0]

if sys.argv[1]=='-i':
  interactive = True
  files = sys.argv[2:]
  outf = sys.argv[2].split('_t')[0]

print('Running analysis on', len(files), 'files')

outf = os.path.dirname(outf) + '/pull_' + os.path.basename(outf) + '.pkl'
print('Will put output in', outf)

plotd = outf.replace('res/','plots/').replace('.pkl','')
print('Plots in', plotd)
os.system('mkdir -p %s'%os.path.dirname(plotd) )

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

res = np.array( [ read(f) for f in files ] )
df  = pd.DataFrame(res, columns=['Ns','Nse','Nb','Nbe','lmbd','lmbdecor','lmbderaw','EquivEv'])

print(df)

truth = {'Ns': 1000, 'Nb': 1000, 'lmbd': 4}
title = {'Ns': r'$N_{s}$', 'Nb': r'$N_{b}$', 'lmbd': r'$\lambda$'}

fig, ax = plt.subplots(3,4,figsize=(20,10))

results = {}

for i, par in enumerate(['Ns','Nb','lmbdc','lmbdr']):
  var = 'lmbd' if par in ['lmbdc','lmbdr'] else par
  err = var+'e'
  if par == 'lmbdc': err += 'cor'
  if par == 'lmbdr': err += 'raw'

  vals = df[var].to_numpy()
  errs = df[err].to_numpy()
  pull = (vals-truth[var])/errs

  (vm,vme),(vs,vse) = plot_pull( vals   , ax[0,i] )
  (em,eme),(es,ese) = plot_pull( errs**2, ax[1,i] )
  (pm,pme),(ps,pse) = plot_pull( pull   , ax[2,i], range=(-3,3) )

  results[par] = { 'vals' : ((vm,vme),(vs,vse)), 'errs': ((em,eme),(es,ese)), 'pull': ((pm,pme),(ps,pse)) }

  ptitle = title[var]
  if par == 'lmbdc': ptitle += ' (corr)'
  if par == 'lmbdr': ptitle += ' (raw)'

  ax[0,i].set_title( ptitle )
  ax[0,i].set_xlabel( '$\langle$'+title[var]+r'$\rangle$' )
  ax[1,i].set_xlabel( '$\sigma($'+title[var]+'$)$')
  ax[2,i].set_xlabel( 'pull('+title[var]+')' )

fig.tight_layout()
fig.savefig( plotd + '.pdf' )

fig, ax = plt.subplots(1,1,figsize=(6,4))
(eqm, eqme),(eqs,eqse) = plot_pull( df['EquivEv'].to_numpy(), ax )
results['Equiv'] = { 'vals' : ((eqm,eqme),(eqs,eqse)) }

ax.set_title('Equivalent Events')
ax.set_xlabel(r'$\left(\sum w \right)^{2} / \sum w^2$')
fig.tight_layout()
fig.savefig( plotd + '_ev.pdf')

with open(outf,'wb') as f:
  pickle.dump(results,f)

# print
rows = []
pm = u'\u00b1'
for par in ['Ns','Nb','lmbdc','lmbdr']:
  vals = results[par]['vals']
  pull = results[par]['pull']

  rows.append( [ par,
      '{:>.2f} {:s} {:<.2f}'.format(vals[0][0],pm,vals[0][1]),
      '{:>.2f} {:s} {:<.2f}'.format(vals[1][0],pm,vals[1][1]),
      '{:>.2f} {:s} {:<.2f}'.format(pull[0][0],pm,pull[0][1]),
      '{:>.2f} {:s} {:<.2f}'.format(pull[1][0],pm,pull[1][1]) ]
    )

print(tabulate(rows,headers=['Par','Value Mean','Value Width','Pull Mean','Pull Width'], colalign=('left','right','right','right','right')))
if interactive: plt.show()
