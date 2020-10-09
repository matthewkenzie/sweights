from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-e','--nevents', default=2500, type=int, help='Total number of events to generate')
parser.add_argument('-z','--zval', default=0.2, type=float, help='z value for signal to background ratio')
opts = parser.parse_args()

import os
import fnmatch

import pandas as pd
import numpy as np

def extract_yields(wfile):
  res = {}
  with open(wfile) as f:
    for line in f.readlines():
      if line.startswith('Method'): continue
      if line.startswith('---'): continue
      else:
        #els = line.split()
        name = line[:22].replace(' ','')
        rest = line[22:]
        res[name] = [ float(x) for x in rest.split() ]
  return res

def extract_slope(sfile):
  res = {}
  with open(sfile) as f:
    for line in f.readlines():
      if line.startswith('Method'): continue
      if line.startswith('---'): continue
      else:
        name = line[:20].replace(' ','')
        rest = line[20:]
        res[name] = [ float(x) for x in rest.split() ]
  return res

def get_vals(df, col, sig=True, nocorrerr=False):

  if col not in list(df.columns): return

  if col.startswith('Slp'): truth = 2.0
  else:
    ind = 0 if sig else 2
    truth = df['YldTruth'].map(lambda x: x[ind]).to_numpy()

  # the values themselves
  if col.startswith('Slp'):
    vals = df[col].map(lambda x: x[0]).to_numpy()
    errs = df[col].map(lambda x: x[1]).to_numpy()
    if nocorrerr:
      errs = df[col].map(lambda x: x[2]).to_numpy()
    vals = vals[errs>0]
    errs = errs[errs>0]
    pull = (vals-truth)/errs
    pull = pull[np.abs(pull)<10]

  else:
    ind = 0 if sig else 2
    vals = df[col].map(lambda x: x[ind]).to_numpy()
    errs = df[col].map(lambda x: x[ind+1]).to_numpy()
    vals = vals[errs>0]
    errs = errs[errs>0]
    pull = (vals-truth)/errs
    pull = pull[np.abs(pull)<10]

  return vals, errs, pull

# collect the files
wts_files = []
slp_files = []
for file in os.listdir('fitres'):
  if fnmatch.fnmatch(file, 'weights_z{0:4.2f}_n{1}_*.txt'.format(opts.zval,opts.nevents)):
    wts_files.append(file)
  if fnmatch.fnmatch(file, 'slope_z{0:4.2f}_n{1}_*.txt'.format(opts.zval,opts.nevents)):
    slp_files.append(file)

wts_files = sorted(wts_files)
slp_files = sorted(slp_files)

# read the files and store in dataframe
data = pd.DataFrame()

for t, wfile in enumerate(wts_files):
  sfile = slp_files[t]
  toyn = int( wfile.split('weights_z{0:4.2f}_n{1}_s'.format(opts.zval,opts.nevents))[1].split('.txt')[0] )
  checkn = int( sfile.split('slope_z{0:4.2f}_n{1}_s'.format(opts.zval,opts.nevents))[1].split('.txt')[0] )
  assert(toyn==checkn)

  yields = extract_yields(os.path.join('fitres',wfile))
  slopes = extract_slope(os.path.join('fitres',sfile))

  if t==0:
    cols = [ 'Yld'+x for x in yields.keys() ] + [ 'Slp'+x for x in slopes.keys() ]
    data = pd.DataFrame( columns=cols )

  for k, i in yields.items():
    ke = 'Yld'+k
    data.at[toyn, ke] = i
  for k, i in slopes.items():
    ke = 'Slp'+k
    data.at[toyn, ke] = i

data = data.sort_index()
print(data)
data.to_pickle('fitres/toyanalysis_z{0:4.2f}_n{1}.pkl'.format(opts.zval,opts.nevents))

import sys
sys.path.append("/Users/matt/Scratch/stats/sweights")
from utils import plot_pull

import matplotlib.pyplot as plt

fig = plt.gcf()
ax  = plt.gca()


from tabulate import tabulate
import uncertainties as u

table = []
os.system('mkdir -p figs/pulls')
os.system('mkdir -p figs/vals')
os.system('mkdir -p figs/errs')

# Signal Yields
for col in data.columns:
  if col=='YldTruth': continue
  if col.startswith('Slp'): continue
  vals, errs, pull = get_vals(data, col)
  if col.startswith('Yld'): col = col.replace('Yld','SigYld')

  # plot the values
  vm, vs = plot_pull(vals, ax)
  ax.set_xlabel('Value')
  ax.set_ylabel('Entries')
  fig.tight_layout()
  fig.savefig('figs/vals/val_z%4.2f_n%d_%s.pdf'%(opts.zval,opts.nevents,col))
  plt.cla()

  # plot the pull
  pm, ps = plot_pull(pull, ax, range=(-5,5))
  ax.set_xlabel('Pull')
  ax.set_ylabel('Entries')
  fig.tight_layout()
  fig.savefig('figs/pulls/pull_z%4.2f_n%d_%s.pdf'%(opts.zval,opts.nevents,col))
  plt.cla()

  # plot the variances
  em, es = plot_pull(errs**2, ax)
  ax.set_xlabel('Variance')
  ax.set_ylabel('Entries')
  fig.tight_layout()
  fig.savefig('figs/errs/var_z%4.2f_n%d_%s.pdf'%(opts.zval,opts.nevents,col))
  plt.cla()

  table.append( [col, vm[0], vm[1], vs[0], vs[1], pm[0], pm[1],ps[0],ps[1], em[0], em[1], es[0], es[1]] )

# Background Yields
for col in data.columns:
  if col=='YldTruth': continue
  if col.startswith('Slp'): continue
  vals, errs, pull = get_vals(data, col, sig=False)
  if col.startswith('Yld'): col = col.replace('Yld','BkgYld')

  # plot the values
  vm, vs = plot_pull(vals, ax)
  ax.set_xlabel('Value')
  ax.set_ylabel('Entries')
  fig.tight_layout()
  fig.savefig('figs/vals/val_z%4.2f_n%d_%s.pdf'%(opts.zval,opts.nevents,col))
  plt.cla()

  # plot the pull
  pm, ps = plot_pull(pull, ax, range=(-5,5))
  ax.set_xlabel('Pull')
  ax.set_ylabel('Entries')
  fig.tight_layout()
  fig.savefig('figs/pulls/pull_z%4.2f_n%d_%s.pdf'%(opts.zval,opts.nevents,col))
  plt.cla()

  # plot the variances
  em, es = plot_pull(errs**2, ax)
  ax.set_xlabel('Variance')
  ax.set_ylabel('Entries')
  fig.tight_layout()
  fig.savefig('figs/errs/var_z%4.2f_n%d_%s.pdf'%(opts.zval,opts.nevents,col))
  plt.cla()

  table.append( [col, vm[0], vm[1], vs[0], vs[1], pm[0], pm[1],ps[0],ps[1], em[0], em[1], es[0], es[1]] )

# Slope Parameters
for col in data.columns:
  if col=='YldTruth': continue
  if not col.startswith('Slp'): continue
  vals, errs, pull = get_vals(data, col)

  # plot the values
  vm, vs = plot_pull(vals, ax)
  ax.set_xlabel('Value')
  ax.set_ylabel('Entries')
  fig.tight_layout()
  fig.savefig('figs/vals/val_z%4.2f_n%d_%s.pdf'%(opts.zval,opts.nevents,col))
  plt.cla()

  # plot the pull
  pm, ps = plot_pull(pull, ax, range=(-5,5))
  ax.set_xlabel('Pull')
  ax.set_ylabel('Entries')
  fig.tight_layout()
  fig.savefig('figs/pulls/pull_z%4.2f_n%d_%s.pdf'%(opts.zval,opts.nevents,col))
  plt.cla()

  # plot the variances
  em, es = plot_pull(errs**2, ax)
  ax.set_xlabel('Variance')
  ax.set_ylabel('Entries')
  fig.tight_layout()
  fig.savefig('figs/errs/var_z%4.2f_n%d_%s.pdf'%(opts.zval,opts.nevents,col))
  plt.cla()

  table.append( [col, vm[0], vm[1], vs[0], vs[1], pm[0], pm[1],ps[0],ps[1], em[0], em[1], es[0], es[1]] )

# Slope Parameters with no correction
for col in data.columns:
  if col=='YldTruth': continue
  if not col.startswith('SlpVar'): continue
  vals, errs, pull = get_vals(data, col, nocorrerr=True)

  # plot the values
  vm, vs = plot_pull(vals, ax)
  ax.set_xlabel('Value')
  ax.set_ylabel('Entries')
  fig.tight_layout()
  fig.savefig('figs/vals/val_nocorr_z%4.2f_n%d_%s.pdf'%(opts.zval,opts.nevents,col))
  plt.cla()

  # plot the pull
  pm, ps = plot_pull(pull, ax, range=(-5,5))
  ax.set_xlabel('Pull')
  ax.set_ylabel('Entries')
  fig.tight_layout()
  fig.savefig('figs/pulls/pull_nocorr_z%4.2f_n%d_%s.pdf'%(opts.zval,opts.nevents,col))
  plt.cla()

  # plot the variances
  em, es = plot_pull(errs**2, ax)
  ax.set_xlabel('Variance')
  ax.set_ylabel('Entries')
  fig.tight_layout()
  fig.savefig('figs/errs/var_nocorr_z%4.2f_n%d_%s.pdf'%(opts.zval,opts.nevents,col))
  plt.cla()

  table.append( [col+'NoCorr', vm[0], vm[1], vs[0], vs[1], pm[0], pm[1],ps[0],ps[1], em[0], em[1], es[0], es[1]] )

heads = [ 'Parameter', 'Mean', 'Err', 'Width', 'Err', 'PullMean', 'Err', 'PullWidth', 'Err', 'VarMean','Err','VarWidth','Err' ]
print(tabulate(table, headers=heads, floatfmt=" 5.2f"))
with open('fitres/toyres_z%4.2f_n%d.log'%(opts.zval,opts.nevents),'w') as f:
  f.write( tabulate(table,headers=heads, floatfmt=" .6f"))
