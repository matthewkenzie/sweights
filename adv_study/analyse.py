from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-e','--nevents', default=2500, type=int, help='Total number of events to generate')
opts = parser.parse_args()

import os
import fnmatch

import pandas as pd
import numpy as np

def extract_shape(sfile):
  res = {}
  with open(sfile) as f:
    for line in f.readlines():
      if line.startswith('Parameter'): continue
      if line.startswith('---'): continue
      else:
        els = line.split()
        res[els[0]] = float(els[1])
  return res

def get_vals(df, col, sig=True):

  if col not in list(df.columns): return

  if col.startswith('Slp'): truth = 2.0
  else:
    ind = 0 if sig else 2
    truth = df['YldTruth'].map(lambda x: x[ind]).to_numpy()

  # the values themselves
  if col.startswith('Slp'):
    vals = df[col].map(lambda x: x[0]).to_numpy()
    errs = df[col].map(lambda x: x[1]).to_numpy()
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
shp_files = []
for file in os.listdir('fitres'):
  if fnmatch.fnmatch(file, 'shape_n{0}_*.txt'.format(opts.nevents)):
    shp_files.append(file)

shp_files = sorted(shp_files)

# read the files and store in dataframe
data = pd.DataFrame()

for t, sfile in enumerate(shp_files):
  toyn = int( sfile.split('shape_n{0}_s'.format(opts.nevents))[1].split('.txt')[0] )
  res = extract_shape(os.path.join('fitres',sfile))

  if t==0:
    data = pd.DataFrame( columns=res.keys() )

  data = data.append( res, ignore_index=True )

data = data.sort_index()
#print(data)

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

truth = { 'Mean': 10, 'Sigma': 0.2 }

for par in ['Mean','Sigma']:
  vals = data[par].to_numpy()
  errs = data[par+'Err'].to_numpy()

  vals = vals[errs>0]
  errs = errs[errs>0]
  pull = (vals - truth[par]) / errs
  pull = pull[np.abs(pull)<10]

  # plot the values
  vm, vs = plot_pull(vals, ax)
  ax.set_xlabel('Value')
  ax.set_ylabel('Entries')
  fig.tight_layout()
  fig.savefig('figs/vals/val_n%d_%s.pdf'%(opts.nevents,par))
  plt.cla()

  # plot the pull
  pm, ps = plot_pull(pull, ax, range=(-5,5))
  ax.set_xlabel('Pull')
  ax.set_ylabel('Entries')
  fig.tight_layout()
  fig.savefig('figs/pulls/pull_n%d_%s.pdf'%(opts.nevents,par))
  plt.cla()

  table.append( [par, vm[0], vm[1], vs[0], vs[1], pm[0], pm[1],ps[0],ps[1]] )

heads = [ 'Parameter', 'Mean', 'Err', 'Width', 'Err', 'PullMean', 'Err', 'PullWidth', 'Err' ]
print(tabulate(table, headers=heads, floatfmt=" 5.2f"))
with open('fitres/toyres_n%d.log'%opts.nevents,'w') as f:
  f.write( tabulate(table,headers=heads, floatfmt=" .6f"))
