from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-e','--nevents', default=2500, type=int, help='Total number of events to generate')
parser.add_argument('-m','--mbins'  , default="auto", type=str          , help='Mass binning')
parser.add_argument('-W','--wrong'  , default=False, action="store_true", help='Look for wrong fitting files')
opts = parser.parse_args()

import os
import fnmatch

import pandas as pd
import numpy as np

def extract_slope(sfile):
  with open(sfile) as f:
    for line in f.readlines():
      els = line.split()
      return float(els[2]),float(els[4])

# collect the files
slp_files = []
for file in os.listdir('fitres'):
  match = 'slope_n{0}_*.txt'.format(opts.nevents) if opts.mbins=='auto' else 'slope_m{0}_n{1}_*.txt'.format(int(opts.mbins),opts.nevents)
  if opts.wrong: match = 'slope_wrong_n{0}_*.txt'.format(opts.nevents)
  if fnmatch.fnmatch(file, match):
    slp_files.append(file)

slp_files = sorted(slp_files)

# read the files and store in dataframe
data = pd.DataFrame()

vals = []
errs = []
pull = []

for t, sfile in enumerate(slp_files):
  toyn = int( sfile.split('_n{0}_s'.format(opts.nevents))[1].split('.txt')[0] )
  slopes = extract_slope(os.path.join('fitres',sfile))

  vals.append( slopes[0] )
  errs.append( slopes[1] )
  pull.append( (2.0 - slopes[0])/slopes[1] )

vals = np.array(vals)
errs = np.array(errs)
pull = np.array(pull)
#vals = vals[(errs>1e-6) & (errs<1)]
#errs = errs[(errs>1e-6) & (errs<1)]
#pull = pull[np.abs(pull)<10]

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

fext = 'n%d'%opts.nevents
if opts.mbins!='auto': fext = 'm%d_n%d'%(int(opts.mbins),opts.nevents)
if opts.wrong: fext = 'wrong_n%d'%opts.nevents

# plot the values
vm, vs = plot_pull(vals, ax)
ax.set_xlabel('Value')
ax.set_ylabel('Entries')
fig.tight_layout()
fig.savefig('figs/vals/val_%s_Slope.pdf'%(fext))
plt.cla()

# plot the pull
pm, ps = plot_pull(pull, ax, range=(-5,5))
ax.set_xlabel('Pull')
ax.set_ylabel('Entries')
fig.tight_layout()
fig.savefig('figs/pulls/pull_%s_Slope.pdf'%(fext))
plt.cla()

# plot the variances
em, es = plot_pull(errs**2, ax)
ax.set_xlabel('Variance')
ax.set_ylabel('Entries')
fig.tight_layout()
fig.savefig('figs/errs/var_%s_Slope.pdf'%(fext))
plt.cla()

table.append( ['Slope', vm[0], vm[1], vs[0], vs[1], pm[0], pm[1], ps[0], ps[1], em[0], em[1], es[0], es[1]] )

heads = [ 'Parameter', 'Mean', 'Err', 'Width', 'Err', 'PullMean', 'Err', 'PullWidth', 'Err', 'VarMean', 'Err', 'VarWidth', 'Err' ]
print(tabulate(table, headers=heads, floatfmt=" 5.2f"))
with open('fitres/toyres_%s.log'%fext,'w') as f:
  f.write( tabulate(table,headers=heads, floatfmt=" .6f"))
