from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-e','--nevents', default=2500, type=int, help='Total number of events to generate')
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
  if fnmatch.fnmatch(file, 'slope_n{0}_*.txt'.format(opts.nevents)):
    slp_files.append(file)

slp_files = sorted(slp_files)

# read the files and store in dataframe
data = pd.DataFrame()

vals = []
errs = []
pull = []

for t, sfile in enumerate(slp_files):
  toyn = int( sfile.split('slope_n{0}_s'.format(opts.nevents))[1].split('.txt')[0] )
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

# plot the values
vm, vs = plot_pull(vals, ax)
ax.set_xlabel('Value')
ax.set_ylabel('Entries')
fig.tight_layout()
fig.savefig('figs/vals/val_n%d_Slope.pdf'%(opts.nevents))
plt.cla()

# plot the pull
pm, ps = plot_pull(pull, ax, range=(-5,5))
ax.set_xlabel('Pull')
ax.set_ylabel('Entries')
fig.tight_layout()
fig.savefig('figs/pulls/pull_n%d_Slope.pdf'%(opts.nevents))
plt.cla()

table.append( ['Slope', vm[0], vm[1], vs[0], vs[1], pm[0], pm[1],ps[0],ps[1]] )

heads = [ 'Parameter', 'Mean', 'Err', 'Width', 'Err', 'PullMean', 'Err', 'PullWidth', 'Err' ]
print(tabulate(table, headers=heads, floatfmt=" 5.2f"))
with open('fitres/toyres_n%d.log'%opts.nevents,'w') as f:
  f.write( tabulate(table,headers=heads, floatfmt=" .6f"))
