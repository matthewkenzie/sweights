import pandas as pd
from argparse import ArgumentParser
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import boost_histogram as bh
from scipy.stats import norm

parser = ArgumentParser()
parser.add_argument('-n','--ntoys', type=int, default=500, help='Number of toys to read')
parser.add_argument('-d','--dir'    , default='toys',                     help='Directory to save toy output in')
parser.add_argument('-s','--startat', default=1, type=int,                help='Start number for toys')
parser.add_argument('-e','--nevents', default=10000, type=int,            help='Number of events per toy')
opts = parser.parse_args()

true_n_sig = []
sum_ws = []
sum_ws2 = []
fit_n_sig = []
fit_err_n_sig = []

for i in range(opts.startat,opts.ntoys+opts.startat):
  if not os.path.exists('%s/toy%d/weights.pkl'%(opts.dir,i)):
    raise RuntimeWarning("No weights.pkl file found in %s/toy%d"%(opts.dir,i))
    continue

  if not os.path.exists('%s/toy%d/fit.pkl'%(opts.dir,i)):
    raise RuntimeWarning("No fit.pkl file found in %s/toy%d"%(opts.dir,i))
    continue

  toy = pd.read_pickle('%s/toy%d/weights.pkl'%(opts.dir,i))

  true_n_sig.append( len( toy[ toy['ctrl']==0 ] ) )
  #print( 'Toy', i, 'found', true_n_sig[-1], 'signal events' )
  sum_ws    .append( np.sum( toy['wsig'].to_numpy() ) )
  sum_ws2   .append( np.sum( toy['wsig'].to_numpy()**2 ) )

  frf = open('%s/toy%d/fit.pkl'%(opts.dir,i), 'rb')
  fr = pickle.load( frf )
  frf.close()

  fit_n_sig.append( fr['sig_y'] )
  fit_err_n_sig.append( fr['error_sig_y'] )

  # specific dalitz plot thing
  #whist = bh.Histogram( bh.axis.Regular(50,5,28), storage=bh.storage.Weight())
  #whist.fill( toy['m2ab'].to_numpy(), weight=toy['wsig'].to_numpy() )

  #fig, ax = plt.subplots(1, 1, figsize=(6,4) )
  #ax.errorbar( whist.axes[0].centers, whist.view().value, whist.view().variance**0.5, fmt='ko' )

  #plt.show()
  #input()

fig, ax = plt.subplots(1, 2, figsize=(12,4) )

#true_n_sig = np.array( true_n_sig )
true_n_sig = float( opts.nevents*0.2 )
fit_n_sig  = np.array( fit_n_sig )
fit_err_n_sig = np.array( fit_err_n_sig )
sum_ws = np.array( sum_ws )
sum_ws2 = np.array( sum_ws2 )

import sys
sys.path.append('/Users/matt/bin')
from plots import pull_plot

v1 = (true_n_sig - fit_n_sig)/fit_err_n_sig
v2 = (true_n_sig - sum_ws)/sum_ws2**0.5

pull_plot( v1, axis=ax[0] )
pull_plot( v2, axis=ax[1] )

#plt.hist( fit_err_n_sig, bins=100 )
#plt.show()
#input()

#nbins = 25
#prange = (-5,5)

#w1, x1 = np.histogram( v1 , bins=nbins, range=prange )
#m1, s1 = norm.fit(v1)
#c1 = 0.5 * ( x1[1:] + x1[:-1] )
#f1 = norm(m1,s1)

#w2, x2 = np.histogram( v2, bins=nbins, range=prange )
#m2, s2 = norm.fit(v2)
#c2 = 0.5 * ( x2[1:] + x2[:-1] )

#x = np.linspace(*prange, 100)

#ax[0].errorbar( c1, w1, w1**0.5, 0.5*(x1[:-1]-x1[1:]), fmt='ko' )
#ax[0].plot(x, norm(m1,s1).pdf(x)/np.diff( norm(m1,s1).cdf(prange) ) * np.diff(prange)/nbins * len(v1), 'b-' )

#ax[1].errorbar( c2, w2, w2**0.5, 0.5*(x2[:-1]-x2[1:]), fmt='ko' )
#ax[1].plot(x, norm(m2,s2).pdf(x)/np.diff( norm(m2,s2).cdf(prange) ) * np.diff(prange)/nbins * len(v2), 'b-' )

#print(np.mean(v1), np.std(v1))
#print(m1, s1)
#print(np.mean(v2), np.std(v2))
#print(m2, s2)

fig.tight_layout()
plt.show()
