import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import pickle

def to_pickle(fname, obj):
  outf = open(fname,'wb')
  pickle.dump( obj, outf )
  outf.close()

def read_pickle(fname):
  inf = open(fname,'rb')
  obj = pickle.load(inf)
  inf.close()
  return obj

def plot_pull(vals, ax=None, bins=25, range=None):

  ax = ax or plt.gca()

  # filter out NaN and inf
  vals = vals[np.isfinite(vals) ]

  w, xe = np.histogram( vals, bins=bins, range=range )
  cx = 0.5 * (xe[1:] + xe[:-1])
  ex = 0.5 * (xe[1:] - xe[:-1])

  # draw the values
  ax.errorbar( cx, w, w**0.5, ex, fmt='ko', elinewidth=1., markersize=3., capsize=1.5 )

  mean = np.mean(vals)
  sdev = np.std(vals)
  merr = sdev / len(vals)**0.5
  serr = sdev / (2*len(vals) - 2)**0.5

  # draw a Gaussian
  x = np.linspace(xe[0],xe[-1],100)
  N = (xe[-1]-xe[0])/len(cx) * len(vals)
  ax.plot( x, N * norm(loc=mean, scale=sdev).pdf(x), 'r-' )

  # plot the values
  ax.text( 0.7, 0.8, r'$\mu = {:.2f} \pm {:.2f}$'.format(mean,merr), transform=ax.transAxes)
  ax.text( 0.7, 0.7, r'$\sigma = {:.2f} \pm {:.2f}$'.format(sdev,serr), transform=ax.transAxes)

  return (mean,merr),(sdev,serr)

def hist(vals, range=None, bins=25, weights=None):
  w, xe = np.histogram(vals, range=range, bins=bins, weights=weights)
  cx = 0.5 * (xe[1:] + xe[:-1])
  if weights is not None:
    w2, xe = np.histogram(vals, range=range, bins=bins, weights=weights**2)
    return cx, w, w2
  return cx, w

