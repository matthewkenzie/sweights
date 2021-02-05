from toy import toy

nt = toy(eff='nonfact', sfact=True, bfact=False, pars={'g1rh': -0.4, 'z0':0.1})

nt.generate(size=10000, fname='toys/nasty_toy_10000.pkl')
nt.draw(name='plots/nasty_toy_10000.pdf', withtoy=True)

import pandas as pd
import numpy as np

data = pd.read_pickle( 'toys/nasty_toy_10000.pkl' )
mrange = nt.mrange
trange = nt.trange
mbins = 50
tbins = 50

# add the efficiency to the data as a normalised weight
effn = np.sum( 1/nt.effmt(data['mass'].to_numpy(), data['time'].to_numpy() ) )
data.insert( len(data.columns), 'ew', 1./nt.effmt(data['mass'].to_numpy(), data['time'].to_numpy() ) )
print(data)

# do it the classic sweights way

# 1. fit the mass distribution
from scipy.stats import norm, expon
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL, ExtendedUnbinnedNLL

def pdf(x,N,z,mu,sigma,lambd):
  s = norm(mu,sigma)
  b = expon(5000, lambd)
  sn = np.diff(s.cdf(mrange))
  bn = np.diff(b.cdf(mrange))
  p  = z*s.pdf(x)/sn + (1-z)*b.pdf(x)/bn
  return (N, N*p)

mi = Minuit( ExtendedUnbinnedNLL( data['mass'].to_numpy() , pdf ), N=1000, z=0.1, mu=5280, sigma=30, lambd=400 )
mi.migrad()
mi.hesse()
print(mi.params)

def plotpdf(x):
  return pdf(x,**mi.values)[1] * np.diff(mrange)/mbins

# def plot fit
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1,figsize=(8,6))
w, xe = np.histogram( data['mass'].to_numpy(), bins=mbins, range=mrange)
cx = 0.5*(xe[1:]+xe[:-1])
ax.errorbar( cx, w, w**0.5, fmt='ko')
x = np.linspace(*mrange,200)
ax.plot(x, plotpdf(x))

# compute classic sweights (ala Schemlling)
import sys
sys.path.append('../')
from SWeighter import SWeight
pdfs = [ norm( mi.values['mu'], mi.values['sigma'] ), expon(5000, mi.values['lambd'] ) ]
ylds = [ mi.values['N']*mi.values['z'], mi.values['N']*(1-mi.values['z']) ]

sw = SWeight( data['mass'].to_numpy(), pdfs=pdfs, yields=ylds, discvarranges=(mrange,), method='summation', compnames=('sig','bkg') )

# plot weights
fig, ax = plt.subplots(1,1,figsize=(8,6))
sw.makeWeightPlot(ax)
ax.set_xlabel('mass')
ax.set_ylabel('weight')

# add weights to frame
data.insert( len(data.columns), 'sw', sw.getWeight(0, data['mass'].to_numpy() ) )
data.insert( len(data.columns), 'bw', sw.getWeight(1, data['mass'].to_numpy() ) )

# fit back the weighted data
def tnll(lambd):
  b = expon(0, lambd)
  bn = np.diff(b.cdf(trange))
  return -np.sum( data['sw'].to_numpy() * data['ew'].to_numpy() * np.log( b.pdf( data['time'].to_numpy() ) / bn ) )

def tpdf(lambd,x):
  b = expon(0,lambd)
  bn = np.diff(b.cdf(trange))
  return b.pdf(x) / bn

tmi = Minuit( tnll, lambd=2, limit_lambd=(1,3), errordef=Minuit.LIKELIHOOD, pedantic=False )
tmi.migrad()
tmi.hesse()
print(tmi.params)
val = tmi.np_values()
cov = tmi.np_covariance()

from CovarianceCorrector import cov_correct
ncov = cov_correct( tpdf, data['time'].to_numpy(), data['sw'].to_numpy() * data['ew'].to_numpy(), tmi.np_values(), tmi.np_covariance() )

fig, ax = plt.subplots(1,1,figsize=(8,6))
#w, xe = np.histogram( data['time'].to_numpy(), range=trange, bins=tbins, weights=data['sw'].to_numpy() * data['ew'].to_numpy() )
w, xe = np.histogram( data['time'].to_numpy(), range=trange, bins=tbins, weights= data['ew'].to_numpy() )
cx = 0.5 * (xe[1:]+xe[:-1])

ax.errorbar(cx, w, np.abs(w)**0.5, fmt='ko')

x = np.linspace(*trange,100)
#tn = np.sum(data['sw'].to_numpy()*data['ew'].to_numpy())*np.diff(trange)/tbins
tn = np.sum(data['ew'].to_numpy())*np.diff(trange)/tbins
ax.plot(x, tn*tpdf(tmi.values['lambd'],x), 'r-')
ax.plot(x, tn*tpdf(2,x), 'b--')
#ax.set_yscale('log')


plt.show()
