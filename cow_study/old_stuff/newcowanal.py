import os
import sys
sys.path.append("/Users/matt/Scratch/stats/sweights")

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-t','--toyn'   , default=0    , type=int           , help='Toy number')
parser.add_argument('-n','--nevs'   , default=1000 , type=int           , help='Events per toy')
parser.add_argument('-b','--bfact'  , default=False, action="store_true", help='Use factorising background truth model' )
parser.add_argument('-e','--eff'    , default=False, action="store_true", help='Include efficiency effects' )
parser.add_argument('-c','--cow'    , default=False, action="store_true", help='Use COWs')
parser.add_argument('-S','--gs'     , default=0    , type=int           , help='Use of gs(m) [0: truth, 1: crazy, 3: obs/eff]')
parser.add_argument('-B','--gb'     , default=0    , type=int           , help='Use of gb(m) [0: truth, 1: crazy, 2: reasonable]')
parser.add_argument('-p','--bpol'   , default=-1   , type=int           , help='Use background polynomials with of this order')
parser.add_argument('-I','--Im'     , default=1    , type=int           , help='Function for I(m) [1: flat, 2: truth, 3: obs/eff2]')
parser.add_argument('-P','--noplots', default=False, action="store_true", help='Don\'t make plots')
opts = parser.parse_args()

# make a unique string for output files so we know what we did
uid_str = 'cowan'
if opts.toyn<0 and opts.bfact: uid_str += '_bf'
if opts.eff: uid_str += '_eff'
if opts.cow:
  uid_str += '_cow'
  uid_str += '_gs%d'%opts.gs
  if opts.bpol>=0: uid_str += '_bp%d'%opts.bpol
  else: uid_str += '_gb%d'%opts.gb
  uid_str += '_I%d'%opts.Im
else:
  uid_str += '_swc'

print('Running COW analysis with UID', uid_str)

plotdir = 'plots/' + uid_str
os.system('mkdir -p %s'%plotdir)
os.system('mkdir -p res')
outfname = 'res/' + uid_str + '_n%d_t%d'%(opts.nevs,opts.toyn) + '.log'

import numpy as np
from scipy.stats import truncnorm, truncexpon
from scipy.integrate import quad
from iminuit import Minuit
import matplotlib.pyplot as plt
from SWeighter import SWeight
from CovarianceCorrector import cov_correct

def hist(vals, range=None, bins=25, weights=None):
  w, xe = np.histogram(vals, range=range, bins=bins, weights=weights)
  if weights is not None:
    w2, xe = np.histogram(vals, range=range, bins=bins, weights=weights**2)
  cx = 0.5 * (xe[1:] + xe[:-1])
  if weights is None: return cx, w
  return cx, w, w2

def mynorm(xmin,xmax,mu,sg):
  a, b = (xmin-mu)/sg, (xmax-mu)/sg
  return truncnorm(a,b,mu,sg)

def myexp(xmin,xmax,lb):
  return truncexpon( (xmax-xmin)/lb, xmin, lb )

def mypol(xmin,xmax,po):
  return lambda x: (po+1)*(1-(x-xmin)/(xmax-xmin))**po / (xmax-xmin)

def eff(m,t,mrange,trange,ea=0.5,eb=0.2,el=-0.05):
  # get m into range 0-1
  mscl = ( m - mrange[0] ) / ( mrange[1] - mrange[0] )
  ascl = ea + eb*mscl
  f     = ascl * ( t - el ) ** ascl
  # scale it so that the max is 1. which happens at trange[1], mrange[1]
  mx = (ea+eb) * ( trange[1] - el ) ** (ea+eb)
  return f/mx

# load toys (no eff)
if opts.toyn<0:
  if opts.eff:
    bkgtoy = np.load('toys/newcoweffbkg.npy')
    sigtoy = np.load('toys/newcoweffsig.npy')
  else:
    if opts.bfact: bkgtoy = np.load('toys/newcowfactbkg.npy')
    else: bkgtoy = np.load('toys/newcowbkg.npy')
    sigtoy = np.load('toys/newcowsig.npy')
else:
  bkgtfile = 'toys/nceffb_s%d_t%d.npy'%(opts.nevs,opts.toyn)
  sigtfile = 'toys/nceffs_s%d_t%d.npy'%(opts.nevs,opts.toyn)
  if not os.path.exists(bkgtfile): raise RuntimeError('No such file at', bkgtfile)
  if not os.path.exists(sigtfile): raise RuntimeError('No such file at', sigtfile)
  bkgtoy = np.load(bkgtfile)
  sigtoy = np.load(sigtfile)

toy = np.concatenate((bkgtoy,sigtoy))

mrange = (5000,5600)
mbins = 50
trange = (0,10)
tbins = 50

# do the cow thing
from cow import cow
if opts.cow:
  # gs function
  if   opts.gs == 0: gs = lambda m: mynorm(*mrange,5280,30).pdf(m)
  elif opts.gs == 1: gs = lambda m: mynorm(*mrange,5200,50).pdf(m)
  elif opts.gs == 3:
    w, xe = np.histogram( toy[:,0], range=mrange, bins=mbins, weights=(1/eff(toy[:,0],toy[:,1],mrange,trange)) )
    w = w/np.sum(w)
    w /= (mrange[1]-mrange[0])/len(w)
    f = lambda m: w[ np.argmin( m >= xe ) - 1 ]
    gs = np.vectorize(f)
  else: raise RuntimeError('This option for gs', opts.gs, 'is not implemented')
  # gb function(s)
  if opts.bpol<0:
    if   opts.gb == 0: gb = lambda m: myexp(*mrange,400).pdf(m)
    elif opts.gb == 1: gb = lambda m: myexp(*mrange,50).pdf(m)
    elif opts.gb == 2: gb = lambda m: myexp(*mrange,173).pdf(m)
    else: raise RuntimeError('This option for gb', opts.gb, 'is not implemented')
  else:
    gb = [ mypol(*mrange,i) for i in range(opts.bpol+1) ]

  # Im function
  obs = None
  trgs = mynorm(*mrange,5280,30)
  trgb = myexp(*mrange,400)
  if   opts.Im==1: Im = 1
  elif opts.Im==2: Im = lambda m: (1000/1800)*trgb.pdf(m) + (800/1800)*trgs.pdf(m)
  elif opts.Im==3:
    Im = 1
    if opts.eff: obs = np.histogram( toy[:,0], range=mrange, bins=mbins, weights=(1/eff(toy[:,0],toy[:,1],mrange,trange)**2) )
    else:        obs = np.histogram( toy[:,0], range=mrange, bins=mbins )
  else: raise RuntimeError('This option for Im', opts.Im, 'is not implemented')

  cw = cow(mrange, gs, gb, Im, obs)

  sws = cw.wk(0,toy[:,0])
  bws = [ cw.wk(i,toy[:,0]) for i in range(1,len(cw.gk)) ]

  if not opts.noplots:
    # plot the functions being used
    fig, ax = plt.subplots(1, 1, figsize=(8,6) )
    ch, wh = hist( toy[:,0], range=mrange, bins=mbins )
    ax.errorbar( ch, wh, wh**0.5, fmt='ko' )
    N = len(toy)*(mrange[1]-mrange[0])/mbins
    x = np.linspace(*mrange,200)
    ax.plot( x, N*cw.gs(x), 'b-', label='$g_{s}(m)$' )
    for i, gk in enumerate(cw.gb):
      lab = '$g_{b}(m)$' if len(cw.gb)==1 else '$g_{b%d}(m)$'%i
      ax.plot( x, N*gk(x), 'r-', label=lab )
    ax.plot( x, N*cw.Im(x), 'k--', label='$I(m)$' )
    ax.legend()
    ax.set_xlabel('Mass [GeV]')
    fig.tight_layout()
    fig.savefig('%s/newcowmfunc.pdf'%plotdir)

    # plot weight functions
    fig, ax = plt.subplots(1, 1, figsize=(8,6) )
    x = np.linspace(*mrange,100)
    ax.plot( x, cw.wk(0,x), 'b-', label='$w_{s}$' )
    for i in range(len(cw.gb)):
      lab = '$w_{b}(m)$' if len(cw.gb)==1 else '$g_{b%d}(m)$'%i
      ax.plot( x, cw.wk(i+1,x), 'r--', label=lab )
    ax.plot( x, cw.wk(0,x) + np.sum( [ cw.wk(i,x) for i in range(1,len(cw.gk)) ], axis=0 ), 'k-', lw=2, label='$\sum_i w_{i}$' )
    ylim = ax.get_ylim()
    if ylim[0]<-10: ylim = (-10,ylim[1])
    if ylim[1]>10:  ylim = (ylim[0],10)
    ax.set_ylim( *ylim )
    ax.legend()
    ax.set_xlabel('Mass [GeV]')
    ax.set_ylabel('sWeight')
    fig.tight_layout()
    fig.savefig('%s/newcowweights.pdf'%plotdir)

# do the classic sweight thing
else:

  # fit the mass
  def nll(ns,nb,mu,sg,lb):
    spdf = mynorm(*mrange,mu,sg)
    bpdf = myexp(*mrange,lb)
    return ns + nb - np.sum(np.log( ns*spdf.pdf(toy[:,0]) + nb*bpdf.pdf(toy[:,0]) ) )

  mi = Minuit(nll,ns=0.5*len(toy),nb=0.5*len(toy),mu=5280,sg=30,lb=400,errordef=Minuit.LIKELIHOOD,pedantic=False)
  mi.migrad()
  mi.hesse()
  print(mi.params)

  # draw the mass fit
  def pdf(m,ns,nb,mu,sg,lb,so=False,bo=False):
    spdf = mynorm(*mrange,mu,sg)
    bpdf = myexp(*mrange,lb)
    if so: return ns*spdf.pdf(m)
    if bo: return nb*bpdf.pdf(m)
    return ns*spdf.pdf(m) + nb*bpdf.pdf(m)

  mhc, mhw = hist(toy[:,0], range=mrange, bins=mbins)
  m = np.linspace(*mrange,200)
  mN = (mrange[1]-mrange[0])/mbins
  s = mN*pdf(m,**mi.values,so=True)
  b = mN*pdf(m,**mi.values,bo=True)

  if not opts.noplots:
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    ax.errorbar(mhc, mhw, mhw**0.5, fmt='ko', label='Toy Data')
    ax.plot(m, s, 'g--', label='Fitted S pdf')
    ax.plot(m, b, 'r--', label='Fitted B pdf')
    ax.plot(m, s+b, 'b-', label='Fitted S+B pdf')
    ax.set_xlabel('Mass [GeV]')
    ax.set_ylabel('Events')
    ax.legend()
    fig.savefig('%s/newcowmfit.pdf'%plotdir)
    fig.tight_layout()

  # now do the classic sweighting
  pdfs = [ mynorm(*mrange, mi.values['mu'], mi.values['sg']),
           myexp(*mrange, mi.values['lb']) ]
  ylds = [ mi.values['ns'], mi.values['nb'] ]

  sw = SWeight( toy[:,0], pdfs=pdfs, yields=ylds, discvarranges=(mrange,), compnames=('sig','bkg'), method='integration' )

  # plot it
  if not opts.noplots:
    fig, ax = plt.subplots(1, 1, figsize=(8,6) )
    sw.makeWeightPlot(ax)
    ax.set_xlabel('Mass [GeV]')
    ax.set_ylabel('sWeight')
    fig.tight_layout()
    fig.savefig('%s/newcowweights.pdf'%plotdir)

  # save weights
  sws = sw.getWeight(0, toy[:,0])
  bws = sw.getWeight(1, toy[:,0])

## NOW FIT BACK WITH OUR WEIGHTS FROM WHATEVER METHOD ##

outf = open(outfname,'w')

# averages for cow
if opts.cow:
  bwssum = np.sum( bws, axis=0 )
  print( 'zs:', np.mean(sws) )
  print( 'zb:', np.mean(bwssum) )
  print( 'Ns:', np.sum(sws), '+/-', np.sum(sws**2)**0.5 )
  print( 'Nb:', np.sum(bwssum), '+/-', np.sum(bwssum**2)**0.5 )

# save
outf.write('Ns: {:f} +/- {:f}\n'.format( np.sum(sws), np.sum(sws**2)**0.5) )
if opts.cow:
  outf.write('Nb: {:f} +/- {:f}\n'.format( np.sum(bwssum), np.sum(bwssum**2)**0.5) )
else:
  outf.write('Nb: {:f} +/- {:f}\n'.format( np.sum(bws), np.sum(bws**2)**0.5) )

# effwts
if opts.eff:
  effs = eff(toy[:,0],toy[:,1],mrange,trange)
  sws /= effs
  bws = np.sum( [ bw/effs for bw in bws ], axis=0 )

# fit back weighted data
def wnll(lb):
  b = myexp(*trange,lb)
  return -np.sum( sws * np.log( b.pdf( toy[:,1] ) ) )

def tpdf(lb,x):
  b = myexp(*trange,lb)
  return b.pdf(x)

tmi = Minuit(wnll, lb=4, limit_lb=(0,20), errordef=Minuit.LIKELIHOOD, pedantic=False)
tmi.migrad()
tmi.hesse()
print(tmi.params)

ncov = cov_correct(tpdf, toy[:,1], sws, tmi.np_values(), tmi.np_covariance(), verbose=True)

print('Fitted {:5.3f} +/- {:5.3f}'.format( tmi.values['lb'], ncov[0,0]**0.5 ) )

outf.write('Fitted: {:7.5f} +/- {:7.5f} (corr) {:7.5f} (raw)\n'.format( tmi.values['lb'], ncov[0,0]**0.5, tmi.errors['lb'] ) )

sum2ws = np.sum(sws)**2
sumws2 = np.sum(sws**2)
equivev = sum2ws/sumws2
print('Equivalent Events: {:f}'.format(equivev))
outf.write('EquivalentEvents: {:f}\n'.format(equivev))

outf.close()

# and plot it
thc, thw, thw2 = hist(toy[:,1], range=trange, bins=tbins, weights=sws)
t = np.linspace(*trange,200)
tN = np.sum(sws)*(trange[1]-trange[0])/tbins

if not opts.noplots:
  fig, ax = plt.subplots(1, 1, figsize=(8,6) )
  ax.errorbar( thc, thw, thw2**0.5, fmt='ko', label='Weighted Toy Data' )
  ax.plot(t, tN*tpdf(4,t), 'r--', label='True PDF')
  ax.plot(t, tN*tpdf(tmi.values['lb'],t), 'b-', label='Fitted PDF')
  ax.set_xlabel('Decay Time [ps]')
  ax.set_ylabel('Weighted Events')
  ax.text(0.7,0.8,r'Fitted $\lambda={:>5.2f} \pm {:<5.2f}$'.format(tmi.values['lb'],ncov[0,0]**0.5), transform=ax.transAxes)
  ax.text(0.7,0.75,r'$\left(\sum w\right)^2 / \sum w^2 = {:<6.1f}$'.format(equivev), transform=ax.transAxes)
  ax.legend()
  fig.tight_layout()
  fig.savefig('%s/newcowtfit.pdf'%plotdir)

  plt.show()
