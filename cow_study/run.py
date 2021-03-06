import os
import sys
sys.path.append("/Users/matt/Scratch/stats/sweights")

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-s','--sfile'  , help='Signal toy file')
parser.add_argument('-b','--bfile'  , help='Background toy file')
parser.add_argument('-o','--oname'  , default='cow', help='Output file name')
parser.add_argument('-e','--eff'    , default=False, action="store_true", help='Include efficiency effects' )
parser.add_argument('-c','--cow'    , default=False, action="store_true", help='Use COWs')
parser.add_argument('-S','--gs'     , default=0    , type=int           , help='Use of gs(m) [0: truth, 1: crazy, 3: obs/eff]', choices=[0,1,3])
parser.add_argument('-B','--gb'     , default=0    , type=int           , help='Use of gb(m) [0: truth, 1: crazy, 2: reasonable]', choices=[0,1,2])
parser.add_argument('-p','--bpol'   , default=-1   , type=int           , help='Use background polynomials with of this order')
parser.add_argument('-I','--Im'     , default=1    , type=int           , help='Function for I(m) [1: flat, 2: truth, 3: obs/eff2]', choices=[1,2,3])
parser.add_argument('-q','--batch'  , default=False, action="store_true", help='Run in batch mode. Will not make plots. Will supress output.')
opts = parser.parse_args()

if opts.batch:
  f = open(os.devnull, 'w')
  sys.stdout = f

def polstr(n):
  res = ''
  for i in range(n+1):
    if i==0: res += '1'
    elif i==1: res += ', x'
    else: res += ', x**%d'%i
  return res

def printopts(opts, file=sys.stdout):
  if file != sys.stdout:
    print('Run Info:', file=file)
  if opts.cow:
    print(' - S toy:', opts.sfile, file=file)
    print(' - B toy:', opts.bfile, file=file)
    print(' - Using COW formalism', file=file)
    if opts.gs==0: print('   - gs(m) = truth:    gaus(5280,30)', file=file)
    if opts.gs==1: print('   - gs(m) = crazy:    gaus(5200,50)', file=file)
    if opts.gs==3: print('   - gs(m) = obs/eff:  p(m)', file=file)
    if opts.bpol>=0:
      print('   - gb(m) = pol%d:     [%s]'%(opts.bpol,polstr(opts.bpol)), file=file)
    else:
      if opts.gb==0: print('   - gb(m) = truth:    expo(400)', file=file)
      if opts.gb==1: print('   - gb(m) = crazy:    expo(50)', file=file)
      if opts.gb==2: print('   - gb(m) = reas:     expo(180)', file=file)
    if opts.Im==1: print('   - I(m)  = flat:     1', file=file)
    if opts.Im==2: print('   - I(m)  = truth:    z*gaus(5280,30) + (1-z)*expo(400)', file=file)
    if opts.Im==3: print('   - I(m)  = obs/eff2: q(m)', file=file)
  else:
    print(' - Using SWeights formalism', file=file)
  if opts.eff:
    print(' - Including efficiency weight', file=file)

print('Running analysis')
if not opts.batch:
  printopts(opts)
  os.system('mkdir -p plots/%s'%os.path.dirname(opts.oname))
os.system('mkdir -p res/%s'%os.path.dirname(opts.oname))
opts.oname = os.path.splitext(opts.oname)[0]

import numpy as np
from scipy.stats import truncnorm, truncexpon
from scipy.integrate import quad
from iminuit import Minuit
import matplotlib.pyplot as plt
from models import mynorm, myexp, mypol, effmodel
from SWeighter import SWeight
from CovarianceCorrector import cov_correct
from utils import hist

# load toys
assert( os.path.exists( opts.sfile ) )
assert( os.path.exists( opts.bfile ) )
bkgtoy = np.load( opts.bfile )
sigtoy = np.load( opts.sfile )
nbkg = len(bkgtoy)
nsig = len(sigtoy)
toy = np.concatenate((bkgtoy,sigtoy))

mrange = (5000,5600)
mbins = 50
trange = (0,10)
tbins = 50

# make efficiency model if needed
if opts.eff:
  eff = effmodel(mrange,trange,0.2,0.2,-0.15, cache='load')

# do the cow thing
from cow import cow
if opts.cow:
  # gs function
  if   opts.gs == 0: gs = lambda m: mynorm(*mrange,5280,30).pdf(m)
  elif opts.gs == 1: gs = lambda m: mynorm(*mrange,5200,50).pdf(m)
  elif opts.gs == 3:
    w, xe = np.histogram( toy[:,0], range=mrange, bins=mbins, weights=(1/eff(toy[:,0],toy[:,1])) )
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
  elif opts.Im==2: Im = lambda m: (nbkg/(nsig+nbkg))*trgb.pdf(m) + (nsig/(nsig+nbkg))*trgs.pdf(m)
  elif opts.Im==3:
    Im = 1
    if opts.eff: obs = np.histogram( toy[:,0], range=mrange, bins=mbins, weights=(1/eff.eff(toy[:,0],toy[:,1])**2) )
    else:        obs = np.histogram( toy[:,0], range=mrange, bins=mbins )
  else: raise RuntimeError('This option for Im', opts.Im, 'is not implemented')

  cw = cow(mrange, gs, gb, Im, obs)

  sws = cw.wk(0,toy[:,0])
  bws = [ cw.wk(i,toy[:,0]) for i in range(1,len(cw.gk)) ]

  if not opts.batch:
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
    fig.savefig('plots/'+opts.oname+'mfunc.pdf')

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
    fig.savefig('plots/'+opts.oname+'wts.pdf')

# do the classic sweight thing
else:

  # fit the mass
  def nll(ns,nb,mu,sg,lb):
    spdf = mynorm(*mrange,mu,sg)
    bpdf = myexp(*mrange,lb)
    return ns + nb - np.sum(np.log( ns*spdf.pdf(toy[:,0]) + nb*bpdf.pdf(toy[:,0]) ) )

  mi = Minuit(nll,ns=nsig,nb=nbkg,mu=5280,sg=30,lb=400,errordef=Minuit.LIKELIHOOD,pedantic=False)
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

  if not opts.batch:
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    ax.errorbar(mhc, mhw, mhw**0.5, fmt='ko', label='Toy Data')
    ax.plot(m, s, 'g--', label='Fitted S pdf')
    ax.plot(m, b, 'r--', label='Fitted B pdf')
    ax.plot(m, s+b, 'b-', label='Fitted S+B pdf')
    ax.set_xlabel('Mass [GeV]')
    ax.set_ylabel('Events')
    ax.legend()
    fig.savefig('plots/'+opts.oname+'mfit.pdf')
    fig.tight_layout()

  # now do the classic sweighting
  pdfs = [ mynorm(*mrange, mi.values['mu'], mi.values['sg']),
           myexp(*mrange, mi.values['lb']) ]
  ylds = [ mi.values['ns'], mi.values['nb'] ]

  sw = SWeight( toy[:,0], pdfs=pdfs, yields=ylds, discvarranges=(mrange,), compnames=('sig','bkg'), method='summation' )

  # plot it
  if not opts.batch:
    fig, ax = plt.subplots(1, 1, figsize=(8,6) )
    sw.makeWeightPlot(ax)
    ax.set_xlabel('Mass [GeV]')
    ax.set_ylabel('sWeight')
    fig.tight_layout()
    fig.savefig('plots/'+opts.oname+'wts.pdf')

  # save weights
  sws = sw.getWeight(0, toy[:,0])
  bws = sw.getWeight(1, toy[:,0])

## NOW FIT BACK WITH OUR WEIGHTS FROM WHATEVER METHOD ##

outf = open('res/'+opts.oname+'.log','w')

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
  effs = eff.eff(toy[:,0],toy[:,1])
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

printopts(opts,outf)

outf.close()

# and plot it
thc, thw, thw2 = hist(toy[:,1], range=trange, bins=tbins, weights=sws)
t = np.linspace(*trange,200)
tN = np.sum(sws)*(trange[1]-trange[0])/tbins

if not opts.batch:
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
  fig.savefig('plots/'+opts.oname+'tfit.pdf')

  plt.show()
