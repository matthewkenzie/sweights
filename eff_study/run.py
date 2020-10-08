## NOTES ##
# total pdf (no eff)
# f(m,t) = z gs(m)hs(t) + (1-z) gb(m)hb(t)

# define g(m) as just the mass part of this
# g(m) = int f(m,t) dt = z gs(m) + (1-z) gb(m)

# eff pdf
# e(m,t)

# we define the eff and pdf product as
# f'(m,t) = e(m,t) f(m,t) / Ne
# where Ne is such that f'(m,t) integrates to unity
# so that
# f'(m,t) = e'(m,t)f(m,t)

## pass arguments
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-n','--nevents', default=2500 , type=int           , help='Total number of events to generate')
parser.add_argument('-r','--regen'  , default=False, action="store_true", help='Regenerate the toy')
parser.add_argument('-f','--refit'  , default=False, action="store_true", help='Rerun the weighted fit')
parser.add_argument('-w','--rewht'  , default=False, action="store_true", help='Recompute the weights')
parser.add_argument('-s','--seed'   , default=1    , type=int           , help='Change the seed')
parser.add_argument('-b','--batch'  , default=False, action="store_true", help='Run in batch mode (don\'t make plots etc.)')
parser.add_argument('-a','--all'    , default=False, action="store_true", help='Rerun all')
parser.add_argument('-p','--poisson', default=False, action="store_true", help='Poisson vary total number of events in toy')
parser.add_argument('-d','--dir'    , default='figs'                    , help='Directory to store plots in')
parser.add_argument('-D','--details', default=False, action="store_true", help='Rerun all the little details')
opts = parser.parse_args()

if opts.all:
  opts.regen = True
  opts.refit = True
  opts.rewht = True

print('Seed:  ', opts.seed)
print('Events:', opts.nevents)

import numpy as np
from scipy.stats import norm, expon
from scipy.integrate import quad, nquad
from scipy.linalg import solve
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd
import boost_histogram as bh
from iminuit import Minuit
import pickle

# directory for plots
import os
os.system('mkdir -p %s'%opts.dir)

## SETUP some variables ##
mrange = (5000,5600)
trange = (0,10)
mass = np.linspace(*mrange,100)
time = np.linspace(*trange,100)

# make an efficiency model (it's not really a pdf but just a mapping)
def eff_pdf(m, t, a=1.2, b=0.5, loc=-0.5):
  # get m into range 0-1
  mscl = (m - mrange[0])/(mrange[1]-mrange[0])
  ascl = a + b*mscl
  f = ascl*(t-loc)**(ascl-1.)
  # scale it so that the max is 1. which happens at trange[1], mrange[1]
  mx = (a+b)*(trange[1]-loc)**(a+b-1.)
  return f/mx

# draw the eff map
if not opts.batch:
  print('Drawing efficiency map')
  fig, ax = plt.subplots(1,1,figsize=(6,4))
  x, y = np.meshgrid(mass,time)
  im = ax.contourf(x, y, eff_pdf(x,y) )
  cb = fig.colorbar(im, ax=ax)
  ax.set_xlabel('mass')
  ax.set_ylabel('decay time')
  cb.set_label('efficiency')
  fig.tight_layout()
  fig.savefig('%s/eff_map_2d.pdf'%opts.dir)

  # draw the efficiency mapping in 1D
  fig, ax = plt.subplots(1,2,figsize=(12,4))

  for mval in np.linspace(*mrange, 5):
    ax[0].plot( time, eff_pdf(mval, time), label='$m=%.0f$'%mval )
  ax[0].set_xlabel('decay time')
  ax[0].set_ylabel('efficiency')
  ax[0].legend()

  for tval in np.linspace(*trange, 5):
    ax[1].plot( mass, eff_pdf(mass,tval), label='$t=%.0f$'%tval )
  ax[1].set_xlabel('mass')
  ax[1].set_ylabel('efficiency')
  ax[1].legend()

  fig.tight_layout()
  fig.savefig('%s/eff_maps_1d.pdf'%opts.dir)

## NOW DEFINE THE PDFS gs, gb, hs, hb and z
fit_pars = { 'gs': (5280, 30),
             'gb': (400,),
             'hs': (2,),
             'hb': (0,3),
             'z' : 0.25
            }

def mt_pdf(m, t, mproj=False, tproj=False, sonly=False, bonly=False, **kwargs  ):

  # declare pdfs
  gsm = norm(*kwargs['gs'])
  gbm = expon(mrange[0], *kwargs['gb'])
  hst = expon(trange[0], *kwargs['hs'])
  hbt = norm(*kwargs['hb'])
  zf  = kwargs['z']

  # compute normalisations for our range
  gsmn = np.diff( gsm.cdf(mrange) )
  gbmn = np.diff( gbm.cdf(mrange) )
  hstn = np.diff( hst.cdf(trange) )
  hbtn = np.diff( hbt.cdf(trange) )

  if mproj:
    hstv = 1.
    hbtv = 1.
  else:
    hstv = hst.pdf(t)/hstn
    hbtv = hbt.pdf(t)/hbtn

  if tproj:
    gsmv = 1.
    gbmv = 1.
  else:
    gsmv = gsm.pdf(m)/gsmn
    gbmv = gbm.pdf(m)/gbmn

  sig = zf*gsmv*hstv
  bkg = (1-zf)*gbmv*hbtv

  if sonly: bkg=0
  if bonly: sig=0

  return sig + bkg

def fmt(m=None, t=None, mproj=False, tproj=False, sonly=False, bonly=False):
  return mt_pdf(m,t,mproj,tproj,sonly,bonly,**fit_pars)

# draw the nominal pdf = f(m,t)
if not opts.batch:
  print('Drawing the nominal f(m,t) PDF')
  fig, ax = plt.subplots(1, 3, figsize=(18,4))
  im = ax[0].contourf( x, y, fmt(x,y) )
  cb = fig.colorbar(im, ax=ax[0])
  ax[0].set_xlabel('mass')
  ax[0].set_ylabel('decay time')
  cb.set_label('arbitary units')

  ax[1].plot( mass, fmt(m=mass, mproj=True), 'k-', label='Total PDF - $z g_{s}(m) + (1-z)g_{b}(m)$'  )
  ax[1].plot( mass, fmt(m=mass, mproj=True, sonly=True), 'r--', label='Signal - $z g_{s}(m)$' )
  ax[1].plot( mass, fmt(m=mass, mproj=True, bonly=True), 'b--', label='Background - $(1-z)g_{b}(m)$' )
  ax[1].set_xlabel('mass')
  ax[1].set_ylabel('arbitary units')
  ax[1].legend()

  ax[2].plot( time, fmt(t=time, tproj=True), 'k-', label='Total PDF - $z h_{s}(t) + (1-z)h_{b}(t)$' )
  ax[2].plot( time, fmt(t=time, tproj=True, sonly=True), 'r--', label='Signal - $z h_{s}(t)$' )
  ax[2].plot( time, fmt(t=time, tproj=True, bonly=True), 'b--', label='Background - $(1-z) h_{b}(t)$')
  ax[2].set_xlabel('decay time')
  ax[2].set_ylabel('arbitary units')
  ax[2].legend()

  fig.tight_layout()
  fig.savefig('%s/fmt_pdfs.pdf'%opts.dir)

#### WITHOUT HAVING TO REDO INTEGRATION ####
# Integral of f(m,t) pdf    =  1.0000000000000002 +/- 6.255787141629912e-11
# Integral of e(m,t) map    =  4933.334318308923 +/- 1.223480014124191e-07
# Integral of e(m,t)*f(m,t) =  0.7297241471298931 +/- 7.22546524050093e-11
# Integral of f'(m,t) pdf   =  1.0 +/- 9.901639598157481e-11

if opts.details:
  ## now check the integral
  fmt_integral = nquad( fmt, (mrange,trange) )
  eff_integral = nquad( eff_pdf, (mrange,trange) )
  print('Integral of f(m,t) pdf    = ', fmt_integral[0] , '+/-', fmt_integral[1])
  print('Integral of e(m,t) map    = ', eff_integral[0] , '+/-', eff_integral[1])

  ## check f(m,t) integrates to unity
  assert( abs( fmt_integral[0] - 1. ) < 5*fmt_integral[1] )

  ## now we need to get the Ne normalisation factor such that f' integrates to unity
  fprimenotnorm = lambda m,t: eff_pdf(m,t)*fmt(m,t)
  Ne, NeErr = nquad( tmpf, (mrange,trange) )
  print('Integral of e(m,t)*f(m,t) = ', Ne, '+/-', NeErr)

  ## save the norm factor
  with open('Ne.pkl','wb') as f:
    pickle.dump( (Ne,NeErr), f )

else:
  ## load the norm factor
  with open('Ne.pkl','rb') as f:
    Ne, NeErr = pickle.load(f)

# can now define eps' and f'
def epsprime(m,t):
  return eff_pdf(m,t)/Ne

def fmtprime(m,t, sonly=False, bonly=False):
  return epsprime(m,t)*fmt(m,t,sonly=sonly,bonly=bonly)

## now check the integral of fmtprime is unity
if opts.details:
  fmtprime_integral = nquad(fmtprime, (mrange,trange) )
  print('Integral of f\'(m,t) pdf   = ', fmtprime_integral[0] , '+/-', fmtprime_integral[1])
  assert( abs( fmtprime_integral[0] - 1. ) < 5*fmtprime_integral[1] )

## now useful to define the integral over only one dimension at a time
## there's probably a neater solution but here just define them twice
## with different orderings (for integrating over a different dimension)

def fmtpm(m, t, sonly=False, bonly=False):
  return epsprime(m,t)*fmt(m,t,sonly=sonly,bonly=bonly)

def fmtpt(t, m, sonly=False, bonly=False):
  return epsprime(m,t)*fmt(m,t,sonly=sonly,bonly=bonly)

def fmtp_mproj(m, sonly=False, bonly=False):
  return quad( fmtpt, *trange, args=(m,sonly,bonly))[0]

def fmtp_tproj(t, sonly=False, bonly=False):
  return quad( fmtpm, *mrange, args=(t,sonly,bonly))[0]

## later on we want the inverse of these as well
def fmtoepm(m, t, sonly=False, bonly=False):
  return fmt(m,t,sonly=sonly,bonly=bonly)/epsprime(m,t)

def fmtoept(t, m, sonly=False, bonly=False):
  return fmt(m,t,sonly=sonly,bonly=bonly)/epsprime(m,t)

def fmtoep_mproj(m, sonly=False, bonly=False):
  return quad( fmtoept, *trange, args=(m,sonly,bonly))[0]

def fmtoep_tproj(t, sonly=False, bonly=False):
  return quad( fmtoepm, *mrange, args=(t,sonly,bonly))[0]

## and vectorize them
vec_fmtp_mproj = np.vectorize(fmtp_mproj)
vec_fmtp_tproj = np.vectorize(fmtp_tproj)
vec_fmtoep_mproj = np.vectorize(fmtoep_mproj)
vec_fmtoep_tproj = np.vectorize(fmtoep_mproj)

## make a generic wrapper that can take 1 or 2D
def fmtp(m=None, t=None, mproj=False, tproj=False, sonly=False, bonly=False):
  if mproj:
    return vec_fmtp_mproj(m, sonly=sonly, bonly=bonly)
  if tproj:
    return vec_fmtp_tproj(t, sonly=sonly, bonly=bonly)
  return fmtprime(m, t, sonly=sonly, bonly=bonly)


## now we want to plot the pdf with the efficiency in it
## i.e draw the f'(m,t) pdf
if not opts.batch:
  print('Drawing the product PDF f\'(m,t)')
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.view_init(35,60)
  ax.plot_wireframe( x, y, fmtprime(x,y), colors='k', rstride=4, cstride=4, zorder=3 )
  zlim = (ax.get_zlim()[0] - 0.1*ax.get_zlim()[1], ax.get_zlim()[1])
  xp = ax.contour ( x, y, fmtprime(x,y), zdir='x', offset=mrange[0], cmap=cm.jet, zorder=2)
  yp = ax.contour ( x, y, fmtprime(x,y), zdir='y', offset=trange[0], cmap=cm.jet, zorder=1)
  ax.contourf( x, y, fmtprime(x,y), zdir='z', offset=zlim[0]  , cmap=cm.coolwarm, zorder=0)
  ax.set_xlabel('mass')
  ax.set_ylabel('time')
  ax.set_zlabel('arbitary units')
  ax.set_zlim(zlim[0], zlim[1])
  fig.tight_layout()
  fig.savefig('%s/fmtprime_pdf.pdf'%opts.dir)

  ## this bit is very slow as it attempts to draw the projections which requires lots of numerical integration
  if opts.details:

    print('Drawing the product PDF projections')
    fig, ax = plt.subplots(1, 3, figsize=(18,4))
    im = ax[0].contourf( x, y, fmtp(x,y) )
    cb = fig.colorbar(im, ax=ax[0])
    ax[0].set_xlabel('mass')
    ax[0].set_ylabel('decay time')
    cb.set_label('arbitary units')

    ax[1].plot( mass, fmtp(m=mass, mproj=True), 'k-', label='Total PDF - $\epsilon\'(m,t)(z g_{s}(m) + (1-z)g_{b}(m))$'  )
    ax[1].plot( mass, fmtp(m=mass, mproj=True, sonly=True), 'r--', label='Signal - $\epsilon\'(m,t)(z g_{s}(m))$' )
    ax[1].plot( mass, fmtp(m=mass, mproj=True, bonly=True), 'b--', label='Background - $\epsilon\'(m,t)((1-z)g_{b}(m))$' )
    ax[1].set_xlabel('mass')
    ax[1].set_ylabel('arbitary units')
    ax[1].legend()

    ax[2].plot( time, fmtp(t=time, tproj=True), 'k-', label='Total PDF - $\epsilon\'(m,t)(z h_{s}(t) + (1-z)h_{b}(t))$' )
    ax[2].plot( time, fmtp(t=time, tproj=True, sonly=True), 'r--', label='Signal - $\epsilon\'(m,t)(z h_{s}(t))$' )
    ax[2].plot( time, fmtp(t=time, tproj=True, bonly=True), 'b--', label='Background - $\epsilon\'(m,t)((1-z) h_{b}(t))$')
    ax[2].set_xlabel('decay time')
    ax[2].set_ylabel('arbitary units')
    ax[2].legend()

    fig.tight_layout()
    fig.savefig('%s/fmtp_pdfs.pdf'%opts.dir)

## Now we can generate a toy
# set random seed
np.random.seed(opts.seed)

os.system('mkdir -p toys')
toy_fname = 'toys/toy_n%d_s%d.pkl'%(opts.nevents,opts.seed)

if opts.regen:
  print('Generating a toy with', opts.nevents, 'events')

  # nevents
  truth_n_tot = opts.nevents if not opts.poisson else np.random.poisson(opts.nevents)
  truth_n_sig = int( fit_pars['z']*truth_n_tot )
  truth_n_bkg = truth_n_tot - truth_n_sig

  # declare pdfs
  gsm = norm(*fit_pars['gs'])
  gbm = expon(mrange[0], *fit_pars['gb'])
  hst = expon(trange[0], *fit_pars['hs'])
  hbt = norm(*fit_pars['hb'])

  data = pd.DataFrame(columns=['mass','time','ctrl'])
  data = data.astype( {'mass':float,'time':float,'ctrl':int} )

  # fill sig vals
  nsig = 0
  while nsig < truth_n_sig:
    mval = gsm.rvs()
    tval = hst.rvs()
    if mval > mrange[1] or mval < mrange[0] or tval > trange[1] or tval < trange[0]: continue
    ## now we need to accept-reject based on the efficiency
    if np.random.uniform() > eff_pdf(mval,tval): continue
    data = data.append( {'mass':mval, 'time': tval, 'ctrl': 0}, ignore_index=True )
    nsig += 1

  nbkg = 0
  while nbkg < truth_n_bkg:
    mval = gbm.rvs()
    tval = hbt.rvs()
    if mval > mrange[1] or mval < mrange[0] or tval > trange[1] or tval < trange[0]: continue
    ## now we need to accept-reject based on the efficiency
    if np.random.uniform() > eff_pdf(mval,tval): continue
    data = data.append( {'mass':mval, 'time': tval, 'ctrl': 1}, ignore_index=True )
    nbkg += 1

  # check events are all in range
  data = data[ (data['mass']>mrange[0]) & (data['mass']<mrange[1]) & (data['time']>trange[0]) & (data['time']<trange[1]) ]
  data.to_pickle(toy_fname)

# read the toy from file
else:
  data = pd.read_pickle(toy_fname)
  truth_n_sig = len(data[data['ctrl']==0])
  truth_n_bkg = len(data[data['ctrl']==1])

if not opts.batch: print( data )

## now plot the toy

if not opts.batch:
  # bin the toy data
  mbins = 50
  tbins = 50
  dhist = bh.Histogram( bh.axis.Regular(mbins,*mrange), bh.axis.Regular(tbins,*trange) )
  dhist.fill( data['mass'].to_numpy(), data['time'].to_numpy() )

  # plot the toy data
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.view_init(35,60)
  dX, dY = np.meshgrid( dhist.axes[0].centers, dhist.axes[1].centers )
  ax.scatter( dX, dY, dhist.view().T )
  pnorm = len(data) * (mrange[1]-mrange[0])/mbins * (trange[1]-trange[0])/tbins
  ax.plot_wireframe( x, y, pnorm*fmtprime(x,y), colors='k', rstride=4, cstride=4, zorder=3 )
  fig.savefig('%s/fmtprime_pdf_with_toys.pdf'%opts.dir)

  ## this bit is very slow attempts to draw the projections with the toy
  if opts.details:

    print('Drawing the product PDF projections with the toy')
    fig, ax = plt.subplots(1, 2, figsize=(12,4))

    # mass hist
    dhistx = dhist.project(0)
    ax[0].errorbar( dhistx.axes[0].centers, dhistx.view(), yerr=dhistx.view()**0.5, xerr=0.5*( dhistx.axes[0].edges[1:] - dhistx.axes[0].edges[:-1]), fmt='ko', ms=4. )
    xnorm = len(data) * (mrange[1]-mrange[0])/mbins

    # mass pdf
    ax[0].plot( mass, xnorm*fmtp(m=mass, mproj=True), 'k-', label='Total PDF - $\epsilon\'(m,t)(z g_{s}(m) + (1-z)g_{b}(m))$'  )
    ax[0].plot( mass, xnorm*fmtp(m=mass, mproj=True, sonly=True), 'r--', label='Signal - $\epsilon\'(m,t)(z g_{s}(m))$' )
    ax[0].plot( mass, xnorm*fmtp(m=mass, mproj=True, bonly=True), 'b--', label='Background - $\epsilon\'(m,t)((1-z)g_{b}(m))$' )
    ax[0].set_xlabel('mass')
    ax[0].set_ylabel('arbitary units')
    ax[0].legend()

    # time hist
    dhisty = dhist.project(1)
    ax[1].errorbar( dhisty.axes[0].centers, dhisty.view(), yerr=dhisty.view()**0.5, xerr=0.5*( dhisty.axes[0].edges[1:] - dhisty.axes[0].edges[:-1]), fmt='ko', ms=4. )
    ynorm = len(data) * (trange[1]-trange[0])/tbins

    # time pdf
    ax[1].plot( time, ynorm*fmtp(t=time, tproj=True), 'k-', label='Total PDF - $\epsilon\'(m,t)(z h_{s}(t) + (1-z)h_{b}(t))$' )
    ax[1].plot( time, ynorm*fmtp(t=time, tproj=True, sonly=True), 'r--', label='Signal - $\epsilon\'(m,t)(z h_{s}(t))$' )
    ax[1].plot( time, ynorm*fmtp(t=time, tproj=True, bonly=True), 'b--', label='Background - $\epsilon\'(m,t)((1-z) h_{b}(t))$')
    ax[1].set_xlabel('decay time')
    ax[1].set_ylabel('arbitary units')
    ax[1].legend()

    fig.tight_layout()
    fig.savefig('%s/fmtp_pdfs_with_toy.pdf'%opts.dir)

### Now attempt the weighted fit to estimate zhat and ghats(m)
### ie minimise the sum of studentized residuals

## bin the mass
mbins = min(10, int(opts.nevents/200))
# a normal unweighted histogram so we know the event count in each bin
dmhist = bh.Histogram( bh.axis.Regular(mbins,*mrange) )
dmhist.fill( data['mass'].to_numpy() )
# get my estimate for \hat{N}_eps
N = len(data)
hatNe = N / np.sum( 1./eff_pdf(data['mass'].to_numpy(), data['time'].to_numpy()) )
if not opts.batch:
  print(N, hatNe, Ne)
  print('Estimate of hatNe:', hatNe, 'vs true Ne', Ne)
# a weighted histogram that can return us the sum of weights or the sum of squared weights in each bin
mhist = bh.Histogram( bh.axis.Regular(mbins,*mrange), storage=bh.storage.Weight() )
mhist.fill( data['mass'].to_numpy(), weight=hatNe/eff_pdf(data['mass'].to_numpy(), data['time'].to_numpy()) )

# a function to return us the weighted function for a given value
# looks up the bin. integrates the pdf in the bin
# scales it by the number of events if needed
def weighted_func(m, z, mean, sigma, slope, normalised=False, sonly=False, bonly=False ):
  gsmin = norm(mean,sigma)
  gbmin = expon(mrange[0],slope)
  gsnorm = np.diff( gsmin.cdf(mrange) )[0]
  gbnorm = np.diff( gbmin.cdf(mrange) )[0]
  edges   = mhist.axes[0].edges
  centers = mhist.axes[0].centers
  ibin = mhist.axes[0].index(m)
  lx = mhist.axes[0].edges[ibin]
  hx = mhist.axes[0].edges[ibin+1]
  Ps = np.diff( gsmin.cdf((lx,hx)) )[0] / gsnorm
  Pb = np.diff( gbmin.cdf((lx,hx)) )[0] / gbnorm
  sig = z*Ps
  bkg = (1-z)*Pb
  if sonly: bkg = 0.
  if bonly: sig = 0.
  pred = sig + bkg
  if not normalised: pred *= N
  return pred

# weighted least squares (sum of studentized residuals)
def weighted_least_squares(z, mean, sigma, slope):
  Q = 0
  for ibin, cx in enumerate(mhist.axes[0].centers):
    pred = weighted_func( cx, z, mean, sigma, slope )
    obs = mhist.view().value[ibin]
    var = mhist.view().variance[ibin]
    resid = (pred-obs)**2 / var
    Q += resid
  return Q

mi = Minuit( weighted_least_squares, z=fit_pars['z'], mean=fit_pars['gs'][0], sigma=fit_pars['gs'][1], slope=fit_pars['gb'][0], pedantic=False )

os.system('mkdir -p fitres')
if opts.refit:
  mi.migrad()
  mi.hesse()

  with open('fitres/fitres_n%d_s%d.pkl'%(opts.nevents,opts.seed),'wb') as f:
    pickle.dump( mi.fitarg, f)
  with open('fitres/par_n%d_s%d.pkl'%(opts.nevents,opts.seed),'wb') as f:
    pickle.dump( mi.np_values(), f)
  with open('fitres/cov_n%d_s%d.pkl'%(opts.nevents,opts.seed),'wb') as f:
    pickle.dump( mi.np_covariance(), f)

with open('fitres/fitres_n%d_s%d.pkl'%(opts.nevents,opts.seed),'rb') as f:
  fitarg = pickle.load(f)
with open('fitres/par_n%d_s%d.pkl'%(opts.nevents,opts.seed),'rb') as f:
  par = pickle.load(f)
with open('fitres/cov_n%d_s%d.pkl'%(opts.nevents,opts.seed),'rb') as f:
  cov = pickle.load(f)

mi = Minuit( weighted_least_squares, **fitarg, pedantic=False )
if not opts.batch: print(mi.params)

## Check the result of the weighted fit
## by plotting the shape histogram against the data
if not opts.batch:
  phist = bh.Histogram( bh.axis.Regular(mbins,*mrange) )
  for ibin, cx in enumerate(phist.axes[0].centers):
    phist[ibin] = weighted_func(cx, **mi.values)

  fig, ax = plt.subplots(1,1, figsize=(6,4) )
  ax.errorbar( mhist.axes[0].centers, mhist.view().value, yerr=mhist.view().variance**0.5, xerr=0.5*( mhist.axes[0].edges[1:] - mhist.axes[0].edges[:-1]), fmt='ko', markersize=4. )
  ax.step( phist.axes[0].centers, phist.view(), where='mid' )
  fig.savefig( '%s/ghat_fit.pdf'%opts.dir )

  ## Plot of q_k with comparison to our knowledge of the truth f(m,t)/eps'(m,t)
  ## we could also consider a fit for a q'(m) here to then use Eq.30 but we don't
  ## really a priori know the shape so will leave that for this example
  if opts.details:
    fig, ax = plt.subplots(1,1,figsize=(6,4))
    bin_width = (mrange[1]-mrange[0])/mbins
    q_k = mhist.view().variance/len(data) / bin_width
    ax.plot( mhist.axes[0].centers, q_k, label='q(k)' )
    fmtoep_proj = vec_fmtoep_mproj(mhist.axes[0].centers)
    # this bit is very slow as needs integration
    ax.plot( mhist.axes[0].centers, fmtoep_proj, label='$\int dt f(m,t) / \epsilon\'(m,t)$' )
    ax.set_xlabel('mass')
    ax.set_ylabel('arbitary units')
    ax.legend()
    fig.tight_layout()
    fig.savefig( '%s/qk_hist.pdf'%opts.dir )

## Now extract the weights
def ghat_prod(m, i, j,  z, mean, sigma, slope):
  assert(i==0 or i==1)
  assert(j==0 or j==1)
  gs = norm(mean,sigma)
  gb = expon(mrange[0],slope)
  gi = gs if i==0 else gb
  gj = gs if j==0 else gb
  ginorm = np.diff( gi.cdf(mrange) )[0]
  gjnorm = np.diff( gj.cdf(mrange) )[0]
  prod = lambda m: gi.pdf(m)/ginorm * gj.pdf(m)/gjnorm
  ibin = mhist.axes[0].index(m)
  lx = mhist.axes[0].edges[ibin]
  hx = mhist.axes[0].edges[ibin+1]
  integral = quad(prod, lx, hx)[0]
  return integral

def Wpij(i,j):
  assert(i==0 or i==1)
  assert(j==0 or j==1)
  s = 0.
  for ibin, cx in enumerate(mhist.axes[0].centers):
    gigj = ghat_prod(cx, i, j, **mi.values)
    s += gigj / mhist.view().variance[ibin]
  bin_width = (mrange[1]-mrange[0])/mbins
  return len(data)*bin_width*s

if opts.rewht:
  # get the W-matrix
  Wxy = np.array( [ [Wpij(0,0), Wpij(0,1)], [Wpij(1,0), Wpij(1,1)] ] )

  # solve for the alpha matrix
  sol = np.identity( len(Wxy) )
  alphas = solve( Wxy, sol, assume_a='pos' )

  print('W-matrix:')
  print('\t', str(Wxy).replace('\n','\n\t '))
  print('alpha-matrix')
  print('\t', str(alphas).replace('\n','\n\t '))

# define a weight function
def sweight(m, t, icomp, z, mean, sigma, slope):
  gs = norm(mean,sigma)
  gb = expon(mrange[0],slope)
  gsnorm = np.diff( gs.cdf(mrange) )[0]
  gbnorm = np.diff( gb.cdf(mrange) )[0]
  alph_vec = alphas[icomp]
  numerator = alph_vec[0]*gs.pdf(m)/gsnorm + alph_vec[1]*gb.pdf(m)/gbnorm
  denominator = (eff_pdf(m,t)/hatNe)*( z*gs.pdf(m)/gsnorm + (1-z)*gb.pdf(m)/gbnorm)
  return numerator / denominator

# define pure m part of weight function
def sweightm(m, icomp, z, mean, sigma, slope):
  gs = norm(mean,sigma)
  gb = expon(mrange[0],slope)
  gsnorm = np.diff( gs.cdf(mrange) )[0]
  gbnorm = np.diff( gb.cdf(mrange) )[0]
  alph_vec = alphas[icomp]
  numerator = alph_vec[0]*gs.pdf(m)/gsnorm + alph_vec[1]*gb.pdf(m)/gbnorm
  denominator =  z*gs.pdf(m)/gsnorm + (1-z)*gb.pdf(m)/gbnorm
  return numerator / denominator

## add the weights to the dataframe
wt_fname = 'toys/toy_n%d_s%d_wts.pkl'%(opts.nevents,opts.seed)
if opts.rewht:
  data.insert(2, 'sw', sweight( data['mass'].to_numpy(), data['time'].to_numpy(), 0, **mi.values ) )
  data.insert(3, 'bw', sweight( data['mass'].to_numpy(), data['time'].to_numpy(), 1, **mi.values ) )

  data.to_pickle(wt_fname)

else:
  data = pd.read_pickle(wt_fname)

if not opts.batch: print(data)

# smooth the weights (just for aesethetics of plotting them)
if not opts.batch:
  sdata = data.sort_values( by=['mass'] )
  swS = savgol_filter( sdata['sw'].to_numpy(), int(len(data)/20)+1, 3)
  bwS = savgol_filter( sdata['bw'].to_numpy(), int(len(data)/20)+1, 3)
  sw = InterpolatedUnivariateSpline( sdata['mass'], swS, k=3 )
  bw = InterpolatedUnivariateSpline( sdata['mass'], bwS, k=3 )

  # plot the raw weights
  fig, ax = plt.subplots(1,1, figsize=(6,4))
  ax.plot( sdata['mass'].to_numpy(), sdata['sw'].to_numpy(), 'r-', label=r'$w_{s}$')
  ax.plot( sdata['mass'].to_numpy(), sdata['bw'].to_numpy(), 'b-', label=r'$w_{b}$')
  ax.plot( sdata['mass'].to_numpy(), sdata['sw'].to_numpy() + sdata['bw'].to_numpy(), 'k-', label=r'$\sum_{i} w_{i}$')
  ax.set_xlabel('mass')
  ax.set_ylabel('weight')
  ax.legend()
  fig.tight_layout()
  fig.savefig('%s/raw_weights.pdf'%opts.dir)

  # plot the smooth weights
  fig, ax = plt.subplots(1,1, figsize=(6,4))
  ax.plot(mass, sw(mass), 'r-', label=r'$w_{s}$')
  ax.plot(mass, bw(mass), 'b-', label=r'$w_{b}$')
  ax.plot(mass, sw(mass)+bw(mass), 'k-', label=r'$\sum_{i} w_{i}$')
  ax.set_xlabel('mass')
  ax.set_ylabel('weight')
  ax.legend()
  fig.tight_layout()
  fig.savefig('%s/smooth_weights.pdf'%opts.dir)

  # plot the mass only dependent weights
  fig, ax = plt.subplots(1,1, figsize=(6,4))
  ax.plot(mass, sweightm(mass,0,**mi.values), 'r-', label=r'$w_{s}$')
  ax.plot(mass, sweightm(mass,1,**mi.values), 'b-', label=r'$w_{b}$')
  ax.plot(mass, sweightm(mass,0,**mi.values)+sweightm(mass,1,**mi.values), 'k-', label=r'$\sum_{i} w_{i}$')
  ax.set_xlabel('mass')
  ax.set_ylabel('weight')
  ax.legend()
  fig.tight_layout()
  fig.savefig('%s/massdep_weights.pdf'%opts.dir)


# fit the weighted control (time) distribution
def tnll(slope):
  tpdf = expon(trange[0], slope)
  tnorm = np.diff( tpdf.cdf(trange) )[0]
  return -np.sum( data['sw'].to_numpy() * np.log( tpdf.pdf( data['time'].to_numpy() ) / tnorm ) )

def timepdf(slope, x):
  tpdf = expon(trange[0],slope)
  tnorm = np.diff( tpdf.cdf(trange) )[0]
  return tpdf.pdf(x)/tnorm

tmi = Minuit(tnll, slope=2, errordef=0.5, pedantic=False)
tmi.migrad()
tmi.hesse()
if not opts.batch: print(tmi.params)
import sys
sys.path.append("/Users/matt/Scratch/stats/sweights")
from CovarianceCorrector import cov_correct
ncov = cov_correct(timepdf, data['time'].to_numpy(), data['sw'].to_numpy(), tmi.np_values(), tmi.np_covariance(), verbose=False)
print('Fitted back: {:6.4f} +/- {:6.4f}'.format(tmi.np_values()[0], ncov[0,0]**0.5))
with open('fitres/slope_n%d_s%d.txt'%(opts.nevents,opts.seed),'w') as f:
  f.write('Fitted back: {:12.10f} +/- {:12.10f}'.format(tmi.np_values()[0], ncov[0,0,]**0.5))

# now draw the weighted control distributions
if not opts.batch:
  fig, ax = plt.subplots(1,1,figsize=(6,4))
  ax.hist( [ data['ctrl'].to_numpy(), data['ctrl'].to_numpy(), data['ctrl'].to_numpy() ], bins=2, range=(0,2), weights=[ data['sw'].to_numpy(), data['bw'].to_numpy(), np.ones( data['sw'].to_numpy().shape ) ], label=['Applying $w\'_{s}$','Applying $w\'_{b}$','Truth'])
  ax.set_xticks( [0.5,1.5] )
  ax.set_xticklabels( ['Signal','Background'] )
  ax.set_xlabel('True Data Category')
  ax.set_ylabel('(weighted) entries')
  ax.legend()
  fig.tight_layout()
  fig.savefig('%s/ctrl_fit.pdf'%opts.dir)

  fig, ax = plt.subplots(1,1,figsize=(6,4))
  tbins = 100
  thist = bh.Histogram( bh.axis.Regular(tbins,*trange), storage=bh.storage.Weight() )
  thist.fill( data['time'].to_numpy(), weight=data['sw'].to_numpy() )
  ax.errorbar( thist.axes[0].centers, thist.view().value, yerr=thist.view().variance**0.5, xerr=0.5*(thist.axes[0].edges[:-1]-thist.axes[0].edges[1:]), fmt='ko', markersize=4., capsize=1., label='sWeighted Data' )
  #ax.plot( time, np.diff(trange)[0]/100 * len(data) * fmt(t=time, tproj=True, sonly=True), label='True PDF' )
  ax.plot( time, np.diff(trange)[0]/100 * thist.sum().value * timepdf( fit_pars['hs'][0], time ), label='True PDF' )
  ax.plot( time, np.diff(trange)[0]/100 * thist.sum().value * timepdf( tmi.values['slope'], time ), label='Fitted PDF' )
  ax.set_xlabel('decay time')
  ax.set_ylabel('weighted entries')
  ax.legend()
  fig.tight_layout()
  fig.savefig('%s/ctrl_tproj_fit.pdf'%opts.dir)
