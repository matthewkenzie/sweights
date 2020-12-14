import os
from toy import toy
from cow import cow
import pandas as pd
from iminuit import Minuit
import matplotlib.pyplot as plt
from scipy.stats import expon, norm, uniform
from scipy.integrate import quad
import numpy as np

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-g','--generate', default=False, action="store_true" , help='Regenerate the toy')
parser.add_argument('-s','--size'    , default=10000, type=int            , help='Toy sample size')
parser.add_argument('-e','--trueeff' , default='1'  , type=str            , help='True efficiency model 1/flat, 2/fact, 3/nonfact')
parser.add_argument('-b','--truebfac', default=True , action="store_false", help='True background factorisies (default True)')
parser.add_argument('-i','--im'      , default='1'  , type=str            , help='I(m) model used for moo cows. Can be 1/flat, 2/truth, 3/obs, 4/model')
parser.add_argument('-B','--imB'     , default=0    , type=int            , help='Number of background polynomials in moo')
opts = parser.parse_args()

trueffopts = ['1','2','3','flat','fact','nonfact']
if opts.trueeff not in trueffopts:
  raise RuntimeError('True efficiency must be one of', trueffopts)

if opts.trueeff=='flat': opts.trueeff=1
if opts.trueeff=='fact': opts.trueeff=2
if opts.trueeff=='nonfact': opts.trueeff=3
opts.trueeff = int(opts.trueeff)

imopts = [ '1','2','3','flat','uniform','truth','obs','observed','model' ]
if opts.im not in imopts:
  raise RuntimeError('I(m) must be one of', imopts)

if opts.im=='flat' or opts.im=='uniform': opts.im = 1
if opts.im=='obs' or opts.im=='observed': opts.im = 2
if opts.im=='model': opts.im=3
opts.im = int(opts.im)

if opts.imB!=0: raise RuntimeError('Not yet implemented in moo cow')

toyfname = 'toys/toy'
if opts.trueeff==1: toyfname += '_eflat'
if opts.trueeff==2: toyfname += '_efact'
if opts.trueeff==3: toyfname += '_enonfact'
if opts.truebfac: toyfname += '_bfact'
else: toyfname += '_bnonfact'
toyfname += '_{0}.pkl'.format(opts.size)

if opts.generate or not os.path.exists(toyfname):
  print( toyfname )
  #t = toy( eff=opts.trueeff, bfact=bfact )
  #t.generate( size=size, fname=toyfname )

bins = 50
mrange = (5000,5600)
trange = (0,10)
h0 = expon(0,2)

data = pd.read_pickle( toyfname )

sig = norm(5280,30)
sn  = np.diff( sig.cdf(mrange) )
bkg = expon(5000,400)
bn  = np.diff( bkg.cdf(mrange) )
unf = uniform(5000,5600)
un  = np.diff( unf.cdf(mrange) )
g0  = lambda m: sig.pdf(m) / sn
g1  = lambda m: bkg.pdf(m) / bn
rho = lambda m: 0.5*g0(m) + 0.5*g1(m)
fold = lambda m: (m-mrange[0])/(mrange[1]-mrange[0])
pn  = lambda n, m: (n+1)*fold(m)**n
p0  = lambda m: fold(m)**0 / np.diff(mrange)
p1  = lambda m: 2*fold(m) / np.diff(mrange)
p2  = lambda m: 3*fold(m)**2 / np.diff(mrange)
p3  = lambda m: 4*fold(m)**3 / np.diff(mrange)
p4  = lambda m: 5*fold(m)**4 / np.diff(mrange)
p5  = lambda m: 6*fold(m)**5 / np.diff(mrange)
I1  = lambda m: unf.pdf(m) / un
fval = None
ferr = None

def timepdf(lambd,x):
  b = expon(trange[0], lambd)
  bn = np.diff( b.cdf(trange) )
  return b.pdf(x) / bn

def cowit(gs, gb, Im, obs):
  mycow = cow(mrange=mrange, gs=gs, gb=gb, Im=Im, obs=obs)
  print( mycow.Wkl() )
  print( mycow.Akl() )
  # fit the weighted data
  wts = mycow.wk(0,data['mass'])
  def wnll(lambd):
    b = expon(trange[0], lambd)
    bn = np.diff( b.cdf(trange) )
    return -np.sum( wts * ( b.logpdf( data['time'] ) - np.log(bn) ) )

  mit = Minuit( wnll, lambd=2, errordef=0.5, pedantic=False )
  mit.migrad()
  mit.hesse()
  import sys
  sys.path.append("/Users/matt/Scratch/stats/sweights")
  from CovarianceCorrector import cov_correct
  cov = cov_correct(timepdf, data['time'], wts, mit.np_values(), mit.np_covariance(), verbose=False)
  fval = mit.values['lambd']
  ferr = cov[0,0]**0.5
  return mycow

# set up a grid-spec
# leave space for widgets
fig = plt.figure(figsize=(15,8))
gs = fig.add_gridspec(2,5)
ax1 = fig.add_subplot(gs[0,0:2])
ax2 = fig.add_subplot(gs[1,0:2])
ax3 = fig.add_subplot(gs[0,2:4])
ax4 = fig.add_subplot(gs[1,2:4])

from matplotlib.widgets import RadioButtons
axw = plt.axes([0.8,0.8,0.1,0.1])
ch = RadioButtons(axw, ('I(m)=1','I(m)=rho(m)', 'I(m)=q(m)'), (False,False,False) )

def plotit(mycow=None):
  # plot the component pdfs used for the weights
  m = np.linspace(*mrange,200)
  if mycow:
    ax1.plot( m, mycow.gs(m), 'b-', label='signal')
    for i, gb in enumerate(mycow.gb):
      label = 'background'
      if len(mycow.gb)>1: label += ' {0}'.format(i)
      ax1.plot( m, gb(m), 'r-', label=label)
    ax1.legend()
  ax1.set_xlabel('mass')
  ax1.set_ylabel('probability')

  # plot the weights
  if mycow:
    sw = mycow.wk(0,m)
    bw = mycow.wk(1,m)
    ax2.plot( m, sw, label='signal' )
    ax2.plot( m, bw, label='background' )
    ax2.plot( m, sw+bw, label='sum' )
  ax2.set_xlabel('mass')
  ax2.set_ylabel('weight')

  # plot the weighted data
  if mycow:
    wts = mycow.wk(0,data['mass'])
    w, xe = np.histogram( data['time'], bins=bins, range=trange, weights=wts )
    cx = 0.5 * (xe[1:] + xe[:-1] )
    ax3.errorbar( cx, w, w**0.5, fmt='bx', label='sCOW weigted data' )
    t = np.linspace(*trange,200)
    pnorm = np.sum(wts)*np.diff(trange)/bins
    ax3.plot( t, pnorm*h0.pdf(t), 'b-', label='True $h_0(t)$ distribution' )
    ax3.plot( t, pnorm*timepdf(fval,t), 'r--', label='Fitted $h_0(t)$ distribution' )
    ax3.set_yscale('log')
    ylim = ax3.get_ylim()
    ax3.set_ylim( 1, ylim[1] )
  ax3.set_xlabel('time')
  ax3.set_ylabel('weighted events')

  if mycow:
    truen = len(data[data['ctrl']==0])
    sum_w = np.sum(wts)
    err_w = np.sum(wts**2)**0.5
    #print(truen, truen**0.5, sum_w, err_w)
    ax3.text(0.32,0.9,'$\lambda = {:.2f} \pm {:.2f}$'.format( fval, ferr ), transform=ax3.transAxes )
    ax3.text(0.6,0.7,'$\sum w = {:.2f} \pm {:.2f}$'.format( sum_w,err_w ), transform=ax3.transAxes )
    ax3.text(0.6,0.62,'True $N_{{s}} = {{{0}}}$'.format( len(data[data['ctrl']==0]) ), transform=ax3.transAxes )
    ax3.legend()

  # plot the observed data with Im
  if mycow:
    w, xe = np.histogram( data['mass'], bins=bins, range=mrange )
    cx = 0.5 * (xe[1:] + xe[:-1] )
    ax4.errorbar( cx, w, w**0.5, fmt='bx', label=r'Observed data')
    pnorm = np.sum(w)*np.diff(mrange)/bins
    ax4.plot( m, pnorm*mycow.Im(m), 'r-', label=r'$I(m)$ model' )
    ax4.legend()
  ax4.set_xlabel('mass')
  ax4.set_ylabel('events')
  fig.tight_layout()
  #fig.savefig('plots/cow%d.pdf'%ctype)


cGs = g0
cGb = [p0,p1,p2]
cIm = I1
cObs = None
plotit(None)

def wfunc(label):
  if label == 'I(m)=1':
    cIm = I1
    cObs = None
  if label == 'I(m)=rho(m)':
    cIm = rho
    cObs = None
  if label == 'I(m)=q(m)':
    cIm = 1
    cObs = np.histogram( data['mass'], bins=bins )

  mycow = cowit( gs=cGs, gb=cGb, Im=cIm, obs=cObs )
  plotit(mycow)

# define a widget for g0 vals
ch.on_clicked(wfunc)
plt.show()


