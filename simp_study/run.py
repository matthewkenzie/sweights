import numpy as np
from scipy.stats import norm, expon
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib import cm

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-n','--nevents', default=2500, type=int, help='Total number of events to generate')
parser.add_argument('-r','--regen', default=False, action="store_true", help='Regenerate the toy')
parser.add_argument('-f','--refit', default=False, action="store_true", help='Refit the toy')
parser.add_argument('-w','--rewht', default=False, action="store_true", help='Recompute the weights')
parser.add_argument('-s','--seed', default=1, type=int, help='Change the seed')
parser.add_argument('-b','--batch', default=False, action="store_true", help='Run in batch mode (don\'t make plots etc.)')
parser.add_argument('-a','--all', default=False, action="store_true", help='Rerun all')
opts = parser.parse_args()

if opts.all:
  opts.regen = True
  opts.refit = True
  opts.rewht = True

print('Seed:  ', opts.seed)
print('Events:', opts.nevents)

import os
os.system('mkdir -p figs')

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


# true parameters for signal and background
truth_n_sig = int(0.2*opts.nevents)
truth_n_bkg = int(0.8*opts.nevents)

# signal mass gauss:(mean, sigma)
truth_sig_m = (5280, 30)
# background mass expo:(loc, scale)
truth_bkg_m = (5000,400,)
# signal time expo:(loc, scale)
truth_sig_t = (0, 2.0,)
# background time gauss:(mean, sigma)
truth_bkg_t = (0., 3.0)

# make the pdfs
sig_pdf_mass = norm(*truth_sig_m)
bkg_pdf_mass = expon(*truth_bkg_m)
sig_pdf_time = expon(*truth_sig_t)
bkg_pdf_time = norm(*truth_bkg_t)

# plot the pdfs
mrange = (5000,5600)
trange = (0,10)
mass = np.linspace(*mrange)
time = np.linspace(*trange)

# get the normalisation for the range
sig_norm_mass = np.diff(sig_pdf_mass.cdf(mrange))
bkg_norm_mass = np.diff(bkg_pdf_mass.cdf(mrange))
sig_norm_time = np.diff(sig_pdf_time.cdf(trange))
bkg_norm_time = np.diff(bkg_pdf_time.cdf(trange))

# make a grid for plots
if not opts.batch:
  fig = plt.figure(figsize=(15,7.5))
  gs = gridspec.GridSpec(2, 3)

  # now make the 2D plots
  x, y = np.meshgrid( mass, time )
  zsig = truth_n_sig*sig_pdf_mass.pdf(x)/sig_norm_mass * truth_n_sig*sig_pdf_time.pdf(y)/sig_norm_time
  zbkg = truth_n_bkg*bkg_pdf_mass.pdf(x)/bkg_norm_mass * truth_n_bkg*bkg_pdf_time.pdf(y)/bkg_norm_time
  ztot = zsig + zbkg

  ax   = fig.add_subplot(gs[0], projection='3d')
  ax.plot_surface(x,y,zsig, cmap=cm.coolwarm)
  ax.set_title('PDF for Signal')
  ax.view_init(50,20)

  ax   = fig.add_subplot(gs[1], projection='3d')
  ax.plot_surface(x,y,zbkg, cmap=cm.coolwarm)
  ax.set_title('PDF for Background')
  ax.view_init(50,20)

  ax   = fig.add_subplot(gs[2], projection='3d')
  ax.plot_surface(x,y,zbkg+zsig, cmap=cm.coolwarm)
  ax.set_title('PDF for Sig + Bkg')
  ax.view_init(50,20)

  # and the 1D plots
  ax = fig.add_subplot(gs[3] )
  ax.plot( mass, truth_n_sig*sig_pdf_mass.pdf(mass)/sig_norm_mass + truth_n_bkg*bkg_pdf_mass.pdf(mass)/bkg_norm_mass , 'k--', label='Both' )
  ax.plot( mass, truth_n_bkg*bkg_pdf_mass.pdf(mass)/bkg_norm_mass, 'b-' , label='Background')
  ax.plot( mass, truth_n_sig*sig_pdf_mass.pdf(mass)/sig_norm_mass, 'r-' , label='Signal' )
  ax.legend()

  ax = fig.add_subplot(gs[4])
  ax.plot( time, truth_n_sig*sig_pdf_time.pdf(time)/sig_norm_time + truth_n_bkg*bkg_pdf_time.pdf(time)/bkg_norm_time, 'k--')
  ax.plot( time, truth_n_bkg*bkg_pdf_time.pdf(time)/bkg_norm_time, 'b-' , label='Background')
  ax.plot( time, truth_n_sig*sig_pdf_time.pdf(time)/sig_norm_time, 'r-' , label='Signal')
  ax.legend()

  fig.tight_layout()
  fig.savefig('figs/true_pdfs.pdf')

# generate a toy
import pandas as pd
np.random.seed(opts.seed)  # fix seed

os.system('mkdir -p toys')
toy_fname = 'toys/toy_n%d_s%d.pkl'%(opts.nevents,opts.seed)

if opts.regen:

  data = pd.DataFrame(columns=['mass','time','ctrl'])

  # fill sig vals
  truth_n_sig = np.random.poisson(truth_n_sig)
  nsig = 0
  while nsig < truth_n_sig:
      mval = sig_pdf_mass.rvs()
      tval = sig_pdf_time.rvs()
      if mval > mrange[1] or mval < mrange[0] or tval > trange[1] or tval < trange[0]: continue
      data = data.append( {'mass': mval, 'time': tval, 'ctrl': 0}, ignore_index=True )
      nsig += 1

  # fill bkg vals
  nbkg = 0
  truth_n_bkg = np.random.poisson(truth_n_bkg)
  while nbkg < truth_n_bkg:
      mval = bkg_pdf_mass.rvs()
      tval = bkg_pdf_time.rvs()
      if mval > mrange[1] or mval < mrange[0] or tval > trange[1] or tval < trange[0]: continue
      data = data.append( {'mass': mval, 'time': tval, 'ctrl': 1}, ignore_index=True )
      nbkg += 1

  data = data.astype({'mass':float,'time':float, 'ctrl':int})
  data.to_pickle(toy_fname)

else:

  data = pd.read_pickle(toy_fname)
  truth_n_sig = len(data[data['ctrl']==0])
  truth_n_bkg = len(data[data['ctrl']==1])

if not opts.batch: print(data)

# plot the toy
if not opts.batch:
  fig, ax = plt.subplots(1, 3, figsize=(14, 4))
  ax[0].hist2d(data['mass'].to_numpy(), data['time'].to_numpy(), bins=(50, 50))
  ax[0].set_xlabel("$m$")
  ax[0].set_ylabel("$t$")
  ax[1].hist((data[data['ctrl']==1]['mass'].to_numpy(), data[data['ctrl']==0]['mass'].to_numpy()), bins=50, stacked=True, label=("Background", "Signal"))
  ax[1].set_xlabel("m")
  ax[1].legend()
  ax[2].hist((data[data['ctrl']==1]['time'].to_numpy(), data[data['ctrl']==0]['time'].to_numpy()), bins=50, stacked=True, label=("Background", "Signal"))
  ax[2].set_xlabel("t")
  ax[2].legend()
  fig.tight_layout()
  fig.savefig('figs/toy_data.pdf')

# fit the toy
from iminuit import Minuit

def nll(n_sig, n_bkg, mu, sigma, lambd):
    s = norm(mu, sigma)
    b = expon(5000, lambd)
    # normalisation factors are needed for pdfs, since x range is restricted
    sn = np.diff(s.cdf(mrange))
    bn = np.diff(b.cdf(mrange))
    no = n_sig + n_bkg
    return no - np.sum(np.log(s.pdf(data['mass'].to_numpy()) / sn * n_sig + b.pdf(data['mass'].to_numpy()) / bn * n_bkg))

mi = Minuit(nll,
            n_sig=truth_n_sig, n_bkg=truth_n_bkg,
            mu=truth_sig_m[0], sigma=truth_sig_m[1], lambd=truth_bkg_m[1],
            errordef=Minuit.LIKELIHOOD,
            pedantic=False)

os.system('mkdir -p fitres')
if opts.refit:

  # free fit
  mi.migrad()
  mi.hesse()

  to_pickle( 'fitres/fitres_n%d_s%d.pkl'%(opts.nevents,opts.seed), mi.fitarg )
  to_pickle( 'fitres/par_n%d_s%d.pkl'%(opts.nevents,opts.seed), mi.np_values() )
  to_pickle( 'fitres/cov_n%d_s%d.pkl'%(opts.nevents,opts.seed), mi.np_covariance() )

  # now fit with shape fixed
  mi.fixed['mu'] = True
  mi.fixed['sigma'] = True
  mi.fixed['lambd'] = True
  mi.migrad()
  mi.hesse()

  to_pickle( 'fitres/fitres_fix_n%d_s%d.pkl'%(opts.nevents,opts.seed), mi.fitarg )
  to_pickle( 'fitres/par_fix_n%d_s%d.pkl'%(opts.nevents,opts.seed), mi.np_values() )
  to_pickle( 'fitres/cov_fix_n%d_s%d.pkl'%(opts.nevents,opts.seed), mi.np_covariance() )

fitarg = read_pickle('fitres/fitres_n%d_s%d.pkl'%(opts.nevents,opts.seed))
par    = read_pickle('fitres/par_n%d_s%d.pkl'%(opts.nevents,opts.seed))
cov    = read_pickle('fitres/cov_n%d_s%d.pkl'%(opts.nevents,opts.seed))

fitarg_fix = read_pickle('fitres/fitres_fix_n%d_s%d.pkl'%(opts.nevents,opts.seed))
par_fix    = read_pickle('fitres/par_fix_n%d_s%d.pkl'%(opts.nevents,opts.seed))
cov_fix    = read_pickle('fitres/cov_fix_n%d_s%d.pkl'%(opts.nevents,opts.seed))

mi = Minuit(nll, **fitarg, pedantic=False)
mi_fix  = Minuit(nll, **fitarg_fix, pedantic=False)
mi_fix.fixed['mu'] = True
mi_fix.fixed['sigma'] = True
mi_fix.fixed['lambd'] = True

print('Free Mass Fit:')
print(mi.params)
print('Yield Only Mass Fit:')
print(mi_fix.params)

# visualise the fitted model
def mass_pdf(m, bonly=False, sonly=False):
    n_sig, n_bkg, mu, sigma, lambd = par
    spdf = norm(mu, sigma)
    bpdf = expon(5000, lambd)

    sn = np.diff(spdf.cdf(mrange))
    bn = np.diff(bpdf.cdf(mrange))

    if sonly: return n_sig * spdf.pdf(m) / sn
    if bonly: return n_bkg * bpdf.pdf(m) / bn
    return n_sig * spdf.pdf(m) / sn + n_bkg * bpdf.pdf(m) / bn

if not opts.batch:
  fig, ax = plt.subplots(1, 1, figsize=(6, 4))
  # bin data
  w, xe = np.histogram(data['mass'].to_numpy(), bins=50, range=mrange)
  cx = 0.5 * (xe[1:] + xe[:-1])
  # bin width to normalise mass_pdf for plotting
  mass_pdfnorm = (mrange[1]-mrange[0])/50

  ax.errorbar( cx, w, w**0.5, fmt='ko')
  ax.plot( cx, mass_pdfnorm*mass_pdf(cx,bonly=True), 'r--')
  ax.plot( cx, mass_pdfnorm*mass_pdf(cx), 'b-')
  ax.set_xlabel('mass')
  fig.tight_layout()
  fig.savefig('figs/mass_fit.pdf')

# compute the weights
wt_fname = 'toys/toy_n%d_s%d_wts.pkl'%(opts.nevents,opts.seed)
methods = ['summation','integration','refit','subhess'] #,'tsplot','roofit']
if opts.rewht:
  import sys
  sys.path.append("/Users/matt/Scratch/stats/sweights")
  from MySWeightClass import SWeight

  # set up the numpy/scipy format of pdfs and yields
  pdfs = [ norm( mi.values['mu'], mi.values['sigma'] ) ,
           expon( 5000, mi.values['lambd'] )
         ]
  yields = [ mi.values['n_sig'], mi.values['n_bkg'] ]

  # set up the roofit format of pdfs and yields
  #import ROOT as r
  #rf_mass = r.RooRealVar('mass','mass',*mrange)
  #rf_mean = r.RooRealVar('mean','mean', mi.values['mu'] )
  #rf_sigma = r.RooRealVar('sigma','sigma', mi.values['sigma'] )
  #rf_lambd = r.RooRealVar('lambd','lambd', -1./mi.values['lambd'] )
  #rfpdfs = [ r.RooGaussian( 'gaus','gaus',rf_mass, rf_mean, rf_sigma ), r.RooExponential( 'expo','expo',rf_mass,rf_lambd) ]

  sweighters = []

  for i, meth in enumerate(methods):
    alphas = None
    if meth=='subhess':
      inv_cov = np.linalg.inv(cov)
      alphas = np.linalg.inv( inv_cov[:2, :2] )

    #if meth=='roofit':
      #sw = SWeight( data[:,0], pdfs=rfpdfs, yields=yields, discvarranges=(mrange,), method=meth, compnames=('sig','bkg'), alphas=alphas, rfobs=[rf_mass] )
    #else:
    sw = SWeight( data['mass'].to_numpy(), pdfs=pdfs, yields=yields, discvarranges=(mrange,), method=meth, compnames=('sig','bkg'), alphas=alphas )

    sweighters.append(sw)

    # plot the weight distributions
    if not opts.batch:
      fig, ax = plt.subplots(1, 1, figsize=(6,4))
      sw.makeWeightPlot(ax)
      ax.set_xlabel('mass')
      ax.set_ylabel('weight')
      fig.tight_layout()
      fig.savefig('figs/wts_%s.pdf'%meth)

  # plot the weight distribution difference
  if not opts.batch:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    x = np.linspace(*mrange,100)
    labels = ['Variant B','Variant A','Variant Ci','Variant Cii']
    for i in [1,0,2,3]:
      sw = sweighters[i]
      ax.plot( x, sweighters[0].getWeight(0,x) - sw.getWeight(0, x), label=labels[i] )

      ax.set_xlabel('mass')
      ax.set_ylabel('$(w^{B}_{s}-w_{s})$')
      ax.legend()
      fig.tight_layout()
      fig.savefig('figs/wts_difference.pdf')

  # save the weights in the frame
  for sw in sweighters:
    data.insert( len(data.columns), 'sw_%s'%sw.method, sw.getWeight(0, data['mass'].to_numpy()) )
    data.insert( len(data.columns), 'bw_%s'%sw.method, sw.getWeight(1, data['mass'].to_numpy()) )

  # save res
  data.to_pickle(wt_fname)

else:
  data = pd.read_pickle(wt_fname)

if not opts.batch: print(data)

# make a table of the weights
table = { 'Method' : ['Truth', 'Fit Result', 'Yield Only Fit Result','Variant A', 'Variant B','Variant C','Variant D'] ,
          'sYield' : [ truth_n_sig, mi.values['n_sig'] , mi_fix.values['n_sig'] ],
          'sError' : [ 0, mi.errors['n_sig'] , mi_fix.errors['n_sig'] ],
          'bYield' : [ truth_n_bkg, mi.values['n_bkg'] , mi_fix.values['n_bkg'] ],
          'bError' : [ 0, mi.errors['n_bkg'] , mi_fix.errors['n_bkg'] ]
        }

for i, meth in enumerate(methods):
  table['sYield'].append( np.sum( data['sw_%s'%meth].to_numpy() ) )
  table['sError'].append( np.sqrt( np.sum( data['sw_%s'%meth].to_numpy()**2 ) ) )
  table['bYield'].append( np.sum( data['bw_%s'%meth].to_numpy() ) )
  table['bError'].append( np.sqrt( np.sum( data['bw_%s'%meth].to_numpy()**2 ) ) )

from tabulate import tabulate
print(tabulate(table, headers="keys", floatfmt=".2f"))
if not opts.batch: print(tabulate(table, headers="keys", tablefmt="latex", floatfmt=".2f"))
with open('fitres/weights_n%d_s%d.txt'%(opts.nevents,opts.seed),'w') as f:
  f.write(tabulate(table, headers="keys", floatfmt=".4f"))

# now fit back the weighted data
import boost_histogram as bh

from scipy.misc import derivative

# make a histogram of the true values
nbins = 50
thist = bh.Histogram( bh.axis.Regular(nbins,*trange) )
thist.fill( data[data['ctrl']==0]['time'].to_numpy() )

# nll for the time fit
# unweighted
def tnll(lambd):
  b = expon(0, lambd)
  bn = np.diff(b.cdf(trange))
  return -np.sum( np.log( b.pdf(data[data['ctrl']==0]['time'].to_numpy()) / bn ) )

# fit the truth
tmi = Minuit( tnll, lambd=truth_sig_t[1], limit_lambd=(1,3), errordef=Minuit.LIKELIHOOD, pedantic=False )
tmi.migrad()
tmi.hesse()
if not opts.batch: print(tmi.params)
truth_fitted_value = tmi.values['lambd']
truth_fitted_error = tmi.errors['lambd']

# weighted
global wmeth
def wnll(lambd):
    global wmeth
    b = expon(0, lambd)
    # normalisation factors are needed for time_pdfs, since x range is restricted
    bn = np.diff(b.cdf(trange))
    return -np.sum( data['sw_%s'%wmeth].to_numpy() * np.log( b.pdf(data['time'].to_numpy()) / bn ) )

# the time pdf
def timepdf(lambd,x):
    b = expon(0,lambd)
    bn = np.diff(b.cdf(trange))
    return b.pdf(x) / bn

labels = ['Variant A','Variant B','Variant Ci','Variant Cii']
fit_back_vals = []
for i, meth in enumerate(methods):
    wmeth = meth

    # do minimisation
    mi = Minuit( wnll, lambd=truth_sig_t[1], limit_lambd=(1,3), errordef=Minuit.LIKELIHOOD, pedantic=False )
    mi.migrad()
    mi.hesse()
    fitted_value = mi.values['lambd']
    fitted_error = mi.errors['lambd']

    # now we do the uncertainty correction
    cov = mi.np_covariance()

    # correction piece
    Djk = np.zeros(cov.shape)
    Djk[0,0] = np.sum( data['sw_%s'%meth].to_numpy()**2 * derivative(timepdf, fitted_value, n=1, args=(data['time'].to_numpy(),))**2 / timepdf(fitted_value, data['time'].to_numpy())**2 )

    # apply the correction
    newcov = cov * Djk * cov.T

    #print( 'Fitted back lambd = {:8.6f} +/- {:8.6f}'.format(fitted_value,fitted_error))
    #print( 'Correc back lambd = {:8.6f} +/- {:8.6f}'.format(fitted_value,newcov[0,0]**0.5))
    fit_back_vals.append( (fitted_value, newcov[0,0]**0.5) )

    # make a weighted histogram
    if not opts.batch:
      nbins = 50
      whist = bh.Histogram( bh.axis.Regular(nbins,*trange), storage=bh.storage.Weight())
      whist.fill( data['time'].to_numpy(), weight=data['sw_%s'%meth].to_numpy() )

      # plot the fits
      fig, ax = plt.subplots(1,1, figsize=(6,4))
      ax.errorbar( whist.axes[0].centers, whist.view().value, whist.view().variance**0.5, fmt='bo', label='sWeighted' )
      ax.errorbar( thist.axes[0].centers, thist.view(), thist.view()**0.5, fmt='ro', label='True' )
      time_pdfnorm = (trange[1]-trange[0])/nbins
      ax.plot( whist.axes[0].centers, np.sum(data['sw_%s'%meth].to_numpy())*time_pdfnorm*timepdf(fitted_value, whist.axes[0].centers), 'b-' , label='Fitted')
      ax.plot( whist.axes[0].centers, np.sum(data['sw_%s'%meth].to_numpy())*time_pdfnorm*timepdf(truth_sig_t[1], whist.axes[0].centers), 'r--', label='True' )
      ax.set_xlabel('decay time')
      ax.set_ylabel('weighted events')
      ax.legend()
      fig.savefig('figs/%s_time.pdf'%meth)

# now do we the full fit in 2D to see the result
# extended likelihood
def nll2d(n_sig, n_bkg, mu, sigma, lambd, slambdt, bmut, bsigmat):
    sm = norm(mu, sigma)
    bm = expon(5000, lambd)
    st = expon(0,slambdt)
    bt = norm(bmut,bsigmat)

    # normalisation factors are needed for pdfs, since x range is restricted
    smn = np.diff(sm.cdf(mrange))
    bmn = np.diff(bm.cdf(mrange))
    stn = np.diff(st.cdf(trange))
    btn = np.diff(bt.cdf(trange))
    no = n_sig + n_bkg
    return no - np.sum(np.log(sm.pdf(data['mass'].to_numpy()) / smn * st.pdf(data['time'].to_numpy()) / stn * n_sig
                            + bm.pdf(data['mass'].to_numpy()) / bmn * bt.pdf(data['time'].to_numpy()) / btn * n_bkg))


# fit
mi = Minuit(nll2d,
            n_sig=truth_n_sig, n_bkg=truth_n_bkg,
            mu=truth_sig_m[0], sigma=truth_sig_m[1], lambd=truth_bkg_m[1],
            slambdt=truth_sig_t[1], bmut=truth_bkg_t[0], bsigmat=truth_bkg_t[1],
            errordef=Minuit.LIKELIHOOD,
            pedantic=False)

mi.migrad()
mi.hesse()
if not opts.batch: print( mi.params )

# make nice table
table = { 'Method' : ['2D Fit Result','1D Truth Fit Result', 'Variant A', 'Variant B','Variant C','Variant D'] ,
          'Slope Value' : [ mi.values['slambdt'], truth_fitted_value ] + [ x[0] for x in fit_back_vals ],
          'Slope Error' : [ mi.errors['slambdt'], truth_fitted_error ] + [ x[1] for x in fit_back_vals ],
        }
print(tabulate(table, headers="keys", floatfmt=".3f"))
with open('fitres/slope_n%d_s%d.txt'%(opts.nevents,opts.seed),'w') as f:
  f.write(tabulate(table, headers="keys", floatfmt=".6f"))

