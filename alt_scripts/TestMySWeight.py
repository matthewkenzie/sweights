import numpy as np
from scipy.stats import norm, expon
import matplotlib.pyplot as plt
import boost_histogram as bh
import os
os.system('mkdir -p figs')

np.random.seed(1)  # fix seed

# true parameters for signal and background
truth_n_sig = 500
truth_n_bkg = 5000

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

## generate some toy data
truth_n_tot = truth_n_bkg + truth_n_sig
data = np.empty( (truth_n_tot,2) )

# fill sig vals
nsig = 0
while nsig < truth_n_sig:
    mval = sig_pdf_mass.rvs(1)
    tval = sig_pdf_time.rvs(1)
    if mval > mrange[1] or mval < mrange[0] or tval > trange[1] or tval < trange[0]: continue
    data[nsig,0] = mval
    data[nsig,1] = tval
    nsig += 1

# fill bkg vals
nbkg = 0
while nbkg < truth_n_bkg:
    mval = bkg_pdf_mass.rvs(1)
    tval = bkg_pdf_time.rvs(1)
    if mval > mrange[1] or mval < mrange[0] or tval > trange[1] or tval < trange[0]: continue
    data[nsig+nbkg,0] = mval
    data[nsig+nbkg,1] = tval
    nbkg += 1

# check events are all in range
subs = ( mrange[0] < data[:,0] ) & ( mrange[1] > data[:,0] ) & ( trange[0] < data[:,1] ) & ( trange[1] > data[:,1] )
assert( len(subs) == len(data) )

# now fit the data
from iminuit import Minuit

# extended likelihood
def nll(n_sig, n_bkg, mu, sigma, lambd):
    s = norm(mu, sigma)
    b = expon(5000, lambd)
    # normalisation factors are needed for pdfs, since x range is restricted
    sn = np.diff(s.cdf(mrange))
    bn = np.diff(b.cdf(mrange))
    no = n_sig + n_bkg
    return no - np.sum(np.log(s.pdf(data[:,0]) / sn * n_sig + b.pdf(data[:,0]) / bn * n_bkg))


mi = Minuit(nll,
            n_sig=truth_n_sig, n_bkg=truth_n_bkg,
            mu=truth_sig_m[0], sigma=truth_sig_m[1], lambd=truth_bkg_m[1],
            errordef=Minuit.LIKELIHOOD,
            pedantic=False)

mi.migrad()
mi.hesse()

print(mi.get_param_states())

par = mi.np_values()
cov = mi.np_covariance()

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

# bin data
w, xe = np.histogram(data[:,0], bins=50, range=mrange)
cx = 0.5 * (xe[1:] + xe[:-1])
# bin width to normalise mass_pdf for plotting
mass_pdfnorm = (mrange[1]-mrange[0])/50

plt.errorbar( cx, w, w**0.5, fmt='ko')
plt.plot( cx, mass_pdfnorm*mass_pdf(cx, bonly=True), 'r--')
plt.plot( cx, mass_pdfnorm*mass_pdf(cx), 'b-')
plt.xlabel('mass')
plt.savefig('figs/mass_fit.pdf')

# now do the sweighting stuff

from MySWeightClass import SWeight

# set up the numpy/scipy format of pdfs and yields
pdfs = [ norm( mi.values['mu'], mi.values['sigma'] ) ,
         expon( 5000, mi.values['lambd'] )
       ]
yields = [ mi.values['n_sig'], mi.values['n_bkg'] ]

# set up the roofit format of pdfs and yields
import ROOT as r
rf_mass = r.RooRealVar('mass','mass',*mrange)
rf_mean = r.RooRealVar('mean','mean', mi.values['mu'] )
rf_sigma = r.RooRealVar('sigma','sigma', mi.values['sigma'] )
rf_lambd = r.RooRealVar('lambd','lambd', -1./mi.values['lambd'] )
rfpdfs = [ r.RooGaussian( 'gaus','gaus',rf_mass, rf_mean, rf_sigma ), r.RooExponential( 'expo','expo',rf_mass,rf_lambd) ]

sweighters = []
for i, meth in enumerate(['summation','integration','refit','subhess','tsplot','roofit']):
  alphas = None
  if meth=='subhess':
    inv_cov = np.linalg.inv(cov)
    alphas = np.linalg.inv( inv_cov[:2, :2] )

  if meth=='roofit':
    sw = SWeight( data[:,0], pdfs=rfpdfs, yields=yields, discvarranges=(mrange,), method=meth, compnames=('sig','bkg'), alphas=alphas, rfobs=[rf_mass] )
  else:
    sw = SWeight( data[:,0], pdfs=pdfs, yields=yields, discvarranges=(mrange,), method=meth, compnames=('sig','bkg'), alphas=alphas )

  sweighters.append(sw)

# plot the weight distributions
fig, ax = plt.subplots(3, 3, figsize=(18, 12))

for i, sw in enumerate(sweighters):

  plax = ax[ i%2, int(i/2) ]
  sw.makeWeightPlot(plax)
  plax.set_title(sw.method)

  x = np.linspace(*mrange,100)
  ax[2,1].plot( x, sweighters[0].getWeight(0,x) - sw.getWeight(0, x), label=sw.method )

ax[2,1].legend()
fig.tight_layout()
fig.savefig('figs/sweights.pdf')

print('#### NOW FIT BACK ####')
## now fit back the weighted data to get the true parameter
global wfunc

def expnll(lambd):
    b = expon(0, lambd)
    # normalisation factors are needed for time_pdfs, since x range is restricted
    bn = np.diff(b.cdf(trange))
    return -np.sum( wfunc(0,data[:,0]) * np.log( b.pdf(data[:,1]) / bn ) )

def timepdf(lambd, x):
    b = expon(0,lambd)
    bn = np.diff(b.cdf(trange))
    return b.pdf(x) / bn

fig, ax = plt.subplots(2,3, figsize=(18,8))

for i, sw in enumerate(sweighters):

  wfunc = sw.getWeight

  mi = Minuit( expnll, lambd=truth_sig_t[1], limit_lambd=(1,3), errordef=Minuit.LIKELIHOOD, pedantic=False )
  mi.migrad()
  mi.hesse()
  fitted_value = mi.values['lambd']
  print( 'Fitted back lambd = {:12.10f}'.format(fitted_value), 'for method', sw.method )

  # make a weighted histogram for time values with sweights
  nbins = 50
  whist = bh.Histogram( bh.axis.Regular(nbins,*trange), storage=bh.storage.Weight())
  whist.fill( data[:,1], weight=sw.getWeight(0,data[:,0]) )

  plax = ax[ i%2, int(i/2) ]
  plax.errorbar( whist.axes[0].centers, whist.view().value, whist.view().variance**0.5, fmt='ko' )
  time_pdfnorm = (trange[1]-trange[0])/nbins
  plax.plot( whist.axes[0].centers, np.sum(sw.getWeight(0,data[:,0]))*time_pdfnorm*timepdf(fitted_value, whist.axes[0].centers), 'b-' , label='Fitted')
  plax.plot( whist.axes[0].centers, np.sum(sw.getWeight(0,data[:,0]))*time_pdfnorm*timepdf(truth_sig_t[1], whist.axes[0].centers), 'r--', label='True' )
  plax.set_title( sw.method )
  plax.legend()

fig.tight_layout()
fig.savefig('figs/time_fits.pdf')

### SAVE THE RESULTS ###
import pandas as pd
df = pd.DataFrame(data, columns=['mass','time'])
for sw in sweighters:
  for icomp in range(sw.ncomps):
    df['w%s_%s'%(sw.compnames[icomp],sw.method)] = sw.getWeight(icomp, data[:,0])
df.to_pickle("sweights.pkl")

print(df)

#plt.show()
