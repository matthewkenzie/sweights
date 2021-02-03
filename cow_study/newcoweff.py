from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-t','--ntoys', default=0, type=int, help='Run this number of toys')
parser.add_argument('-s','--nsig' , default=1000, type=int, help='Signal evs per toy')
parser.add_argument('-b','--nbkg' , default=1000, type=int, help='Background evs per toy')
parser.add_argument('-p','--poiss', default=False, action="store_true", help='Poisson flucutate evs in toys')
opts = parser.parse_args()

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import pickle

from scipy.integrate import quad, nquad
from scipy.stats import truncnorm, truncexpon
from scipy.interpolate import interp1d

from iminuit import Minuit
from tqdm import tqdm

def hist(vals, range=None, bins=25, weights=None):
  w, xe = np.histogram(vals, range=range, bins=bins, weights=weights)
  cx = 0.5 * (xe[1:] + xe[:-1])
  return cx, w

def mynorm(xmin,xmax,mu,sg):
  a, b = (xmin-mu)/sg, (xmax-mu)/sg
  return truncnorm(a,b,mu,sg)

def myexp(xmin,xmax,lb):
  return truncexpon( (xmax-xmin)/lb, xmin, lb )

class bkgweffmodel:
  def __init__(self,mrange, trange, lb, mu, sg, slb, smu, ssg, ea, eb, el, cache='load'):
    print('Initialising background model')
    self.mrange = mrange
    self.trange = trange
    self.lb     = lb
    self.mu     = mu
    self.sg     = sg
    self.slb    = slb
    self.smu    = smu
    self.ssg    = ssg
    self.ea     = ea
    self.eb     = eb
    self.el     = el
    self.pars = { 'mrange': self.mrange,
                  'trange': self.trange,
                  'lb'    : self.lb,
                  'mu'    : self.mu,
                  'sg'    : self.sg,
                  'slb'   : self.slb,
                  'smu'   : self.smu,
                  'ssg'   : self.ssg,
                  'ea'    : self.ea,
                  'eb'    : self.eb,
                  'el'    : self.el
                 }
    self.N      = 1
    self.eN     = 1
    # get normalisations
    self.N, self.Nerr   = nquad( self.pdf, (self.mrange, self.trange) )
    self.eN, self.eNerr = nquad( self.eff, (self.mrange, self.trange) )
    # get the majorant for toys
    f = lambda m, t: -self.pdf(m,t)
    mi = Minuit(f, m=self.mrange[0], t=self.trange[0], limit_m=self.mrange, limit_t=self.trange, pedantic=False)
    mi.migrad()
    self.maj = -mi.fval
    print(' --> Majorant of {:4.2g} found at ({:7.2f},{:4.2f})'.format(self.maj, mi.values['m'], mi.values['t']))
    # vectorise projection pdfs
    self.pdfm   = np.vectorize(self._pdfm)
    self.pdft   = np.vectorize(self._pdft)
    self.effm   = np.vectorize(self._effm)
    self.efft   = np.vectorize(self._efft)
    if isinstance(cache,str):
      self.load_caches()
    elif isinstance(cache,int):
      self.make_caches(cache)
      self.load_caches()

  def make_caches(self, points):
    cache_dir = 'cache/bkgweffmodel'
    print('Caching into', cache_dir)
    os.system(f'mkdir -p {cache_dir}')
    # save pars
    with open(f'{cache_dir}/pars.pkl','wb') as f:
      pickle.dump(self.pars,f)
    # compute arrays and save them
    m = np.linspace(*self.mrange,points)
    t = np.linspace(*self.trange,points)
    fm = self.pdfm(m)
    ft = self.pdft(t)
    em = self.effm(m)
    et = self.efft(t)
    np.savez(f'{cache_dir}/arrs.npz', m=m, fm=fm, em=em, t=t, ft=ft, et=et)

  def load_caches(self):
    cache_dir = 'cache/bkgweffmodel'
    # read pars
    assert( os.path.exists(f'{cache_dir}/pars.pkl') )
    with open(f'{cache_dir}/pars.pkl','rb') as f:
      pars = pickle.load(f)
      for name, val in pars.items():
        assert( self.pars[name] == val )
    # read arrays
    npzfile = np.load(f'{cache_dir}/arrs.npz')
    m = npzfile['m']
    t = npzfile['t']
    fm = npzfile['fm']
    ft = npzfile['ft']
    em = npzfile['em']
    et = npzfile['et']
    self.pdfm = interp1d(m,fm,kind='quadratic')
    self.pdft = interp1d(t,ft,kind='quadratic')
    self.effm = interp1d(m,em,kind='quadratic')
    self.efft = interp1d(t,et,kind='quadratic')

  def eff(self, m, t):
    # get m into range 0-1
    mscl = ( m - self.mrange[0] ) / ( self.mrange[1] - self.mrange[0] )
    ascl = self.ea + self.eb*mscl
    f     = ascl * ( t - self.el ) ** ascl
    # scale it so that the max is 1. which happens at trange[1], mrange[1]
    mx = (self.ea+self.eb) * ( trange[1] - self.el ) ** (self.ea+self.eb)
    return f/mx

  def pdf(self, m, t, mproj=False, tproj=False):
    ############
    # eff part #
    ############

    eff = self.eff(m,t)

    ############
    # pdf part #
    ############

    # lambda for mass part
    dt = 2 * ( t - self.trange[0] ) / ( self.trange[1] - self.trange[0] ) - 1
    flb = self.lb + self.slb*dt
    # mu, sigma for time part
    dm = 2 * ( m - self.mrange[0] ) / ( self.mrange[1] - self.mrange[0] ) - 1
    fmu = self.mu + self.smu*dm
    fsg = self.sg + self.ssg*dm

    # pdfs
    mpdf = myexp(*self.mrange, flb)
    tpdf = mynorm(*self.trange, fmu, fsg)

    if mproj: return eff*mpdf.pdf(m) / self.eN
    if tproj: return eff*tpdf.pdf(t) / self.eN
    return eff*mpdf.pdf(m)*tpdf.pdf(t) / self.N

  def _pdfm(self,m):
    f = lambda t: self.pdf(m,t)
    return quad(f,*trange)[0]

  def _pdft(self,t):
    f = lambda m: self.pdf(m,t)
    return quad(f,*mrange)[0]

  def _effm(self,m):
    f = lambda t: self.eff(m,t)
    return quad(f,*trange)[0]

  def _efft(self,t):
    f = lambda m: self.eff(m,t)
    return quad(f,*mrange)[0]

  # this is the very slow accept/reject method
  def generate(self, size=1, progress=None, save=None, seed=None):
    if progress is None:
      progress = False if size<10 else True

    if seed is not None:
      np.random.seed(seed)

    vals = np.empty((size,2))
    ngen = 0
    if progress: bar = tqdm(desc='Generating Background', total=size)
    while ngen < size:
      accept = False
      m = None
      t = None
      while not accept:
        m = np.random.uniform(*mrange)
        t = np.random.uniform(*trange)
        p = self.pdf(m,t)
        if p>self.maj:
          print('Warning found p(m,t) =', p, 'at', m, ',', t, 'is larger than majorant', self.maj, '. Having to update the majorant')
          self.maj = p
        h = np.random.uniform(0,self.maj)
        if h<p: accept=True

      vals[ngen] = (m,t)
      if progress: bar.update()
      ngen += 1

    if progress: bar.close()
    if save is not None:
      np.save(f'{save}',vals)

    return vals

class sigweffmodel:
  def __init__(self, mrange, trange, mu, sg, lb, ea, eb, el, cache='load'):
    print('Initialising signal model')
    self.mrange = mrange
    self.trange = trange
    self.mu     = mu
    self.sg     = sg
    self.lb     = lb
    self.ea     = ea
    self.eb     = eb
    self.el     = el
    self.pars = { 'mrange': self.mrange,
                  'trange': self.trange,
                  'mu'    : self.mu,
                  'sg'    : self.sg,
                  'lb'    : self.lb,
                  'ea'    : self.ea,
                  'eb'    : self.eb,
                  'el'    : self.el
                 }
    # can directly make the pdfs here because there is no correlation
    self.mpdf = mynorm(*self.mrange, self.mu, self.sg)
    self.tpdf = myexp(*self.trange, self.lb)
    # get normalisations
    self.N      = 1
    self.eN     = 1
    self.N, self.Nerr   = nquad( self.pdf, (self.mrange, self.trange) )
    self.eN, self.eNerr = nquad( self.eff, (self.mrange, self.trange) )
    # get the majorant for toys
    f = lambda m, t: -self.pdf(m,t)
    mi = Minuit(f, m=self.mu, t=self.trange[0], limit_m=self.mrange, limit_t=self.trange, pedantic=False)
    mi.migrad()
    self.maj = -mi.fval
    print(' --> Majorant of {:4.2g} found at ({:7.2f},{:4.2f})'.format(self.maj, mi.values['m'], mi.values['t']))
    # vectorise projection pdfs
    self.pdfm   = np.vectorize(self._pdfm)
    self.pdft   = np.vectorize(self._pdft)
    self.effm   = np.vectorize(self._effm)
    self.efft   = np.vectorize(self._efft)
    if isinstance(cache,str):
      self.load_caches()
    elif isinstance(cache,int):
      self.make_caches(cache)
      self.load_caches()

  def make_caches(self, points):
    cache_dir = 'cache/sigweffmodel'
    print('Caching into', cache_dir)
    os.system(f'mkdir -p {cache_dir}')
    # save pars
    with open(f'{cache_dir}/pars.pkl','wb') as f:
      pickle.dump(self.pars,f)
    # compute arrays and save them
    m = np.linspace(*self.mrange,points)
    t = np.linspace(*self.trange,points)
    fm = self.pdfm(m)
    ft = self.pdft(t)
    em = self.effm(m)
    et = self.efft(t)
    np.savez(f'{cache_dir}/arrs.npz', m=m, fm=fm, em=em, t=t, ft=ft, et=et)

  def load_caches(self):
    cache_dir = 'cache/sigweffmodel'
    # read pars
    assert( os.path.exists(f'{cache_dir}/pars.pkl') )
    with open(f'{cache_dir}/pars.pkl','rb') as f:
      pars = pickle.load(f)
      for name, val in pars.items():
        assert( self.pars[name] == val )
    # read arrays
    npzfile = np.load(f'{cache_dir}/arrs.npz')
    m = npzfile['m']
    t = npzfile['t']
    fm = npzfile['fm']
    ft = npzfile['ft']
    em = npzfile['em']
    et = npzfile['et']
    self.pdfm = interp1d(m,fm,kind='quadratic')
    self.pdft = interp1d(t,ft,kind='quadratic')
    self.effm = interp1d(m,em,kind='quadratic')
    self.efft = interp1d(t,et,kind='quadratic')

  def eff(self, m, t):
    # get m into range 0-1
    mscl = ( m - self.mrange[0] ) / ( self.mrange[1] - self.mrange[0] )
    ascl = self.ea + self.eb*mscl
    f     = ascl * ( t - self.el ) ** ascl
    # scale it so that the max is 1. which happens at trange[1], mrange[1]
    mx = (self.ea+self.eb) * ( trange[1] - self.el ) ** (self.ea+self.eb)
    return f/mx

  def pdf(self, m, t, mproj=False, tproj=False):
    ############
    # eff part #
    ############

    eff = self.eff(m,t)

    ############
    # pdf part #
    ############
    if mproj: return eff*self.mpdf.pdf(m) / self.eN
    if tproj: return eff*self.tpdf.pdf(t) / self.eN
    return eff*self.mpdf.pdf(m)*self.tpdf.pdf(t) / self.N

  def _pdfm(self,m):
    f = lambda t: self.pdf(m,t)
    return quad(f,*trange)[0]

  def _pdft(self,t):
    f = lambda m: self.pdf(m,t)
    return quad(f,*mrange)[0]

  def _effm(self,m):
    f = lambda t: self.eff(m,t)
    return quad(f,*trange)[0]

  def _efft(self,t):
    f = lambda m: self.eff(m,t)
    return quad(f,*mrange)[0]

  # this is the very slow accept/reject method
  def generate(self, size=1, progress=None, save=None, seed=None):
    if progress is None:
      progress = False if size<10 else True

    if seed is not None:
      np.random.seed(seed)

    vals = np.empty((size,2))
    ngen = 0
    if progress: bar = tqdm(desc='Generating Signal    ', total=size)
    while ngen < size:
      accept = False
      m = None
      t = None
      while not accept:
        m = self.mpdf.rvs() #np.random.uniform(*mrange)
        t = self.tpdf.rvs() #np.random.uniform(*trange)
        h = np.random.uniform()
        if h < self.eff(m,t): accept=True

      vals[ngen] = (m,t)
      if progress: bar.update()
      ngen += 1

    if progress: bar.close()
    if save is not None:
      np.save(f'{save}',vals)

    return vals


def gensignal( smpdf, stpdf, size=1, save=None, seed=None ):
  if progress is None:
    progress = False if size<10 else True

  if seed is not None:
    np.random.seed(seed)

  vals =  np.column_stack( (smpdf.rvs(size=size), stpdf.rvs(size=size) ) )
  if save is not None:
    np.save(f'{save}',vals)

  return vals


mrange = (5000,5600)
mbins = 50
mmu = 5280
msg = 30
mlb = 400
trange = (0,10)
tbins = 50
tmu = -1
tsg = 2
tlb = 4
slb = 300
smu = 0.2
ssg = 0.8
ea  = 4
eb  = 0.2
el  = -3

# plot pdf projs
fig, ax = plt.subplots(1,2,figsize=(16,6))

# pdfs
spdf = sigweffmodel(mrange,trange,mmu,msg,tlb,0.5,0.2,-0.05,cache='load')
bpdf = bkgweffmodel(mrange,trange,mlb,tmu,tsg,slb,smu,ssg,0.5,0.2,-0.05,cache='load')

if opts.ntoys>0:
  for nt in range(opts.ntoys):
    print('Generating Toy', nt, '/', opts.ntoys )
    bkg_vals = bpdf.generate(size=opts.nbkg, seed=nt     , save='toys/nceffb_s%d_t%d.npy'%(opts.nbkg,nt))
    sig_vals = spdf.generate(size=opts.nsig, seed=1000+nt, save='toys/nceffs_s%d_t%d.npy'%(opts.nsig,nt))
  sys.exit('Generating Done.')

#bkg_vals = bpdf.generate(nbkg,save='toys/newcoweffbkg.npy')
#sig_vals = spdf.generate(nsig,save='toys/newcoweffsig.npy')
bkg_vals = np.load('toys/newcoweffbkg.npy')
sig_vals = np.load('toys/newcoweffsig.npy')
nbkg = len(bkg_vals)
nsig = len(sig_vals)
toy_vals = np.concatenate((bkg_vals,sig_vals))
#mhc, mhw = hist(bkg_vals[:,0], range=mrange, bins=mbins)
#thc, thw = hist(bkg_vals[:,1], range=trange, bins=tbins)
#mhc, mhw = hist(sig_vals[:,0], range=mrange, bins=mbins)
#thc, thw = hist(sig_vals[:,1], range=trange, bins=tbins)
mhc, mhw = hist(toy_vals[:,0], range=mrange, bins=mbins)
thc, thw = hist(toy_vals[:,1], range=trange, bins=tbins)

# m plot
m = np.linspace(*mrange,200)
#mBN = 1
#mSN = 1
mBN = nbkg*(mrange[1]-mrange[0])/mbins
mSN = nsig*(mrange[1]-mrange[0])/mbins
# plot the data
ax[0].errorbar( mhc, mhw, mhw**0.5, fmt='ko', label='Toy Data' )
# compute signal and background functions
ms = mSN*spdf.pdfm(m)
mb = mBN*bpdf.pdfm(m)
# plot the signal
ax[0].plot(m, ms, 'g--', label='True S pdf')
# plot the background
ax[0].plot(m, mb, 'r--', label='True B pdf')
# plot both
ax[0].plot(m, ms+mb, 'b-', label='True S+B pdf')
# plot variation
#for tval in np.linspace(*trange,3):
  #ax[0].plot(m, mBN*bpdf.pdf(m,tval,mproj=True), label=f'B t={tval}' )
# overlay the efficiency
mseff = spdf.effm(m)
mseff = mseff*np.max(ms+mb)/np.max(mseff)
ax[0].plot(m, mseff, c='0.8', ls='-', label='S Efficiency')
mbeff = bpdf.effm(m)
mbeff = mbeff*np.max(ms+mb)/np.max(mbeff)
ax[0].plot(m, mbeff, 'k:', label='B Efficiency')

ax[0].set_xlabel('Mass [GeV]')
ax[0].set_ylabel('Events')
ax[0].legend()

# t plot
t = np.linspace(*trange,200)
#tBN = 1
#tSN = 1
tBN = nbkg*(trange[1]-trange[0])/tbins
tSN = nsig*(trange[1]-trange[0])/tbins
# plot the data
ax[1].errorbar( thc, thw, thw**0.5, fmt='ko', label='Toy Data' )
# compute signal and background functions
ts = tSN*spdf.pdft(t)
tb = tBN*bpdf.pdft(t)
# plot the signal
ax[1].plot(t, ts, 'g--', label='True S pdf')
# plot the background
ax[1].plot(t, tb, 'r--', label='True B pdf')
# plot both
ax[1].plot(t, ts+tb, 'b-', label='True S+B pdf')
# plot variation
#for mval in np.linspace(*mrange,3):
  #ax[1].plot(t, tBN*bpdf.pdf(mval,t,tproj=True), label=f'B m={mval}')
# overlay the efficiency
tseff = spdf.efft(t)
tseff = tseff*np.max(ts+tb)/np.max(tseff)
ax[1].plot(t, tseff, c='0.8', ls='-', label='S Efficiency')
tbeff = bpdf.efft(t)
tbeff = tbeff*np.max(ts+tb)/np.max(tbeff)
ax[1].plot(t, tbeff, 'k:', label='B Efficiency')

ax[1].set_xlabel('Time [ps]')
ax[1].set_ylabel('Events')
ax[1].legend()
#ax[1].set_yscale('log')

fig.tight_layout()
fig.savefig('plots/newcoweff.pdf')

# 2D lad
fig, ax = plt.subplots(1,3,figsize=(18,4))
x,y = np.meshgrid(m,t)
ax[0].set_title('Efficiency')
cb1 = ax[0].contourf(x,y,bpdf.eff(x,y))
fig.colorbar(cb1,ax=ax[0]).set_label('Efficiency')
ax[0].set_xlabel('Mass [GeV]')
ax[0].set_ylabel('Time [ps]')

ax[1].set_title('True Background PDF')
cb2 = ax[1].contourf(x,y,bpdf.pdf(x,y))
fig.colorbar(cb2,ax=ax[1])
ax[1].set_xlabel('Mass [GeV]')
ax[1].set_ylabel('Time [ps]')

ax[2].set_title('True Signal PDF')
cb3 = ax[2].contourf(x,y,spdf.pdf(x,y))
fig.colorbar(cb3,ax=ax[2])
ax[2].set_xlabel('Mass [GeV]')
ax[2].set_ylabel('Time [ps]')

fig.tight_layout()
fig.savefig('plots/newcoweff2d.pdf')

#print( quad( bpdf.pdfm, *mrange ) )
#print( quad( bpdf.pdft, *trange ) )
#print( nquad( bpdf.pdf, (mrange,trange) ) )

plt.show()
