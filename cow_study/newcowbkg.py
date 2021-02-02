import numpy as np
import matplotlib.pyplot as plt
import time
import os
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

class bkgmodel:
  def __init__(self,mrange, trange, lb, mu, sg, slb, smu, ssg, cache=None):
    print('Initialising background model')
    self.mrange = mrange
    self.trange = trange
    self.lb     = lb
    self.mu     = mu
    self.sg     = sg
    self.slb    = slb
    self.smu    = smu
    self.ssg    = ssg
    self.pars = { 'mrange': self.mrange,
                  'trange': self.trange,
                  'lb'    : self.lb,
                  'mu'    : self.mu,
                  'sg'    : self.sg,
                  'slb'   : self.slb,
                  'smu'   : self.smu,
                  'ssg'   : self.ssg
                 }
    self.N      = 1
    # get normalisation
    self.N, self.Nerr = nquad( self.pdf, (self.mrange, self.trange) )
    # get the majorant for toys
    f = lambda m, t: -self.pdf(m,t)
    mi = Minuit(f, m=self.mrange[0], t=self.trange[0], limit_m=self.mrange, limit_t=self.trange, pedantic=False)
    mi.migrad()
    self.maj = -mi.fval
    # vectorise projection pdfs
    self.pdfm   = np.vectorize(self._pdfm)
    self.pdft   = np.vectorize(self._pdft)
    if isinstance(cache,str):
      self.load_caches()
    elif isinstance(cache,int):
      self.make_caches(cache)
      self.load_caches()

  def make_caches(self, points):
    cache_dir = 'cache/bkgmodel'
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
    np.savez(f'{cache_dir}/arrs.npz', m=m, fm=fm, t=t, ft=ft)

  def load_caches(self):
    cache_dir = 'cache/bkgmodel'
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
    self.pdfm = interp1d(m,fm,kind='quadratic')
    self.pdft = interp1d(t,ft,kind='quadratic')

  def pdf(self, m, t, mproj=False, tproj=False):
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

    if mproj: return mpdf.pdf(m)
    if tproj: return tpdf.pdf(t)
    return mpdf.pdf(m)*tpdf.pdf(t) / self.N

  def _pdfm(self,m):
    f = lambda t: self.pdf(m,t)
    return quad(f,*trange)[0]

  def _pdft(self,t):
    f = lambda m: self.pdf(m,t)
    return quad(f,*mrange)[0]

  # this is the very slow accept/reject method
  def generate(self, size=1, progress=None, save=None, seed=None):
    if progress is None:
      progress = False if size<10 else True

    if seed is not None:
      np.random.seed(seed)

    vals = np.empty((size,2))
    ngen = 0
    if progress: bar = tqdm(total=size)
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
      bar.update()
      ngen += 1

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

# plot pdf projs
fig, ax = plt.subplots(1,2,figsize=(16,6))

# pdfs
smpdf = mynorm(*mrange,mmu, msg)
stpdf = myexp(*trange,tlb)
bpdf = bkgmodel(mrange,trange,mlb,tmu,tsg,slb,smu,ssg,cache='load')

nbkg = 1000
nsig = 800
#bkg_vals = bpdf.generate(nbkg,save='toys/newcowbkg.npy')
#sig_vals = gensignal(smpdf,stpdf,nsig,save='toys/newcowsig.npy')
bkg_vals = np.load('toys/newcowbkg.npy')
sig_vals = np.load('toys/newcowsig.npy')
toy_vals = np.concatenate((bkg_vals,sig_vals))
mhc, mhw = hist(toy_vals[:,0], range=mrange, bins=mbins)
thc, thw = hist(toy_vals[:,1], range=trange, bins=tbins)

# m plot
m = np.linspace(*mrange,200)
mBN = nbkg*(mrange[1]-mrange[0])/mbins
mSN = nsig*(mrange[1]-mrange[0])/mbins
ax[0].errorbar( mhc, mhw, mhw**0.5, fmt='ko' )
#ax[0].plot(m, smpdf.pdf(m), label='S pdf' )
for tval in np.linspace(*trange,3):
  ax[0].plot(m, mBN*bpdf.pdf(m,tval,mproj=True), label=f'B t={tval}' )
ax[0].plot(m, mBN*bpdf.pdfm(m), 'r--', label='B pdf')
ax[0].plot(m, mBN*bpdf.pdfm(m)+mSN*smpdf.pdf(m), 'b-', label='S+B pdf')

ax[0].legend()

# t plot
t = np.linspace(*trange,200)
tBN = nbkg*(trange[1]-trange[0])/tbins
tSN = nsig*(trange[1]-trange[0])/tbins
ax[1].errorbar( thc, thw, thw**0.5, fmt='ko' )
#ax[1].plot(t, stpdf.pdf(t), label='S pdf' )
for mval in np.linspace(*mrange,3):
  ax[1].plot(t, tBN*bpdf.pdf(mval,t,tproj=True), label=f'B m={mval}')
ax[1].plot(t, tBN*bpdf.pdft(t), 'r--', label='B pdf')
ax[1].plot(t, tBN*bpdf.pdft(t)+tSN*stpdf.pdf(t), 'b-', label='S+B pdf')
ax[1].legend()
ax[1].set_yscale('log')
fig.tight_layout()

# 2D lad
fig, ax = plt.subplots(1,1,figsize=(8,6))
x,y = np.meshgrid(m,t)
ax.contourf(x,y,bpdf.pdf(x,y))

#print( quad( bpdf.pdfm, *mrange ) )
#print( quad( bpdf.pdft, *trange ) )
#print( nquad( bpdf.pdf, (mrange,trange) ) )


plt.show()
