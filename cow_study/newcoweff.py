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

class bkgweffmodel:
  def __init__(self,mrange, trange, lb, mu, sg, slb, smu, ssg, ea, eb, el,cache='load'):
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
    #f    = ascl * ( t - self.el ) ** (ascl - 1.)
    f     = ascl * ( t - self.el ) ** ascl
    # scale it so that the max is 1. which happens at trange[1], mrange[1]
    #mx = (self.ea+self.eb)*(self.trange[1]-self.el)**(self.ea+self.eb-1.)
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
ea  = 4
eb  = 0.2
el  = -3

# plot pdf projs
fig, ax = plt.subplots(1,2,figsize=(16,6))

# pdfs
smpdf = mynorm(*mrange,mmu, msg)
stpdf = myexp(*trange,tlb)
#bpdf = bkgweffmodel(mrange,trange,mlb,tmu,tsg,slb,smu,ssg,0.5,0.2,-0.05,cache='load')
bpdf = bkgweffmodel(mrange,trange,mlb,tmu,tsg,slb,smu,ssg,0.5,0.2,-0.05,cache=400)

nbkg = 1000
nsig = 800
bkg_vals = bpdf.generate(nbkg,save='toys/newcoweffbkg.npy')
#sig_vals = gensignal(smpdf,stpdf,nsig,save='toys/newcowsig.npy')
bkg_vals = np.load('toys/newcoweffbkg.npy')
sig_vals = np.load('toys/newcowsig.npy')
nbkg = len(bkg_vals)
nsig = len(sig_vals)
toy_vals = np.concatenate((bkg_vals,sig_vals))
mhc, mhw = hist(toy_vals[:,0], range=mrange, bins=mbins)
thc, thw = hist(toy_vals[:,1], range=trange, bins=tbins)

# m plot
m = np.linspace(*mrange,200)
#mBN = 1
#mSN = 1
mBN = nbkg*(mrange[1]-mrange[0])/mbins
mSN = nsig*(mrange[1]-mrange[0])/mbins
#ax[0].plot(m, bpdf.effm(m))
ax[0].errorbar( mhc, mhw, mhw**0.5, fmt='ko' )
#ax[0].plot(m, mBN*smpdf.pdf(m), label='S pdf' )
#for tval in np.linspace(*trange,3):
  #ax[0].plot(m, mBN*bpdf.pdf(m,tval,mproj=True), label=f'B t={tval}' )
ax[0].plot(m, mBN*bpdf.pdfm(m), 'r--', label='B pdf')
ax[0].plot(m, mBN*bpdf.pdfm(m)+mSN*smpdf.pdf(m), 'b-', label='S+B pdf')


ax[0].legend()

# t plot
t = np.linspace(*trange,200)
#tBN = 1
#tSN = 1
tBN = nbkg*(trange[1]-trange[0])/tbins
tSN = nsig*(trange[1]-trange[0])/tbins
#ax[1].plot(t, bpdf.efft(t))
ax[1].errorbar( thc, thw, thw**0.5, fmt='ko' )
#ax[1].plot(t, tSN*stpdf.pdf(t), label='S pdf' )
#for mval in np.linspace(*mrange,3):
  #ax[1].plot(t, tBN*bpdf.pdf(mval,t,tproj=True), label=f'B m={mval}')
ax[1].plot(t, tBN*bpdf.pdft(t), 'r--', label='B pdf')
ax[1].plot(t, tBN*bpdf.pdft(t)+tSN*stpdf.pdf(t), 'b-', label='S+B pdf')
ax[1].legend()
#ax[1].set_yscale('log')
fig.tight_layout()
fig.savefig('plots/newcoweff.pdf')

# 2D lad
fig, ax = plt.subplots(1,2,figsize=(16,6))
x,y = np.meshgrid(m,t)
cb1 = ax[0].contourf(x,y,bpdf.eff(x,y))
fig.colorbar(cb1,ax=ax[0])
cb2 = ax[1].contourf(x,y,bpdf.pdf(x,y))
fig.colorbar(cb2,ax=ax[1])
fig.tight_layout()

# eff
fig, ax = plt.subplots(1,2,figsize=(16,6))
ax[0].plot(m, bpdf.effm(m) )
for tval in np.linspace(*trange,3):
  ax[0].plot(m, bpdf.eff(m,tval), label=f'eff t={tval}' )
ax[0].set_ylim( (0, ax[0].get_ylim()[1] ) )
ax[1].plot(t, bpdf.efft(t) )
for mval in np.linspace(*mrange,3):
  ax[1].plot(t, bpdf.eff(mval,t), label=f'eff m={mval}')
ax[1].set_ylim( (0, ax[1].get_ylim()[1] ) )
fig.tight_layout()

#print( quad( bpdf.pdfm, *mrange ) )
#print( quad( bpdf.pdft, *trange ) )
#print( nquad( bpdf.pdf, (mrange,trange) ) )

plt.show()
