import os
import numpy as np
import pandas as pd
from scipy.stats import norm, expon
from scipy.stats import multivariate_normal as mvnorm
from scipy.integrate import quad, nquad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tabulate import tabulate
from tqdm import tqdm

class model:
  def __init__(self, eff='flat', bfact=True, load_cache=True):
    # some stuff
    self.mrange = (5000,5600)
    self.trange = (0,10)

    self.eff   = eff # 'fact', 'nonfact'
    if self.eff not in ['flat','fact','nonfact']:
      raise NotImplementedError('%s is an unsupported efficiency'%self.eff)

    self.bfact = bfact
    # always assume signal factorises

    # fractions
    self.z0 = 0.5
    self.z1 = 1-self.z0

    # signal
    self.g0 = norm( 5280, 30 )
    self.g0n = np.diff( self.g0.cdf(self.mrange) )
    self.h0 = expon(0, 2)
    self.h0n = np.diff( self.h0.cdf(self.trange) )

    # background
    self.g1 = expon( 5000, 400 )
    self.g1n = np.diff( self.g1.cdf(self.mrange) )
    self.h1 = norm(0, 1.5)
    self.h1n = np.diff( self.h1.cdf(self.trange) )
    self.gh1 = mvnorm( [ 4000, 0 ], [ [600**2, -0.2*600*1.5], [-0.2*600*1.5, 1.5**2] ] )
    gh1_f = lambda m, t: self.gh1.pdf( (m,t) )
    gh1_v = np.vectorize( gh1_f )
    self.gh1n = nquad( gh1_v, (self.mrange,self.trange) )[0]

    # load projection cache
    if not load_cache: self.cache_fmt()
    self.load_cache()

  # rho(m,t) = 1/D eff(m,t) * f(m,t)

  def effmap(self, m, t, a=1.2, b=0.5, loc=-0.5):
    # get m into range 0-1
    mscl = (m - self.mrange[0])/(self.mrange[1]-self.mrange[0])
    ascl = a + b*mscl
    f = ascl*(t-loc)**(ascl-1.)
    # scale it so that the max is 1. which happens at trange[1], mrange[1]
    mx = (a+b)*(self.trange[1]-loc)**(a+b-1.)
    return f/mx

  def effmt(self,m,t):
    if   self.eff=='flat'   :
      if isinstance(m,float): return 1
      else: return np.ones( m.shape )
    elif self.eff=='fact'   : return self.effmap(m,t,b=0)
    elif self.eff=='nonfact': return self.effmap(m,t)
    else: raise RuntimeError('Not a valid efficiency option')

  def chfm(self, m, so=False, bo=False):
    sm = self.z0*self.g0.pdf(m) / self.g0n
    bm = self.z1*self.g1.pdf(m) / self.g1n
    if so: return sm
    if bo: return bm
    return sm+bm

  def chft(self, t, so=False, bo=False):
    st = self.z0*self.h0.pdf(t) / self.h0n
    bt = self.z1*self.h1.pdf(t) / self.h1n
    if so: return st
    if bo: return bt
    return st+bt

  def fmt(self,m,t, so=False, bo=False):
    sm = self.g0.pdf(m) / self.g0n
    st = self.h0.pdf(t) / self.h0n
    s  = self.z0*sm*st

    if self.bfact:
      bm = self.g1.pdf(m) / self.g1n
      bt = self.h1.pdf(t) / self.h1n
      b = self.z1*bm*bt
    else:
      b = self.z1*self.gh1.pdf( np.dstack((m,t)) ) / self.gh1n

    if so: return s
    if bo: return b
    return s+b

  def ftm(self,t,m,so,bo):
    return self.fmt(m,t,so,bo)

  def _fm(self,m,so=False,bo=False):
    return quad( self.ftm, *self.trange, args=(m,so,bo) )[0]

  def _ft(self,t,so=False,bo=False):
    return quad( self.fmt, *self.mrange, args=(t,so,bo) )[0]

  def rmt_unnorm(self,m,t):
    return self.effmt(m,t) * self.fmt(m,t)

  def rhomt(self,m,t,so=False,bo=False):
    return self.effmt(m,t) * self.fmt(m,t,so,bo) / self.N

  def rhotm(self,t,m,so,bo):
    return self.rhomt(m,t,so,bo)

  def _rhom(self,m,so=False,bo=False):
    return quad( self.rhotm, *self.trange, args=(m,so,bo) )[0]

  def _rhot(self,t,so=False,bo=False):
    return quad( self.rhomt, *self.mrange, args=(t,so,bo) )[0]

  def cache_rmt_norm(self):
    N = nquad( self.rmt_unnorm, (self.mrange,self.trange) )[0]
    os.system('mkdir -p cache')
    bname = 'fact' if self.bfact else 'nonfact'
    fname = 'cache/rmt_b%s_e%s_norm.npy'%(bname, self.eff)
    np.save(fname, N)

  def cache_fmt(self):
    print('Caching through the snow, f(m,t)')
    os.system('mkdir -p cache')
    m = np.linspace(*self.mrange,400)
    t = np.linspace(*self.trange,400)
    sm = np.array([ self._fm(mval,so=True) for mval in m ])
    bm = np.array([ self._fm(mval,bo=True) for mval in m ])
    fm = np.array([ self._fm(mval) for mval in m ])
    st = np.array([ self._ft(tval,so=True) for tval in t ])
    bt = np.array([ self._ft(tval,bo=True) for tval in t ])
    ft = np.array([ self._ft(tval) for tval in t ])
    fname = 'cache/fmt_bfact.npz'
    if not self.bfact: fname.replace('bfact','bnonfact')
    np.savez(fname, m=m, t=t, sm=sm, bm=bm, fm=fm, st=st, bt=bt, ft=ft )

  def cache_rhomt(self):
    print('Caching through the snow, rho(m,t)')
    os.system('mkdir -p cache')
    m = np.linspace(*self.mrange,400)
    t = np.linspace(*self.trange,400)
    rsm = np.array([ self._rhom(mval,so=True) for mval in m ])
    rbm = np.array([ self._rhom(mval,bo=True) for mval in m ])
    rm = np.array([ self._rhom(mval) for mval in m ])
    rst = np.array([ self._rhot(tval,so=True) for tval in t ])
    rbt = np.array([ self._rhot(tval,bo=True) for tval in t ])
    rt = np.array([ self._rhot(tval) for tval in t ])
    bname = 'fact' if self.bfact else 'nonfact'
    fname = 'cache/rmt_b%s_e%s.npz'%(bname, self.eff)
    np.savez(fname, m=m, t=t, rsm=rsm, rbm=rbm, rm=rm, rst=rst, rbt=rbt, rt=rt )

  def load_cache(self):
    bname = 'fact' if self.bfact else 'nonfact'

    # rmt N cache
    fname = 'cache/rmt_b%s_e%s_norm.npy'%(bname, self.eff)
    if os.path.exists(fname):
      self.N = np.load(fname)
    else:
      print('Cache file', fname, 'not found so trying to produce it')
      self.cache_rmt_norm()
      self.load_cache()

    # fmt cache
    fname = 'cache/fmt_b%s.npz'%bname
    if os.path.exists(fname):
      fmtfil = np.load(fname)
      m  = fmtfil['m']
      t  = fmtfil['t']
      sm = fmtfil['sm']
      bm = fmtfil['bm']
      fm = fmtfil['fm']
      st = fmtfil['st']
      bt = fmtfil['bt']
      ft = fmtfil['ft']

      self.sm = interp1d( m, sm, kind='cubic', bounds_error=False, fill_value=0 )
      self.bm = interp1d( m, bm, kind='cubic', bounds_error=False, fill_value=0 )
      self.fm = interp1d( m, fm, kind='cubic', bounds_error=False, fill_value=0 )

      self.st = interp1d( t, st, kind='cubic', bounds_error=False, fill_value='extrapolate')
      self.bt = interp1d( t, bt, kind='cubic', bounds_error=False, fill_value='extrapolate')
      self.ft = interp1d( t, ft, kind='cubic', bounds_error=False, fill_value='extrapolate')
    else:
      print('Cache file', fname, 'not found so trying to produce it')
      self.cache_fmt()
      self.load_cache()

    # rhomt cache
    fname = 'cache/rmt_b%s_e%s.npz'%(bname, self.eff)
    if os.path.exists(fname):
      fmtfil = np.load(fname)
      m  = fmtfil['m']
      t  = fmtfil['t']
      rsm = fmtfil['rsm']
      rbm = fmtfil['rbm']
      rm  = fmtfil['rm']
      rst = fmtfil['rst']
      rbt = fmtfil['rbt']
      rt  = fmtfil['rt']

      self.rsm = interp1d( m, rsm, kind='cubic', bounds_error=False, fill_value=0 )
      self.rbm = interp1d( m, rbm, kind='cubic', bounds_error=False, fill_value=0 )
      self.rm  = interp1d( m, rm , kind='cubic', bounds_error=False, fill_value=0 )

      self.rst = interp1d( t, rst, kind='cubic', bounds_error=False, fill_value='extrapolate')
      self.rbt = interp1d( t, rbt, kind='cubic', bounds_error=False, fill_value='extrapolate')
      self.rt  = interp1d( t, rt , kind='cubic', bounds_error=False, fill_value='extrapolate')
    else:
      print('Cache file', fname, 'not found so trying to produce it')
      self.cache_rhomt()
      self.load_cache()

  def check(self):
    rows = []
    rows.append( [ 'rho(m,t)', nquad( self.rhomt, (self.mrange,self.trange) )[0] ] )
    rows.append( [ 'rho(m)'  , quad ( self.rm   , *self.mrange )[0] ] )
    rows.append( [ 'rho(t)'  , quad ( self.rt   , *self.trange )[0] ] )
    print( tabulate(rows, headers=['func','integral']) )

  def generate(self, size=1, poisson=True, seed=None, fname='toy.pkl', eff_force=False):

    np.random.seed(seed)

    self.toy = pd.DataFrame(columns=['mass','time','ctrl'])

    nsig = np.random.poisson( size*self.z0 ) if poisson else size*self.z0
    nbkg = np.random.poisson( size*self.z1 ) if poisson else size*self.z1

    pbar = tqdm( total=(nsig+nbkg), desc='Generating Toy', ascii=True )

    # generate the signal
    ns = 0
    while ns < nsig:
      m = self.g0.rvs()
      t = self.h0.rvs()
      if m > self.mrange[1] or m < self.mrange[0]: continue
      if t > self.trange[1] or t < self.trange[0]: continue
      if self.eff!='flat' and np.random.uniform() > self.effmt(m,t) and not eff_force: continue
      self.toy = self.toy.append( {'mass': m, 'time': t, 'ctrl': 0}, ignore_index=True )
      ns += 1
      pbar.update()

    # generate the background
    nb = 0
    while nb < nbkg:
      if self.bfact:
        m = self.g1.rvs()
        t = self.h1.rvs()
      else:
        m, t = self.gh1.rvs()
      if m > self.mrange[1] or m < self.mrange[0]: continue
      if t > self.trange[1] or t < self.trange[0]: continue
      if self.eff!='flat' and np.random.uniform() > self.effmt(m,t) and not eff_force: continue
      self.toy = self.toy.append( {'mass': m, 'time': t, 'ctrl': 1}, ignore_index=True )
      nb += 1
      pbar.update()

    pbar.close()
    self.toy = self.toy.astype( {'mass': float, 'time': float, 'ctrl': int} )
    self.toy.to_pickle(fname)

  def draw(self, save=None, with_toy=None, fig=None, ax=None):
    m = np.linspace(*self.mrange,100)
    t = np.linspace(*self.trange,100)
    x, y = np.meshgrid(m,t)

    if with_toy is not None:
      self.toy = pd.read_pickle(with_toy)

    # toy data if it's there
    mn = 1
    tn = 1
    if with_toy and hasattr(self,'toy'):
      mw, me = np.histogram( self.toy['mass'], bins=50, range=self.mrange )
      msw, _ = np.histogram( self.toy[self.toy['ctrl']==0]['mass'], bins=50, range=self.mrange )
      mbw, _ = np.histogram( self.toy[self.toy['ctrl']==1]['mass'], bins=50, range=self.mrange )
      mc = 0.5 * (me[1:]+me[:-1])
      mn = np.sum(mw) * np.diff(self.mrange) / 50
      tw, te = np.histogram( self.toy['time'], bins=50, range=self.trange )
      tsw, _ = np.histogram( self.toy[self.toy['ctrl']==0]['time'], bins=50, range=self.trange )
      tbw, _ = np.histogram( self.toy[self.toy['ctrl']==1]['time'], bins=50, range=self.trange )
      tc = 0.5 * (te[1:]+te[:-1])
      tn = np.sum(tw) * np.diff(self.trange) / 50
      print( mn, np.sum(mw), len(self.toy), np.sum(msw), np.sum(mbw) )
      print( tn, np.sum(tw), len(self.toy), np.sum(tsw), np.sum(tbw) )

    if fig is None or ax is None:
      fig, ax = plt.subplots(2,3,figsize=(18,8))

    cb1 = ax[0,0].contourf( x, y, self.fmt(x,y) )
    fig.colorbar(cb1, ax=ax[0,0])
    ax[0,0].set_title('True PDF: $f(m,t)$')
    ax[0,0].set_xlabel('mass')
    ax[0,0].set_ylabel('time')

    cb2 = ax[1,0].contourf( x, y, self.effmt(x,y), levels=np.linspace(0,1,11) )
    fig.colorbar(cb2, ax=ax[1,0]).set_label('Efficiency')
    ax[1,0].set_title('Efficiency Map: $\epsilon(m,t)$')
    ax[1,0].set_xlabel('mass')
    ax[1,0].set_ylabel('time')

    ax[0,1].plot(m,self.sm(m),'b--', label='Signal')
    ax[0,1].plot(m,self.bm(m),'r--', label='Background')
    ax[0,1].plot(m,self.fm(m),'k-' , label='Both')
    ax[0,1].legend()
    ax[0,1].set_title('True PDF $m$ projection: $f(m)$')
    ax[0,1].set_xlabel('mass')

    ax[0,2].plot(m,mn*self.rsm(m),'b--', label='Signal')
    ax[0,2].plot(m,mn*self.rbm(m),'r--', label='Background')
    ax[0,2].plot(m,mn*self.rm(m) ,'k-' , label='Both')
    if with_toy and hasattr(self,'toy'):
      ax[0,2].errorbar( mc, msw, msw**0.5, fmt='b+', ms=4, capsize=2, label='Toy Data' )
      ax[0,2].errorbar( mc, mbw, mbw**0.5, fmt='rx', ms=4, capsize=2, label='Toy Data' )
      ax[0,2].errorbar( mc, mw, mw**0.5, fmt='ko', ms=4, capsize=2, label='Toy Data' )
    ax[0,2].legend()
    ax[0,2].set_title(r'Observed PDF $m$ projection: $\rho(m)$')
    ax[0,2].set_xlabel('mass')

    ax[1,1].plot(t,self.st(t),'b--', label='Signal')
    ax[1,1].plot(t,self.bt(t),'r--', label='Background')
    ax[1,1].plot(t,self.ft(t),'k-' , label='Both')
    ax[1,1].legend()
    ax[1,1].set_title('True PDF $t$ projection: $f(t)$')
    ax[1,1].set_xlabel('time')

    ax[1,2].plot(t,tn*self.rst(t),'b--', label='Signal')
    ax[1,2].plot(t,tn*self.rbt(t),'r--', label='Background')
    ax[1,2].plot(t,tn*self.rt(t) ,'k-' , label='Both')
    if with_toy and hasattr(self,'toy'):
      ax[1,2].errorbar( tc, tsw, tsw**0.5, fmt='b+', ms=4, capsize=2, label='Toy Data' )
      ax[1,2].errorbar( tc, tbw, tbw**0.5, fmt='rx', ms=4, capsize=2, label='Toy Data' )
      ax[1,2].errorbar( tc, tw, tw**0.5, fmt='ko', ms=4, capsize=2, label='Toy Data' )
    ax[1,2].legend()
    ax[1,2].set_title(r'Observed PDF $t$ projection: $\rho(t)$')
    ax[1,2].set_xlabel('time')

    bname = 'factorising' if self.bfact else 'non-factorising'
    fig.suptitle('Model. Efficiency: %s. Background: %s'%(self.eff,bname))
    fig.tight_layout()
    if save is not None: fig.savefig(save)

#os.system('mkdir -p plots')
#for bfact in [True,False]:
  ##for eff in ['flat','fact','nonfact']:
  #for eff in ['nonfact']:
    #mod = model( eff=eff, bfact=bfact )
    #bname = 'fact' if bfact else 'nonfact'

    #size = 10000
    #tname = 'toys/toy_e%s_b%s_s%d.pdf'%(eff,bname,size)
    #mod.generate(size=size, poisson=False, seed=210187, fname=tname)

    #pname = 'plots/model_e%s_b%s.pdf'%(eff,bname)
    #mod.draw( save=pname, with_toy=tname )
    ##mod.check()

#plt.show()
