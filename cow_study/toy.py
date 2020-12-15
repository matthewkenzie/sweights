import numpy as np
from scipy.stats import norm, expon
from scipy.stats import multivariate_normal as mvnorm
from scipy.integrate import quad, nquad
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class toy():
  def __init__(self, eff=None, sfact=True, bfact=True):
    if not sfact: raise RuntimeError('Must have factorising signal')
    self.eff = eff or 1
    if eff == 'flat' or eff == 'uniform': self.eff = 1
    if eff == 'factorising' or eff == 'fact': self.eff = 2
    if eff == 'non-factorising' or eff == 'nonfact': self.eff = 3
    if self.eff not in [1,2,3]: raise RuntimeError('Not a valid efficiency option')
    self.sfact = sfact
    self.bfact = bfact
    self.mrange = (5000,5600)
    self.trange = (0,10)
    self.pars = {
        'g0mu': 5280,
        'g0sg': 30,
        'g1lb': 400,
        'h0lb': 2,
        'h1mu': 0,
        'h1sg': 1.5,
        'z0': 0.5,
        'g1mu': 4000,
        'g1sg': 600,
        'g1rh': -0.2
        }
    self._fmt = lambda m, t, so, bo: self.fmtbase(m,t,so,bo)
    self._ftm = lambda t, m, so, bo: self.fmtbase(m,t,so,bo)
    self._fm = lambda m, so, bo: quad( self._ftm, *self.trange, args=(m,so,bo) )[0]
    self._ft = lambda t, so, bo: quad( self._fmt, *self.mrange, args=(t,so,bo) )[0]
    self._fmv = np.vectorize(self._fm, excluded=['so','bo'])
    self._ftv = np.vectorize(self._ft, excluded=['so','bo'])
    self.setmodel()

  def setmodel(self):

    # fractions
    self.z0 = self.pars['z0']
    self.z1 = 1-self.pars['z0']

    # signal
    self.g0  = norm( self.pars['g0mu'],self.pars['g0sg'] )
    self.g0n = np.diff( self.g0.cdf(self.mrange) )
    self.h0  = expon( self.trange[0], self.pars['h0lb'] )
    self.h0n = np.diff( self.h0.cdf(self.trange) )

    # background
    self.g1  = expon(self.mrange[0], self.pars['g1lb'])
    self.g1n = np.diff( self.g1.cdf(self.mrange) )
    self.h1  = norm( self.trange[0], self.pars['h1sg'])
    self.h1n = np.diff( self.h1.cdf(self.trange) )

    # background non-factorising
    self.f1  = mvnorm( [ self.pars['g1mu'], self.pars['h1mu'] ] , \
        [ [ self.pars['g1sg']**2, self.pars['g1rh']*self.pars['g1sg']*self.pars['h1sg'] ],
          [ self.pars['g1rh']*self.pars['g1sg']*self.pars['h1sg'], self.pars['h1sg']**2 ] ] )

    f1_f = lambda m, t: self.f1.pdf( (m,t) )
    f1_v = np.vectorize(f1_f)
    self.f1n = nquad( f1_v, (self.mrange,self.trange) )[0]

  # the idea is we have a generic function for f(m,t) that can be evaluated
  # for just m or just t, for signal or background only
  def fmtbase(self, m=None, t=None, sonly=False, bonly=False):

    if not self.bfact and (m is None or t is None): raise RuntimeError('Can\'t do this')
    if not self.sfact and (m is None or t is None): raise RuntimeError('Can\'t do this')

    sm = self.g0.pdf(m)/self.g0n if m is not None else 1
    st = self.h0.pdf(t)/self.h0n if t is not None else 1
    bm = self.g1.pdf(m)/self.g1n if m is not None else 1
    bt = self.h1.pdf(t)/self.h1n if t is not None else 1

    s = sm*st
    b = bm*bt
    if not self.bfact: b = self.f1.pdf( np.dstack((m,t)) ) / self.f1n

    if sonly: return self.z0*s
    if bonly: return self.z1*b
    return self.z0*s + self.z1*b

  def fmt(self, m=None, t=None, sonly=False, bonly=False):
    if self.sfact and self.bfact: return self.fmtbase(m,t,sonly,bonly)
    elif not self.bfact:
      if m is None:   return self._ftv(t, sonly, bonly)
      elif t is None: return self._fmv(m, sonly, bonly)

  def effmap(self, m, t, a=1.2, b=0.5, loc=-0.5):
    # get m into range 0-1
    mscl = (m - self.mrange[0])/(self.mrange[1]-self.mrange[0])
    ascl = a + b*mscl
    f = ascl*(t-loc)**(ascl-1.)
    # scale it so that the max is 1. which happens at trange[1], mrange[1]
    mx = (a+b)*(self.trange[1]-loc)**(a+b-1.)
    return f/mx

  def effmt(self, m, t):
    if   self.eff==1: return np.ones( m.shape )
    elif self.eff==2: return self.effmap(m,t,b=0)
    elif self.eff==3: return self.effmap(m,t)

  def rhomt(self, m=None, t=None, sonly=False, bonly=False):
    # efficiency is flat
    if self.eff==1: return self.fmt(m,t,sonly,bonly)
    # efficiency factorises
    elif self.eff==2:
      if m is None:   return self.effmap(m=self.mrange[0],t=t,b=0) * self.fmt(m,t,sonly,bonly)
      elif t is None: return self.effmap(m=m,t=self.trange[0],b=0) * self.fmt(m,t,sonly,bonly)
      else:           return self.effmap(m,t,b=0) * self.fmt(m,t,sonly,bonly)
    # efficiency does not factorise so need to integrate
    elif self.eff==3:
      rmt = lambda m, t, so, bo: self.effmap(m,t) * self.fmtbase(m,t,so,bo)
      rtm = lambda t, m, so, bo: self.effmap(m,t) * self.fmtbase(m,t,so,bo)
      rm = lambda m, so, bo: quad( rtm, *self.trange, args=(m,so,bo) )[0]
      rt = lambda t, so, bo: quad( rmt, *self.mrange, args=(t,so,bo) )[0]
      rmv = np.vectorize(rm, excluded=['so','bo'])
      rtv = np.vectorize(rt, excluded=['so','bo'])

      if m is None:   return rtv(t,sonly,bonly)
      elif t is None: return rmv(m,sonly,bonly)
      else:           return self.effmap(m,t)     * self.fmtbase(m,t,sonly,bonly)

  def draw(self, name=None, withtoy=False):

    print('Plotting. If this is slow it\'s because it has to perform a lot of integrations')
    fig, ax = plt.subplots(2,3,figsize=(18,8))
    m = np.linspace(*self.mrange,100)
    t = np.linspace(*self.trange,100)
    x,y = np.meshgrid(m,t)

    # model
    ax[0,0].set_title('Truth model: f(m,t)')
    cb1 = ax[0,0].contourf( x, y, self.fmtbase(x,y) )
    fig.colorbar(cb1,ax=ax[0,0])

    # efficiency
    ax[1,0].set_title('Efficiency map')
    cb2 = ax[1,0].contourf( x, y, self.effmt(x,y), levels=np.linspace(0,1,11) )
    fig.colorbar(cb2,ax=ax[1,0])

    # m proj
    ax[0,1].set_title('Truth model projection in mass')
    ax[0,1].plot( m, self.fmt(m=m,t=None,bonly=True), 'r--', label='Background' )
    ax[0,1].plot( m, self.fmt(m=m,t=None) , 'b-' , label='Signal+Background')
    ax[0,1].legend()
    ax[0,1].set_xlabel('mass')

    # t proj
    ax[1,1].set_title('Truth model projection in time')
    ax[1,1].plot( t, self.fmt(m=None,t=t,bonly=True), 'r--', label='Background' )
    ax[1,1].plot( t, self.fmt(m=None,t=t) , 'b-' , label='Signal+Background')
    ax[1,1].legend()
    ax[1,1].set_xlabel('time')

    # m proj w/ eff
    norm = 1
    if withtoy and hasattr(self,'toy'):
      bins = 50
      w, xe = np.histogram( self.toy['mass'], bins=bins, range=self.mrange )
      norm = np.sum(w) * np.diff(self.mrange)/bins
      if self.eff!=1: norm /= quad(lambda m: self.rhomt(m=m, t=None),*self.mrange)[0]
      cx = 0.5 * (xe[1:]+xe[:-1])
      ax[0,2].errorbar( cx, w, w**0.5, fmt='ko', ms=4, capsize=2, label='Toy Data' )
    ax[0,2].set_title('Truth model with efficiency projection in mass')
    ax[0,2].plot( m, norm*self.rhomt(m=m,t=None,bonly=True), 'r--', label='Background' )
    ax[0,2].plot( m, norm*self.rhomt(m=m,t=None) , 'b-' , label='Signal+Background')
    ax[0,2].legend()
    ax[0,2].set_xlabel('mass')

    # t proj w/ eff
    norm = 1
    if withtoy and hasattr(self,'toy'):
      bins = 50
      w, xe = np.histogram( self.toy['time'], bins=bins, range=self.trange )
      norm = np.sum(w) * np.diff(self.trange)/bins
      if self.eff!=1: norm /= quad(lambda t: self.rhomt(m=None, t=t),*self.trange)[0]
      cx = 0.5 * (xe[1:]+xe[:-1])
      ax[1,2].errorbar( cx, w, w**0.5, fmt='ko', ms=4, capsize=2, label='Toy Data' )
    ax[1,2].set_title('Truth model with efficiency projection in time')
    ax[1,2].plot( t, norm*self.rhomt(m=None,t=t,bonly=True), 'r--', label='Background' )
    ax[1,2].plot( t, norm*self.rhomt(m=None,t=t) , 'b-' , label='Signal+Background')
    ax[1,2].legend()
    ax[1,2].set_xlabel('time')

    fig.suptitle('Toy generation model')
    fig.tight_layout()
    if name is not None:
      fig.savefig(name)

  def generate(self,size=1,fname='toy.pkl'):

    self.toy = pd.DataFrame(columns=['mass','time','ctrl'])

    nsig = np.random.poisson( size*self.pars['z0'] )
    nbkg = np.random.poisson( size*(1-self.pars['z0']) )

    pbar = tqdm( total=(nsig+nbkg), desc='Generating Toy', ascii=True )

    # generate the signal
    ns = 0
    while ns < nsig:
      m = None
      t = None
      if self.sfact:
        m = self.g0.rvs()
        t = self.h0.rvs()
      if m > self.mrange[1] or m < self.mrange[0]: continue
      if t > self.trange[1] or t < self.trange[0]: continue
      if self.eff!=1:
        if np.random.uniform() > self.effmt(m,t): continue

      self.toy = self.toy.append( {'mass': m, 'time': t, 'ctrl': 0}, ignore_index=True )
      ns += 1
      pbar.update()

    # generate the signal
    nb = 0
    while nb < nbkg:
      m = None
      t = None
      if self.bfact:
        m = self.g1.rvs()
        t = self.h1.rvs()
      else:
        m, t = self.f1.rvs()

      if m > self.mrange[1] or m < self.mrange[0]: continue
      if t > self.trange[1] or t < self.trange[0]: continue
      if self.eff!=1:
        if np.random.uniform() > self.effmt(m,t): continue

      self.toy = self.toy.append( {'mass': m, 'time': t, 'ctrl': 1}, ignore_index=True )
      nb += 1
      pbar.update()

    pbar.close()

    self.toy = self.toy.astype( {'mass': float, 'time': float, 'ctrl': int} )

    self.toy.to_pickle(fname)
    return self.toy


#for eff in [1,2,3]:
  #t = toy(eff=eff)
  #t.draw(name='plots/gen%d.pdf'%eff)
#t = toy(eff=3 )
#t.generate(size=1000,fname='toy.pkl')
#t.draw(withtoy=True)

#plt.show()
