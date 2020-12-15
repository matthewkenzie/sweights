import numpy as np
from scipy.stats import norm, expon
from scipy.stats import multivariate_normal as mvnorm
from scipy.integrate import quad, nquad
import matplotlib.pyplot as plt

class model:
  def __init__(self):
    # some stuff
    self.mrange = (5000,5600)
    self.trange = (0,10)

    self.eff   = 'flat' # 'fact', 'nonfact'
    self.bfact = True
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
    self.gh1 = mvnorm( [ 5000, 0 ], [ [400**2, -0.2*400*1.5], [-0.2*400*1.5, 1.5**2] ] )
    gh1_f = lambda m, t: self.gh1.pdf( (m,t) )
    gh1_v = np.vectorize( gh1_f )
    self.gh1n = nquad( gh1_v, (self.mrange,self.trange) )[0]

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
    if   self.eff=='flat'   : return np.ones( m.shape )
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

  def fm(self,m,so=False,bo=False):
    return quad( self.ftm, *self.trange, args=(m,so,bo) )[0]

  def ft(self,t,so=False,bo=False):
    return quad( self.fmt, *self.mrange, args=(t,so,bo) )[0]

  def checkfmt(self):
    m = np.linspace(*self.mrange,100)
    t = np.linspace(*self.trange,100)

    cfig, cax = plt.subplots()
    x, y = np.meshgrid(m,t)
    cax.contourf( x, y, self.fmt(x,y) )

    fig, ax = plt.subplots(2,2,figsize=(12,8))

    sm = [ self.fm(mval,so=True) for mval in m ]
    bm = [ self.fm(mval,bo=True) for mval in m ]
    ym = [ self.fm(mval) for mval in m ]
    st = [ self.ft(tval,so=True) for tval in t ]
    bt = [ self.ft(tval,bo=True) for tval in t ]
    yt = [ self.ft(tval) for tval in t ]

    ax[0,0].plot(m,sm,'r--')
    ax[0,0].plot(m,bm,'b--')
    ax[0,0].plot(m,ym,'k-')
    ax[0,1].plot(m,self.chfm(m,so=True),'r--')
    ax[0,1].plot(m,self.chfm(m,bo=True),'b--')
    ax[0,1].plot(m,self.chfm(m),'k-')

    ax[1,0].plot(t,st,'r--')
    ax[1,0].plot(t,bt,'b--')
    ax[1,0].plot(t,yt,'k-')
    ax[1,1].plot(t,self.chft(t,so=True),'r--')
    ax[1,1].plot(t,self.chft(t,bo=True),'b--')
    ax[1,1].plot(t,self.chft(t),'k-')

    plt.show()

mod = model()
mod.checkfmt()
