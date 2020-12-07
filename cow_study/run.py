from scipy.stats import norm, expon, rv_continuous
from scipy.special import binom
from scipy._lib._util import _lazywhere
from scipy import poly1d
from numpy.polynomial.chebyshev import Chebyshev as cheb
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit

# B(n,i)(x) = (n!)/(i!*(n-i)!) * (1-x)^(n-i) * x^i

def bernk(n,k,x):
  return binom(n,k) * (1-x)**(n-k) * x**k

def bern(x,*coeffs):
  n = len(coeffs)-1
  res = 0
  for i in range(len(coeffs)):
    res += coeffs[i] * bernk(n,i,x)
  return res

def bernI(x,*coeffs):
  n = len(coeffs)-1
  res = 0
  for i in range(n+1):
    term = 0
    for j in range(i,n+1):
      term += ((-1)**(j-i)) * binom(n,j) * binom(j,i) * (x**(j+1)) / (j+1)
    term *= coeffs[i]
    res += term
  return res

def bernN(*coeffs):
    return 1/bernI(1,*coeffs)

class bpoly1_gen(rv_continuous):
  def _pdf(self,x,p0,p1):
    N = bernN(p0,p1)
    return N * _lazywhere( (x>=0) & (x<=1) , (x,p0,p1), f=lambda x,p0,p1: bern(x,p0,p1), f2=lambda x,p0,p1: 0 )
  def _cdf(self,x,p0,p1):
    N = bernN(p0,p1)
    ret = N * _lazywhere( (x>=0) & (x<=1), (x,p0,p1), f=lambda x,p0,p1: bernI(x,p0,p1), f2=lambda x,p0,p1: 0 )
    ret[x>1] = 1
    return ret
bpoly1 = bpoly1_gen(name='bern1')

class bpoly2_gen(rv_continuous):
  def _pdf(self,x,p0,p1,p2):
    N = bernN(p0,p1,p2)
    return N * _lazywhere( (x>=0) & (x<=1) , (x,p0,p1,p2), f=lambda x,p0,p1,p2: bern(x,p0,p1,p2), f2=lambda x,p0,p1,p2: 0 )
  def _cdf(self,x,p0,p1,p2):
    N = bernN(p0,p1,p2)
    ret = N * _lazywhere( (x>=0) & (x<=1), (x,p0,p1,p2), f=lambda x,p0,p1,p2: bernI(x,p0,p1,p2), f2=lambda x,p0,p1,p2: 0 )
    ret[x>1] = 1
    return ret
bpoly2 = bpoly2_gen(name='bern2')

class bpoly3_gen(rv_continuous):
  def _pdf(self,x,p0,p1,p2,p3):
    N = bernN(p0,p1,p2,p3)
    return N * _lazywhere( (x>=0) & (x<=1) , (x,p0,p1,p2,p3), f=lambda x,p0,p1,p2,p3: bern(x,p0,p1,p2,p3), f2=lambda x,p0,p1,p2,p3: 0 )
  def _cdf(self,x,p0,p1,p2,p3):
    N = bernN(p0,p1,p2,p3)
    ret = N * _lazywhere( (x>=0) & (x<=1), (x,p0,p1,p2,p3), f=lambda x,p0,p1,p2,p3: bernI(x,p0,p1,p2,p3), f2=lambda x,p0,p1,p2,p3: 0 )
    ret[x>1] = 1
    return ret
bpoly3 = bpoly3_gen(name='bern3')

class bpoly4_gen(rv_continuous):
  def _pdf(self,x,p0,p1,p2,p3,p4):
    pars = [p0,p1,p2,p3,p4]
    xpars = [x]+pars
    N = bernN(*pars)
    return N * _lazywhere( (x>=0) & (x<=1) , xpars, f=lambda *xpars: bern(*xpars), f2=lambda *xpars: 0 )
  def _cdf(self,x,p0,p1,p2,p3,p4):
    pars = [p0,p1,p2,p3,p4]
    xpars = [x]+pars
    N = bernN(*pars)
    ret = N * _lazywhere( (x>=0) & (x<=1), xpars, f=lambda *xpars: bernI(*xpars), f2=lambda *xpars: 0 )
    ret[x>1] = 1
    return ret
bpoly4 = bpoly4_gen(name='bern4')

class bpoly5_gen(rv_continuous):
  def _pdf(self,x,p0,p1,p2,p3,p4,p5):
    pars = [p0,p1,p2,p3,p4,p5]
    xpars = [x]+pars
    N = bernN(*pars)
    return N * _lazywhere( (x>=0) & (x<=1) , xpars, f=lambda *xpars: bern(*xpars), f2=lambda *xpars: 0 )
  def _cdf(self,x,p0,p1,p2,p3,p4,p5):
    pars = [p0,p1,p2,p3,p4,p5]
    xpars = [x]+pars
    N = bernN(*pars)
    ret = N * _lazywhere( (x>=0) & (x<=1), xpars, f=lambda *xpars: bernI(*xpars), f2=lambda *xpars: 0 )
    ret[x>1] = 1
    return ret
bpoly5 = bpoly5_gen(name='bern5')

class bpoly6_gen(rv_continuous):
  def _pdf(self,x,p0,p1,p2,p3,p4,p5,p6):
    pars = [p0,p1,p2,p3,p4,p5,p6]
    xpars = [x]+pars
    N = bernN(*pars)
    return N * _lazywhere( (x>=0) & (x<=1) , xpars, f=lambda *xpars: bern(*xpars), f2=lambda *xpars: 0 )
  def _cdf(self,x,p0,p1,p2,p3,p4,p5,p6):
    pars = [p0,p1,p2,p3,p4,p5,p6]
    xpars = [x]+pars
    N = bernN(*pars)
    ret = N * _lazywhere( (x>=0) & (x<=1), xpars, f=lambda *xpars: bernI(*xpars), f2=lambda *xpars: 0 )
    ret[x>1] = 1
    return ret
bpoly6 = bpoly6_gen(name='bern6')

if __name__ == '__main__':
  mrange = (5000,5600)
  trange = (0,10)
  g0 = norm(5280,30)
  g1 = expon(5000,400)
  h0 = expon(0,2)
  h1 = norm(0,0.5)
  z0 = 0.5
  N = 10000

  import pandas as pd
  np.random.seed(210187)

  # generate toy
  generate = False
  fit = False
  if generate:
    data = pd.DataFrame(columns=['mass','time','ctrl'])

    n_sig = np.random.poisson(N*z0)
    n_bkg = np.random.poisson(N*(1-z0))

    ns = 0
    while ns < n_sig:
      m = g0.rvs()
      t = h0.rvs()
      if m > mrange[1] or m < mrange[0]: continue
      if t > trange[1] or t < trange[0]: continue
      data = data.append( {'mass': m, 'time': t, 'ctrl': 0}, ignore_index=True )
      ns += 1

    nb = 0
    while nb < n_bkg:
      m = g1.rvs()
      t = h1.rvs()
      if m > mrange[1] or m < mrange[0]: continue
      if t > trange[1] or t < trange[0]: continue
      data = data.append( {'mass': m, 'time': t, 'ctrl': 1}, ignore_index=True )
      nb += 1

    data = data.astype( {'mass': float, 'time': float, 'ctrl': int} )
    data.to_pickle('toy.pkl')

  else:
    data = pd.read_pickle('toy.pkl')

  # draw toy
  fig, ax = plt.subplots(1,2, figsize=(12,4))
  sig = data[data['ctrl']==0]
  bkg = data[data['ctrl']==1]

  ax[0].hist( [bkg['mass'],sig['mass']], bins=50, stacked=True )
  ax[1].hist( [bkg['time'],sig['time']], bins=50, stacked=True )
  ax[1].set_yscale('log')
  fig.tight_layout()

  def nll_exp(N, z, mu, sg, lb):
    s = norm(mu,sg)
    b = expon(mrange[0], lb)
    sn = np.diff( s.cdf(mrange) )
    bn = np.diff( b.cdf(mrange) )
    ns = N*z
    nb = N*(1-z)
    return N - np.sum( np.log ( s.pdf( data['mass'].to_numpy() ) / sn * ns + b.pdf( data['mass'].to_numpy() ) / bn * nb ) )

  def nll_poly(pars):
    N  = pars[0]
    z  = pars[1]
    mu = pars[2]
    sg = pars[3]
    s = norm(mu,sg)
    sn = np.diff( s.cdf(mrange) )
    n = len(pars[4:])
    if n==1: b  = bpoly1(1,pars[4],5000,600)
    if n==2: b  = bpoly2(1,pars[4],pars[5],5000,600)
    if n==3: b  = bpoly3(1,pars[4],pars[5],pars[6],5000,600)
    if n==4: b  = bpoly4(1,pars[4],pars[5],pars[6],pars[7],5000,600)

    bn = np.diff( b.cdf(mrange) )
    ns = N*z
    nb = N*(1-z)
    return N - np.sum( np.log ( s.pdf( data['mass'].to_numpy() ) / sn * ns + b.pdf ( data['mass'].to_numpy() ) / bn * nb ) )

  mi = Minuit( nll_exp, N=N, z=z0, mu=5280, sg=30, lb=400, errordef=0.5, pedantic=False )

  if fit:
    mi.migrad()
    mi.hesse()
    print(mi.params)

  mips =  [ Minuit.from_array_func( nll_poly, (N,z0,5280,30,0.5)            , name=('N','z','mu','sg','p1')               , limit=((0.5*N,1.5*N),(0,1),(5200,5300),(0,50),(0,1)), errordef=0.5, pedantic=False ),
            Minuit.from_array_func( nll_poly, (N,z0,5280,30,0.5,0.5)        , name=('N','z','mu','sg','p1','p2')          , limit=((0.5*N,1.5*N),(0,1),(5200,5300),(0,50),(0,1),(0,1)), errordef=0.5, pedantic=False ),
            Minuit.from_array_func( nll_poly, (N,z0,5280,30,0.5,0.5,0.5)    , name=('N','z','mu','sg','p1','p2','p3')     , limit=((0.5*N,1.5*N),(0,1),(5200,5300),(0,50),(0,1),(0,1),(0,1)), errordef=0.5, pedantic=False ),
            Minuit.from_array_func( nll_poly, (N,z0,5280,30,0.5,0.5,0.5,0.5), name=('N','z','mu','sg','p1','p2','p3','p4'), limit=((0.5*N,1.5*N),(0,1),(5200,5300),(0,50),(0,1),(0,1),(0,1),(0,1)), errordef=0.5, pedantic=False ) ]

  if fit:
    for mip in mips:
      mip.migrad()
      mip.hesse()
      print(mip.params)

# draw pdf
  def pdf(m, bonly=False, sonly=False, poly=None):

    values = mi.values if poly is None else mips[poly-1].values

    mu = values['mu']
    sg = values['sg']
    spdf = norm( mu, sg )
    sn   = np.diff(spdf.cdf(mrange))

    if poly is None:
      bpdf = expon( 5000, mi.values['lb'] )
      bn   = np.diff(bpdf.cdf(mrange))
    else:
      assert( type(poly)==int )
      args = [ values['p%d'%(p+1)] for p in range(poly) ]
      if poly == 1: bpdf = bpoly1(1,*args,5000,600)
      if poly == 2: bpdf = bpoly2(1,*args,5000,600)
      if poly == 3: bpdf = bpoly3(1,*args,5000,600)
      if poly == 4: bpdf = bpoly4(1,*args,5000,600)
      bn   = np.diff(bpdf.cdf(mrange))

    ns   = values['N']*values['z']
    nb   = values['N']*(1-values['z'])
    sr = spdf.pdf(m)
    br = bpdf.pdf(m)
    if sonly: return ns * sr / sn
    if bonly: return nb * br / bn
    return ns * sr / sn + nb * br / bn

  fig = plt.figure()
  ax = fig.gca()
  bins = 50
  pn = (mrange[1]-mrange[0])/bins
  x = np.linspace(*mrange,400)

  w, xe = np.histogram( data['mass'].to_numpy(), bins=50, range=mrange )
  cx = 0.5 * (xe[1:] + xe[:-1] )

  ax.errorbar( cx, w, w**0.5, fmt='ko')
  #ax.plot( x, pn*pdf(x,bonly=True), 'r--')
  ax.plot( x, pn*pdf(x))
  #ax.plot( x, pn*pdf(x,bonly=True,poly=1), 'g--')
  ax.plot( x, pn*pdf(x,poly=1))
  #ax.plot( x, pn*pdf(x,bonly=True,poly=2), 'g--')
  ax.plot( x, pn*pdf(x,poly=2))
  ax.plot( x, pn*pdf(x,poly=3))
  ax.plot( x, pn*pdf(x,poly=4))

  fig.tight_layout()
  plt.show()

