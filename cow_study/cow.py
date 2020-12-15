from scipy.stats import uniform
from scipy.integrate import quad
from scipy import linalg
import numpy as np

class cow():
  def __init__(self, mrange, gs, gb, Im=1, obs=None):
    '''
    mrange: a two element tuple for the integration range in mass
    gs:     a function for the signal (numerator) must accept a single argument in this case mass
    gb:     a function or list of functions for the backgrond (numerator) must accept a single argument in this case mass
    Im:     a function to evaluate the I(m) (denominator) must accept a single argument in this case mass (default is uniform 1)
    obs:    you can pass the observed distribution to evaluate Im instead. this expects the weights and bin edges in a two element tuple
            like the return value of np.histogram
    '''
    self.mrange = mrange
    self.gs = gs
    self.gb = gb if hasattr(gb,'__iter__') else [gb]
    self.gk = [self.gs] + self.gb
    self.Im = Im
    self.obs = obs
    if obs:
      if len(obs)!=2: raise ValueError('The observation must be passed as length two object containing weights and bin edges (w,xe) - ie. what is return by np.histogram()')
      w, xe = obs
      if len(w)!=len(xe)-1: raise ValueError('The bin edges and weights do not have the right respective dimensions')
      # normalise
      w = w/np.sum(w)
      f = lambda m: w[ np.argmin( m >= xe )-1 ]
      self.Im = np.vectorize(f)
    if self.Im == 1:
      un = uniform(*mrange)
      n  = np.diff( un.cdf(mrange) )
      self.Im = lambda m: un.pdf(m) / n

  def WklElem(self, k, l):
    # check it's available in m
    assert( k < len(self.gk) )
    assert( l < len(self.gk) )

    def integral(m):
      return self.gk[k](m) * self.gk[l](m) / self.Im(m)

    if self.obs is None:
      return quad( integral, self.mrange[0], self.mrange[1] )[0]
    else:
      tint = 0
      xe = self.obs[1]
      for le, he in zip(xe[:-1],xe[1:]):
        tint += quad( integral, le, he )[0]
      return tint

  def Wkl(self):

    n = len(self.gk)

    ret = np.identity(n)

    for i in range(n):
      for j in range(i,n):
        ret[i,j] = self.WklElem(i,j)

    # symmetrise and cache
    self._Wkl = ret + ret.T - np.diag( ret.diagonal() )
    return self._Wkl

  def Akl(self, use_cache=False):
    if not use_cache: self.Wkl()
    self._Akl = linalg.inv( self._Wkl )
    return self._Akl

  def wk(self, k, m, use_cache=True):
    if not use_cache or not hasattr(self,'_Akl'): self.Akl()

    n = len(self.gk)

    return np.sum( [ self._Akl[k,l] * self.gk[l](m) / self.Im(m) for l in range(n) ], axis=0 )



